from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from sts2_ai.agents import Agent, Decision, ExactStateAgent
from sts2_ai.emulator import (
    EmulatorBackend,
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
)
from sts2_ai.emulator.run_environment import (
    LEGACY,
    cumulative_floor_progress,
    reset_training_run,
)
from sts2_ai.evaluation.boss_progress import BossProgress, BossProgressTracker


@dataclass(frozen=True, slots=True)
class RunSummary:
    seed: str
    outcome: str
    terminal_act: int | None
    terminal_floor: int | None
    decisions: int
    emulator_transitions: int
    wall_seconds: float
    agent_compute_seconds: float
    frontier_progress: float
    frontier_enemy_hp: int | None
    hp_trajectory: tuple[int, ...]
    # Optional defaults preserve compatibility with historical serialized runs.
    act1_cleared: bool | None = None
    full_game_victory: bool | None = None
    censored: bool = False
    episode_goal_version: str = "prototype-three-act-v0"
    environment: str = LEGACY
    boss_progress: BossProgress | None = None

    @property
    def won(self) -> bool:
        return self.outcome == "victory"

    @property
    def terminal_progress(self) -> float:
        result, _ = cumulative_floor_progress(
            self.terminal_act, self.terminal_floor, self.environment
        )
        return result

    @property
    def transitions_per_decision(self) -> float:
        if self.decisions == 0:
            return 0.0
        return self.emulator_transitions / self.decisions


def play_run(
    backend: EmulatorBackend,
    agent: Agent | ExactStateAgent,
    *,
    seed: str,
    policy: InformationPolicy,
    ascension: int = 0,
    max_decisions: int | None = None,
    environment: str = LEGACY,
    decision_observer: (
        Callable[[StateHandle, Observation, tuple[LegalAction, ...], Decision], None]
        | None
    ) = None,
) -> RunSummary:
    """Drive one complete emulator run while keeping only the live state handle."""

    started = time.perf_counter()
    state = reset_training_run(backend, seed, environment, ascension)
    hp_trajectory: list[int] = []
    decisions = 0
    emulator_transitions = 0
    agent_compute_seconds = 0.0
    final_state: dict[str, Any] = {}
    preterminal_state: dict[str, Any] | None = None
    boss_tracker = BossProgressTracker(environment)

    try:
        while True:
            observation = backend.observe(state, policy)
            raw = json.loads(observation.payload_json)
            current = cast(dict[str, Any], raw if isinstance(raw, dict) else {})
            final_state = current
            boss_tracker.observe(current)
            hp = current.get("hp")
            if isinstance(hp, int):
                hp_trajectory.append(hp)

            if backend.is_terminal(state):
                break
            if max_decisions is not None and decisions >= max_decisions:
                break

            preterminal_state = current
            legal_actions = tuple(backend.legal_actions(state))
            choose_started = time.perf_counter()
            if isinstance(agent, ExactStateAgent):
                decision = agent.choose_state(state, observation, legal_actions)
            else:
                decision = agent.choose(observation, legal_actions)
            agent_compute_seconds += time.perf_counter() - choose_started
            # Optional passive research hook while the exact root handle is live.
            # Its extra work is excluded from the agent compute/transition totals.
            if decision_observer is not None:
                decision_observer(state, observation, legal_actions, decision)

            transition = backend.step(state, decision.action)
            emulator_transitions += 1 + _search_transitions(decision.metadata_json)
            old_state = state
            state = transition.child
            backend.release_many([old_state])
            decisions += 1

        terminal = backend.is_terminal(state)
        outcome_raw = final_state.get("terminal_outcome")
        outcome = (
            "truncated"
            if not terminal
            else (str(outcome_raw) if isinstance(outcome_raw, str) else "unknown")
        )
        act = final_state.get("act")
        floor = final_state.get("floor")
        frontier_state = (
            preterminal_state
            if outcome == "defeat" and preterminal_state is not None
            else final_state
        )
        return RunSummary(
            seed=seed,
            outcome=outcome,
            terminal_act=act if isinstance(act, int) else None,
            terminal_floor=floor if isinstance(floor, int) else None,
            decisions=decisions,
            emulator_transitions=emulator_transitions,
            wall_seconds=time.perf_counter() - started,
            agent_compute_seconds=agent_compute_seconds,
            frontier_progress=_continuous_progress(frontier_state, environment),
            frontier_enemy_hp=_remaining_enemy_hp(frontier_state),
            hp_trajectory=tuple(hp_trajectory),
            # Reaching Act 2 certifies an Act-1 clear even if the run
            # later truncates; an unfinished Act-1 run remains unknown.
            act1_cleared=(
                True if (isinstance(act, int) and act >= 2) or outcome == "victory"
                else False if outcome == "defeat"
                else None
            ),
            full_game_victory=(
                True if outcome == "victory"
                else False if outcome == "defeat"
                else None
            ),
            censored=outcome not in ("victory", "defeat"),
            environment=environment,
            boss_progress=boss_tracker.result(
                act1_cleared=(
                    outcome == "victory"
                    or (isinstance(act, int) and act >= 2)
                ),
            ),
        )
    finally:
        backend.release_many([state])


def _search_transitions(metadata_json: str) -> int:
    try:
        raw = json.loads(metadata_json)
    except json.JSONDecodeError:
        return 0
    if not isinstance(raw, dict):
        return 0
    value = raw.get("search_transitions")
    return value if isinstance(value, int) and value >= 0 else 0


def _continuous_progress(
    state: dict[str, Any], environment: str = LEGACY
) -> float:
    """Geometry-aware progress, plus within-combat fraction of a room.

    Native-structure mode uses 16 rooms in Act 1 but still six-room
    prototype Acts 2/3. The old benchmark keeps its 6/6/6 geometry.
    """
    act = int(_number(state.get("act"), 1.0))
    floor = int(_number(state.get("floor"), 0.0))
    progress, _ = cumulative_floor_progress(act, floor, environment)
    enemy_hp = _remaining_enemy_hp(state)
    if enemy_hp is not None and floor > 0:
        import math

        completion = math.exp(-enemy_hp / 80.0)
        return max(0.0, progress - 1.0 + completion)
    return progress


def _remaining_enemy_hp(state: dict[str, Any]) -> int | None:
    combat = state.get("combat")
    if not isinstance(combat, dict):
        return None
    enemies = combat.get("enemies")
    if not isinstance(enemies, list):
        return None

    total = 0
    found = False
    for raw_enemy in enemies:
        if not isinstance(raw_enemy, dict):
            continue
        hp = raw_enemy.get("hp")
        if isinstance(hp, int):
            total += max(0, hp)
            found = True
    return total if found else None


def _number(value: object, default: float) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default
