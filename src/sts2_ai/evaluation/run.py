from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, cast

from sts2_ai.agents import Agent, ExactStateAgent
from sts2_ai.emulator import EmulatorBackend, InformationPolicy


@dataclass(frozen=True, slots=True)
class RunSummary:
    seed: str
    outcome: str
    terminal_act: int | None
    terminal_floor: int | None
    decisions: int
    emulator_transitions: int
    wall_seconds: float
    hp_trajectory: tuple[int, ...]

    @property
    def won(self) -> bool:
        return self.outcome == "victory"

    @property
    def terminal_progress(self) -> float:
        if self.terminal_act is None:
            return float(self.terminal_floor or 0)
        return float(((self.terminal_act - 1) * 20) + (self.terminal_floor or 0))

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
) -> RunSummary:
    """Drive one complete emulator run while keeping only the live state handle."""

    started = time.perf_counter()
    state = backend.reset(seed, ascension)
    hp_trajectory: list[int] = []
    decisions = 0
    emulator_transitions = 0
    final_state: dict[str, Any] = {}

    try:
        while True:
            observation = backend.observe(state, policy)
            raw = json.loads(observation.payload_json)
            current = cast(dict[str, Any], raw if isinstance(raw, dict) else {})
            final_state = current
            hp = current.get("hp")
            if isinstance(hp, int):
                hp_trajectory.append(hp)

            if backend.is_terminal(state):
                break
            if max_decisions is not None and decisions >= max_decisions:
                break

            legal_actions = tuple(backend.legal_actions(state))
            if isinstance(agent, ExactStateAgent):
                decision = agent.choose_state(state, observation, legal_actions)
            else:
                decision = agent.choose(observation, legal_actions)

            transition = backend.step(state, decision.action)
            emulator_transitions += 1 + _search_transitions(decision.metadata_json)
            old_state = state
            state = transition.child
            backend.release_many([old_state])
            decisions += 1

        terminal = backend.is_terminal(state)
        outcome_raw = final_state.get("terminal_outcome")
        outcome = (
            str(outcome_raw)
            if isinstance(outcome_raw, str)
            else ("truncated" if not terminal else "unknown")
        )
        act = final_state.get("act")
        floor = final_state.get("floor")
        return RunSummary(
            seed=seed,
            outcome=outcome,
            terminal_act=act if isinstance(act, int) else None,
            terminal_floor=floor if isinstance(floor, int) else None,
            decisions=decisions,
            emulator_transitions=emulator_transitions,
            wall_seconds=time.perf_counter() - started,
            hp_trajectory=tuple(hp_trajectory),
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
