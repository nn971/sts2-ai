from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass

from sts2_ai.emulator import EmulatorBackend, InformationPolicy, LegalAction, StateHandle

from .base import ActionEvaluation, SearchBudget, SearchResult


@dataclass(slots=True)
class _Accumulator:
    action: LegalAction
    total_value: float = 0.0
    visits: int = 0


class MonteCarloRolloutSearch:
    """Small deterministic rollout baseline for validating real emulator search plumbing."""

    search_version = "mc-rollout-v0"

    def __init__(
        self,
        backend: EmulatorBackend,
        *,
        seed: int = 0,
        information_policy: InformationPolicy | None = None,
        max_rollout_depth: int = 512,
    ) -> None:
        if max_rollout_depth <= 0:
            raise ValueError("max_rollout_depth must be positive")
        self._backend = backend
        self._seed = seed
        self._information_policy = information_policy or InformationPolicy("prototype-fair-v0")
        self._max_rollout_depth = max_rollout_depth

    def search(self, state: StateHandle, budget: SearchBudget) -> SearchResult:
        root_actions = tuple(self._backend.legal_actions(state))
        root_hash = self._backend.exact_hash(state)
        if not root_actions:
            return SearchResult(
                root_state_hash=root_hash,
                evaluations=(),
                expanded_nodes=0,
                search_version=self.search_version,
            )

        max_nodes = budget.max_nodes if budget.max_nodes is not None else 1_000
        if max_nodes <= 0:
            raise ValueError("SearchBudget.max_nodes must be positive when provided")
        if budget.max_seconds is not None and budget.max_seconds <= 0:
            raise ValueError("SearchBudget.max_seconds must be positive when provided")

        accumulators = [_Accumulator(action) for action in root_actions]
        rng = random.Random(self._seed)
        expanded_nodes = 0
        started = time.monotonic()
        next_action_index = 0

        while expanded_nodes < max_nodes:
            if self._time_exhausted(started, budget):
                break

            accumulator = accumulators[next_action_index]
            next_action_index = (next_action_index + 1) % len(accumulators)

            value, used_nodes = self._rollout(
                state,
                accumulator.action,
                rng,
                max_nodes - expanded_nodes,
                started,
                budget,
            )
            if used_nodes == 0:
                break

            accumulator.total_value += value
            accumulator.visits += 1
            expanded_nodes += used_nodes

        evaluations = tuple(
            ActionEvaluation(
                action=item.action,
                value=item.total_value / item.visits if item.visits else float("-inf"),
                visits=item.visits,
            )
            for item in accumulators
        )
        return SearchResult(
            root_state_hash=root_hash,
            evaluations=evaluations,
            expanded_nodes=expanded_nodes,
            search_version=self.search_version,
        )

    def _rollout(
        self,
        root: StateHandle,
        root_action: LegalAction,
        rng: random.Random,
        remaining_nodes: int,
        started: float,
        budget: SearchBudget,
    ) -> tuple[float, int]:
        if remaining_nodes <= 0 or self._time_exhausted(started, budget):
            return (self._evaluate(root), 0)

        created: list[StateHandle] = []
        used_nodes = 0
        try:
            branch = self._backend.fork(root)
            created.append(branch)

            transition = self._backend.step(branch, root_action)
            current = transition.child
            created.append(current)
            used_nodes += 1
            terminal = transition.terminal
            depth = 1

            while (
                not terminal
                and depth < self._max_rollout_depth
                and used_nodes < remaining_nodes
                and not self._time_exhausted(started, budget)
            ):
                actions = tuple(self._backend.legal_actions(current))
                if not actions:
                    break
                action = rng.choice(actions)
                transition = self._backend.step(current, action)
                current = transition.child
                created.append(current)
                used_nodes += 1
                depth += 1
                terminal = transition.terminal

            return (self._evaluate(current), used_nodes)
        finally:
            if created:
                self._backend.release_many(created)

    def _evaluate(self, state: StateHandle) -> float:
        observation = self._backend.observe(state, self._information_policy)
        payload = json.loads(observation.payload_json)
        if not isinstance(payload, dict):
            raise RuntimeError("Emulator observation payload is not a JSON object")

        outcome = payload.get("terminalOutcome")
        if outcome == "victory":
            return 1.0
        if outcome == "defeat":
            return -1.0

        hp = payload.get("hp")
        max_hp = payload.get("maxHp")
        act = payload.get("act")
        floor = payload.get("floor")

        hp_ratio = (
            float(hp) / float(max_hp)
            if isinstance(hp, int) and isinstance(max_hp, int) and max_hp > 0
            else 0.0
        )
        progress = 0.0
        if isinstance(act, int):
            progress += max(0.0, min(1.0, (act - 1) / 3.0))
        if isinstance(floor, int):
            progress += max(0.0, min(1.0, floor / 18.0))

        # Deliberately weak fallback: terminal outcomes should dominate once rollouts become deep.
        return min(0.95, (0.15 * progress) + (0.10 * hp_ratio))

    @staticmethod
    def _time_exhausted(started: float, budget: SearchBudget) -> bool:
        return (
            budget.max_seconds is not None
            and time.monotonic() - started >= budget.max_seconds
        )
