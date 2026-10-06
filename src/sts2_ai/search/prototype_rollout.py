from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from sts2_ai.emulator import EmulatorBackend, InformationPolicy, LegalAction, Observation, StateHandle
from sts2_ai.search.base import ActionEvaluation, SearchBudget, SearchResult


@dataclass(slots=True)
class _Accumulator:
    visits: int = 0
    total: float = 0.0
    total_squared: float = 0.0

    def add(self, value: float) -> None:
        self.visits += 1
        self.total += value
        self.total_squared += value * value

    def mean(self) -> float:
        return self.total / self.visits if self.visits else float("-inf")

    def standard_error(self) -> float | None:
        if self.visits < 2:
            return None
        mean = self.total / self.visits
        variance = max(0.0, (self.total_squared / self.visits) - (mean * mean))
        return math.sqrt(variance / self.visits)


class PrototypeFlatRolloutSearch:
    """Small deterministic baseline that exercises the real whole-run emulator.

    This is deliberately not a claim of strategic strength. Each root action receives
    round-robin random rollouts, and leaf states are scored with a simple progress/HP
    heuristic over the fair observation.
    """

    search_version = "prototype-flat-rollout-v0"

    def __init__(
        self,
        backend: EmulatorBackend,
        policy: InformationPolicy,
        *,
        seed: int = 0,
        rollout_depth: int = 64,
        leaf_value: Callable[[Observation], float] | None = None,
    ) -> None:
        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive")
        self._backend = backend
        self._policy = policy
        self._rng = random.Random(seed)
        self._rollout_depth = rollout_depth
        self._leaf_value = leaf_value or self._prototype_leaf_value

    def search(self, state: StateHandle, budget: SearchBudget) -> SearchResult:
        root_hash = self._backend.exact_hash(state)
        root_actions = tuple(
            sorted(self._backend.legal_actions(state), key=lambda action: action.action_id)
        )
        if not root_actions:
            return SearchResult(
                root_state_hash=root_hash,
                evaluations=(),
                expanded_nodes=0,
                search_version=self.search_version,
            )

        max_nodes = budget.max_nodes if budget.max_nodes is not None else 256
        if max_nodes < 0:
            raise ValueError("max_nodes cannot be negative")

        deadline = (
            time.monotonic() + budget.max_seconds
            if budget.max_seconds is not None
            else None
        )
        if budget.max_seconds is not None and budget.max_seconds < 0:
            raise ValueError("max_seconds cannot be negative")

        accumulators = {action.action_id: _Accumulator() for action in root_actions}
        expanded_nodes = 0
        action_index = 0

        while expanded_nodes < max_nodes:
            if deadline is not None and time.monotonic() >= deadline:
                break

            action = root_actions[action_index % len(root_actions)]
            action_index += 1

            remaining = max_nodes - expanded_nodes
            if remaining <= 0:
                break

            value, used = self._rollout(state, action, remaining, deadline)
            if used == 0:
                break

            expanded_nodes += used
            accumulators[action.action_id].add(value)

        evaluations = tuple(
            ActionEvaluation(
                action=action,
                value=accumulators[action.action_id].mean(),
                visits=accumulators[action.action_id].visits,
                uncertainty=accumulators[action.action_id].standard_error(),
            )
            for action in root_actions
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
        remaining_nodes: int,
        deadline: float | None,
    ) -> tuple[float, int]:
        state = self._backend.fork(root)
        transition = self._backend.step(state, root_action)
        state = transition.child
        used = 1

        while (
            used < remaining_nodes
            and used < self._rollout_depth
            and not transition.terminal
        ):
            if deadline is not None and time.monotonic() >= deadline:
                break

            actions = tuple(
                sorted(
                    self._backend.legal_actions(state),
                    key=lambda action: action.action_id,
                )
            )
            if not actions:
                break

            action = self._rng.choice(actions)
            transition = self._backend.step(state, action)
            state = transition.child
            used += 1

        observation = self._backend.observe(state, self._policy)
        return self._leaf_value(observation), used

    @staticmethod
    def _prototype_leaf_value(observation: Observation) -> float:
        payload = json.loads(observation.payload_json)
        if not isinstance(payload, dict):
            raise RuntimeError("Prototype observation payload must be an object")

        terminal = payload.get("terminalOutcome")
        act = float(payload.get("act") or 0)
        floor = float(payload.get("floor") or 0)
        hp = float(payload.get("hp") or 0)
        max_hp = max(1.0, float(payload.get("maxHp") or 1))
        gold = float(payload.get("gold") or 0)
        deck = payload.get("deck")
        deck_size = float(len(deck)) if isinstance(deck, list) else 0.0

        score = (act * 500.0) + (floor * 50.0)
        score += (hp / max_hp) * 100.0
        score += gold * 0.05
        score += deck_size * 0.25

        if terminal == "victory":
            score += 10_000.0
        elif terminal == "defeat":
            score -= 10_000.0

        return score
