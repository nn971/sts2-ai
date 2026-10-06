from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from sts2_ai.emulator import (
    EmulatorBackend,
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
)
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


@dataclass(slots=True)
class _Rollout:
    root_action: LegalAction
    state: StateHandle
    terminal: bool
    used_nodes: int


class PrototypeFlatRolloutSearch:
    """Small deterministic baseline that exercises the real whole-run emulator.

    This is deliberately not a claim of strategic strength. Root actions receive
    round-robin random rollouts, while independent rollout states advance in batched
    wavefronts to reduce transport overhead. Leaf states are scored by an injectable
    evaluator; the default is a simple fair-observation progress/HP heuristic.
    """

    search_version = "prototype-flat-rollout-v2"

    def __init__(
        self,
        backend: EmulatorBackend,
        policy: InformationPolicy,
        *,
        seed: int = 0,
        rollout_depth: int = 64,
        rollout_batch_size: int = 16,
        leaf_value: Callable[[Observation], float] | None = None,
    ) -> None:
        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive")
        if rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")

        self._backend = backend
        self._policy = policy
        self._seed = seed
        self._rollout_depth = rollout_depth
        self._rollout_batch_size = rollout_batch_size
        self._leaf_value = leaf_value or self._prototype_leaf_value

    def search(self, state: StateHandle, budget: SearchBudget) -> SearchResult:
        root_hash = self._backend.exact_hash(state)
        rng = random.Random(self._state_seed(root_hash))
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
        if budget.max_seconds is not None and budget.max_seconds < 0:
            raise ValueError("max_seconds cannot be negative")

        deadline = (
            time.monotonic() + budget.max_seconds
            if budget.max_seconds is not None
            else None
        )
        accumulators = {action.action_id: _Accumulator() for action in root_actions}
        expanded_nodes = 0
        next_root_action = 0

        while expanded_nodes < max_nodes:
            if deadline is not None and time.monotonic() >= deadline:
                break

            wave_size = min(self._rollout_batch_size, max_nodes - expanded_nodes)
            assigned_actions = tuple(
                root_actions[(next_root_action + index) % len(root_actions)]
                for index in range(wave_size)
            )
            next_root_action += wave_size

            root_transitions = tuple(
                self._backend.batch_step(
                    tuple((state, action) for action in assigned_actions)
                )
            )
            if len(root_transitions) != wave_size:
                raise RuntimeError(
                    "Emulator batch_step returned a different number of transitions "
                    "than requested"
                )

            rollouts = [
                _Rollout(
                    root_action=action,
                    state=transition.child,
                    terminal=transition.terminal,
                    used_nodes=1,
                )
                for action, transition in zip(
                    assigned_actions,
                    root_transitions,
                    strict=True,
                )
            ]
            expanded_nodes += wave_size

            while expanded_nodes < max_nodes:
                if deadline is not None and time.monotonic() >= deadline:
                    break

                eligible = [
                    index
                    for index, rollout in enumerate(rollouts)
                    if not rollout.terminal
                    and rollout.used_nodes < self._rollout_depth
                ]
                if not eligible:
                    break

                allowance = min(len(eligible), max_nodes - expanded_nodes)
                eligible = eligible[:allowance]
                parent_handles = tuple(rollouts[index].state for index in eligible)
                batches = tuple(self._backend.batch_expand(parent_handles))
                if len(batches) != len(parent_handles):
                    raise RuntimeError(
                        "Emulator batch_expand returned a different number of batches "
                        "than requested"
                    )

                release_handles: list[StateHandle] = []
                for rollout_index, parent, expansions in zip(
                    eligible,
                    parent_handles,
                    batches,
                    strict=True,
                ):
                    ordered = tuple(
                        sorted(
                            expansions,
                            key=lambda item: item.action.action_id,
                        )
                    )
                    if not ordered:
                        rollouts[rollout_index].terminal = True
                        continue

                    chosen = rng.choice(ordered)
                    release_handles.append(parent)
                    release_handles.extend(
                        transition.child
                        for transition in ordered
                        if transition.child != chosen.child
                    )

                    rollout = rollouts[rollout_index]
                    rollout.state = chosen.child
                    rollout.terminal = chosen.terminal
                    rollout.used_nodes += 1
                    expanded_nodes += 1

                if release_handles:
                    self._backend.release_many(tuple(release_handles))

            observations = tuple(
                self._backend.batch_observe(
                    tuple(rollout.state for rollout in rollouts),
                    self._policy,
                )
            )
            if len(observations) != len(rollouts):
                raise RuntimeError(
                    "Emulator batch_observe returned a different number of observations "
                    "than requested"
                )

            for rollout, observation in zip(rollouts, observations, strict=True):
                accumulators[rollout.root_action.action_id].add(
                    self._leaf_value(observation)
                )

            self._backend.release_many(tuple(rollout.state for rollout in rollouts))

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

    def _state_seed(self, root_hash: str) -> int:
        digest = hashlib.sha256(
            f"{self._seed}:{root_hash}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def _prototype_leaf_value(observation: Observation) -> float:
        payload = json.loads(observation.payload_json)
        if not isinstance(payload, dict):
            raise RuntimeError("Prototype observation payload must be an object")

        terminal = payload.get("terminal_outcome")
        act = float(payload.get("act") or 0)
        floor = float(payload.get("floor") or 0)
        hp = float(payload.get("hp") or 0)
        max_hp = max(1.0, float(payload.get("max_hp") or 1))
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
