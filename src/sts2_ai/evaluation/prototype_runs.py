from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from sts2_ai.emulator import (
    EmulatorBackend,
    InformationPolicy,
    LegalAction,
)
from sts2_ai.search import ActionEvaluation, PrototypeFlatRolloutSearch, SearchBudget
from sts2_ai.strategy_db import (
    CachedActionEvaluation,
    SQLiteStrategyStore,
)


@dataclass(frozen=True, slots=True)
class PrototypeSearchDecision:
    seed: str
    decision_index: int
    state_hash: str
    action_history: tuple[str, ...]
    evaluations: tuple[ActionEvaluation, ...]


@dataclass(frozen=True, slots=True)
class PrototypeRunEvaluation:
    seed: str
    outcome: str | None
    decisions: int
    search_decisions: int
    expanded_nodes: int
    cache_hits: int
    unique_search_states: int
    repeated_search_states: int
    final_state_hash: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PrototypeEvaluationSummary:
    search_version: str
    runs: tuple[PrototypeRunEvaluation, ...]
    victories: int
    defeats: int
    unknown_terminal_outcomes: int
    total_decisions: int
    total_search_decisions: int
    total_expanded_nodes: int
    total_cache_hits: int
    unique_search_states: int
    repeated_search_states: int
    elapsed_seconds: float

    @property
    def victory_rate(self) -> float:
        return self.victories / len(self.runs) if self.runs else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return (
            self.total_cache_hits / self.total_search_decisions
            if self.total_search_decisions
            else 0.0
        )

    @property
    def exact_state_recurrence_rate(self) -> float:
        total = self.unique_search_states + self.repeated_search_states
        return self.repeated_search_states / total if total else 0.0

    @property
    def decisions_per_second(self) -> float:
        return (
            self.total_decisions / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )

    @property
    def nodes_per_second(self) -> float:
        return (
            self.total_expanded_nodes / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["victory_rate"] = self.victory_rate
        payload["cache_hit_rate"] = self.cache_hit_rate
        payload["exact_state_recurrence_rate"] = self.exact_state_recurrence_rate
        payload["decisions_per_second"] = self.decisions_per_second
        payload["nodes_per_second"] = self.nodes_per_second
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class PrototypeSearchRunEvaluator:
    """Deterministic full-run evaluation using the prototype rollout search at choices.

    Forced actions are taken without search. Every state handle that is no longer needed is
    released promptly so multi-seed evaluations exercise the backend under realistic long-lived
    process conditions instead of accumulating an artificial state-store leak.
    """

    def __init__(
        self,
        backend: EmulatorBackend,
        policy: InformationPolicy,
        *,
        nodes_per_decision: int = 128,
        rollout_depth: int = 32,
        rollout_batch_size: int = 16,
        search_seed: int = 0,
        max_decisions: int = 5_000,
        strategy_store: SQLiteStrategyStore | None = None,
        game_build: str = "prototype-unbound",
        model_id: str | None = None,
        on_search_decision: Callable[[PrototypeSearchDecision], None] | None = None,
        search_type: type[PrototypeFlatRolloutSearch] = PrototypeFlatRolloutSearch,
    ) -> None:
        if nodes_per_decision <= 0:
            raise ValueError("nodes_per_decision must be positive")
        if rollout_depth <= 0:
            raise ValueError("rollout_depth must be positive")
        if rollout_batch_size <= 0:
            raise ValueError("rollout_batch_size must be positive")
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")

        self._backend = backend
        self._policy = policy
        self._nodes_per_decision = nodes_per_decision
        self._rollout_depth = rollout_depth
        self._rollout_batch_size = rollout_batch_size
        self._search_seed = search_seed
        self._max_decisions = max_decisions
        self._strategy_store = strategy_store
        self._game_build = game_build
        self._model_id = model_id
        self._on_search_decision = on_search_decision
        self._search_type = search_type

    def evaluate(self, seeds: tuple[str, ...]) -> PrototypeEvaluationSummary:
        started = time.perf_counter()
        seen_search_states: set[str] = set()
        runs = tuple(
            self._evaluate_one(seed, run_index, seen_search_states)
            for run_index, seed in enumerate(seeds)
        )
        elapsed = time.perf_counter() - started

        victories = sum(run.outcome == "victory" for run in runs)
        defeats = sum(run.outcome == "defeat" for run in runs)
        unknown = len(runs) - victories - defeats

        return PrototypeEvaluationSummary(
            search_version=self._search_type.search_version,
            runs=runs,
            victories=victories,
            defeats=defeats,
            unknown_terminal_outcomes=unknown,
            total_decisions=sum(run.decisions for run in runs),
            total_search_decisions=sum(run.search_decisions for run in runs),
            total_expanded_nodes=sum(run.expanded_nodes for run in runs),
            total_cache_hits=sum(run.cache_hits for run in runs),
            unique_search_states=len(seen_search_states),
            repeated_search_states=sum(run.repeated_search_states for run in runs),
            elapsed_seconds=elapsed,
        )

    def _evaluate_one(
        self,
        seed: str,
        run_index: int,
        seen_search_states: set[str],
    ) -> PrototypeRunEvaluation:
        run_search_seed = self._search_seed + run_index
        search = self._search_type(
            self._backend,
            self._policy,
            seed=run_search_seed,
            rollout_depth=self._rollout_depth,
            rollout_batch_size=self._rollout_batch_size,
        )

        state = self._backend.reset(seed)
        decisions = 0
        search_decisions = 0
        expanded_nodes = 0
        cache_hits = 0
        run_search_states: set[str] = set()
        repeated_search_states = 0
        search_config_id = self._search_config_id(run_search_seed)
        action_history: list[str] = []
        started = time.perf_counter()

        try:
            while not self._backend.is_terminal(state):
                if decisions >= self._max_decisions:
                    raise RuntimeError(
                        f"Prototype run {seed!r} exceeded "
                        f"{self._max_decisions} decisions"
                    )

                legal = tuple(self._backend.legal_actions(state))
                if not legal:
                    raise RuntimeError(
                        f"Nonterminal prototype state for seed {seed!r} has no legal actions"
                    )

                if len(legal) == 1:
                    action = legal[0]
                else:
                    search_decisions += 1
                    state_hash = self._backend.exact_hash(state)
                    if state_hash in seen_search_states:
                        repeated_search_states += 1
                    else:
                        seen_search_states.add(state_hash)
                    run_search_states.add(state_hash)

                    cached = self._cached_evaluations(
                        state_hash=state_hash,
                        legal=legal,
                        search_config_id=search_config_id,
                    )

                    if cached is not None:
                        cache_hits += 1
                        evaluations = cached
                    else:
                        result = search.search(
                            state,
                            SearchBudget(max_nodes=self._nodes_per_decision),
                        )
                        expanded_nodes += result.expanded_nodes
                        evaluations = result.evaluations
                        self._cache_evaluations(
                            evaluations,
                            state_hash=state_hash,
                            search_config_id=search_config_id,
                        )

                    if self._on_search_decision is not None:
                        self._on_search_decision(
                            PrototypeSearchDecision(
                                seed=seed,
                                decision_index=decisions,
                                state_hash=state_hash,
                                action_history=tuple(action_history),
                                evaluations=evaluations,
                            )
                        )
                    action = self._best_action(evaluations)

                transition = self._backend.step(state, action)
                previous = state
                state = transition.child
                self._backend.release_many((previous,))
                action_history.append(action.action_id)
                decisions += 1

            observation = self._backend.observe(state, self._policy)
            payload = json.loads(observation.payload_json)
            if not isinstance(payload, dict):
                raise RuntimeError("Prototype terminal observation must be an object")

            raw_outcome = payload.get("terminal_outcome")
            outcome = raw_outcome if isinstance(raw_outcome, str) else None
            final_hash = self._backend.exact_hash(state)
            elapsed = time.perf_counter() - started

            return PrototypeRunEvaluation(
                seed=seed,
                outcome=outcome,
                decisions=decisions,
                search_decisions=search_decisions,
                expanded_nodes=expanded_nodes,
                cache_hits=cache_hits,
                unique_search_states=len(run_search_states),
                repeated_search_states=repeated_search_states,
                final_state_hash=final_hash,
                elapsed_seconds=elapsed,
            )
        finally:
            self._backend.release_many((state,))

    def _search_config_id(self, search_seed: int) -> str:
        return (
            f"nodes={self._nodes_per_decision};"
            f"depth={self._rollout_depth};"
            f"batch={self._rollout_batch_size};"
            f"seed={search_seed}"
        )

    def _cached_evaluations(
        self,
        *,
        state_hash: str,
        legal: tuple[LegalAction, ...],
        search_config_id: str,
    ) -> tuple[ActionEvaluation, ...] | None:
        if self._strategy_store is None:
            return None

        entries = self._strategy_store.cached_action_evaluations(
            state_hash=state_hash,
            information_policy=self._policy.policy_id,
            search_version=self._search_type.search_version,
            search_config_id=search_config_id,
            emulator_revision=self._backend.emulator_revision,
            game_build=self._game_build,
        )
        if not entries:
            return None

        legal_by_id = {action.action_id: action for action in legal}
        if set(legal_by_id) != {entry.action_id for entry in entries}:
            return None

        return tuple(
            ActionEvaluation(
                action=legal_by_id[entry.action_id],
                value=entry.value,
                visits=entry.visits,
                uncertainty=entry.uncertainty,
            )
            for entry in entries
        )

    def _cache_evaluations(
        self,
        evaluations: tuple[ActionEvaluation, ...],
        *,
        state_hash: str,
        search_config_id: str,
    ) -> None:
        if self._strategy_store is None:
            return

        self._strategy_store.cache_action_evaluations(
            tuple(
                CachedActionEvaluation(
                    state_hash=state_hash,
                    information_policy=self._policy.policy_id,
                    action_id=evaluation.action.action_id,
                    value=evaluation.value,
                    visits=evaluation.visits,
                    uncertainty=evaluation.uncertainty,
                    search_version=self._search_type.search_version,
                    search_config_id=search_config_id,
                    model_id=self._model_id,
                    emulator_revision=self._backend.emulator_revision,
                    game_build=self._game_build,
                )
                for evaluation in evaluations
            )
        )

    @staticmethod
    def _best_action(evaluations: tuple[ActionEvaluation, ...]) -> LegalAction:
        if not evaluations:
            raise RuntimeError("Search returned no root evaluations")

        # ActionEvaluation is intentionally duck-typed here only to keep this helper local and
        # avoid widening the public evaluator API.
        best = max(
            evaluations,
            key=lambda item: (item.value, item.visits),
        )
        tied = [
            item
            for item in evaluations
            if item.value == best.value and item.visits == best.visits
        ]
        return min(tied, key=lambda item: item.action.action_id).action



@dataclass(frozen=True, slots=True)
class PrototypeRandomRunEvaluation:
    seed: str
    outcome: str | None
    decisions: int
    final_state_hash: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class PrototypeRandomEvaluationSummary:
    runs: tuple[PrototypeRandomRunEvaluation, ...]
    victories: int
    defeats: int
    unknown_terminal_outcomes: int
    total_decisions: int
    elapsed_seconds: float

    @property
    def victory_rate(self) -> float:
        return self.victories / len(self.runs) if self.runs else 0.0

    @property
    def decisions_per_second(self) -> float:
        return (
            self.total_decisions / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["victory_rate"] = self.victory_rate
        payload["decisions_per_second"] = self.decisions_per_second
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class PrototypeRandomRunEvaluator:
    """Same-seed deterministic random-policy control for search evaluation."""

    def __init__(
        self,
        backend: EmulatorBackend,
        policy: InformationPolicy,
        *,
        random_seed: int = 0,
        max_decisions: int = 5_000,
    ) -> None:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        self._backend = backend
        self._policy = policy
        self._random_seed = random_seed
        self._max_decisions = max_decisions

    def evaluate(self, seeds: tuple[str, ...]) -> PrototypeRandomEvaluationSummary:
        started = time.perf_counter()
        runs = tuple(
            self._evaluate_one(seed, run_index)
            for run_index, seed in enumerate(seeds)
        )
        elapsed = time.perf_counter() - started

        victories = sum(run.outcome == "victory" for run in runs)
        defeats = sum(run.outcome == "defeat" for run in runs)
        return PrototypeRandomEvaluationSummary(
            runs=runs,
            victories=victories,
            defeats=defeats,
            unknown_terminal_outcomes=len(runs) - victories - defeats,
            total_decisions=sum(run.decisions for run in runs),
            elapsed_seconds=elapsed,
        )

    def _evaluate_one(
        self,
        seed: str,
        run_index: int,
    ) -> PrototypeRandomRunEvaluation:
        rng = random.Random(self._random_seed + run_index)
        state = self._backend.reset(seed)
        decisions = 0
        started = time.perf_counter()

        try:
            while not self._backend.is_terminal(state):
                if decisions >= self._max_decisions:
                    raise RuntimeError(
                        f"Prototype random run {seed!r} exceeded "
                        f"{self._max_decisions} decisions"
                    )

                legal = tuple(
                    sorted(
                        self._backend.legal_actions(state),
                        key=lambda action: action.action_id,
                    )
                )
                if not legal:
                    raise RuntimeError(
                        f"Nonterminal prototype state for seed {seed!r} has no legal actions"
                    )

                action = legal[0] if len(legal) == 1 else rng.choice(legal)
                transition = self._backend.step(state, action)
                previous = state
                state = transition.child
                self._backend.release_many((previous,))
                decisions += 1

            observation = self._backend.observe(state, self._policy)
            payload = json.loads(observation.payload_json)
            if not isinstance(payload, dict):
                raise RuntimeError("Prototype terminal observation must be an object")

            raw_outcome = payload.get("terminal_outcome")
            outcome = raw_outcome if isinstance(raw_outcome, str) else None
            return PrototypeRandomRunEvaluation(
                seed=seed,
                outcome=outcome,
                decisions=decisions,
                final_state_hash=self._backend.exact_hash(state),
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            self._backend.release_many((state,))
