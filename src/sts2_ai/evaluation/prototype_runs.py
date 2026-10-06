from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sts2_ai.emulator import (
    EmulatorBackend,
    InformationPolicy,
    LegalAction,
    StateHandle,
)
from sts2_ai.search import ActionEvaluation, PrototypeFlatRolloutSearch, SearchBudget


@dataclass(frozen=True, slots=True)
class PrototypeRunEvaluation:
    seed: str
    outcome: str | None
    decisions: int
    search_decisions: int
    expanded_nodes: int
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
    elapsed_seconds: float

    @property
    def victory_rate(self) -> float:
        return self.victories / len(self.runs) if self.runs else 0.0

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

    def evaluate(self, seeds: tuple[str, ...]) -> PrototypeEvaluationSummary:
        started = time.perf_counter()
        runs = tuple(
            self._evaluate_one(seed, run_index)
            for run_index, seed in enumerate(seeds)
        )
        elapsed = time.perf_counter() - started

        victories = sum(run.outcome == "victory" for run in runs)
        defeats = sum(run.outcome == "defeat" for run in runs)
        unknown = len(runs) - victories - defeats

        return PrototypeEvaluationSummary(
            search_version=PrototypeFlatRolloutSearch.search_version,
            runs=runs,
            victories=victories,
            defeats=defeats,
            unknown_terminal_outcomes=unknown,
            total_decisions=sum(run.decisions for run in runs),
            total_search_decisions=sum(run.search_decisions for run in runs),
            total_expanded_nodes=sum(run.expanded_nodes for run in runs),
            elapsed_seconds=elapsed,
        )

    def _evaluate_one(self, seed: str, run_index: int) -> PrototypeRunEvaluation:
        search = PrototypeFlatRolloutSearch(
            self._backend,
            self._policy,
            seed=self._search_seed + run_index,
            rollout_depth=self._rollout_depth,
            rollout_batch_size=self._rollout_batch_size,
        )

        state = self._backend.reset(seed)
        decisions = 0
        search_decisions = 0
        expanded_nodes = 0
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
                    result = search.search(
                        state,
                        SearchBudget(max_nodes=self._nodes_per_decision),
                    )
                    search_decisions += 1
                    expanded_nodes += result.expanded_nodes
                    action = self._best_action(result.evaluations)

                transition = self._backend.step(state, action)
                previous = state
                state = transition.child
                self._backend.release_many((previous,))
                decisions += 1

            observation = self._backend.observe(state, self._policy)
            payload = json.loads(observation.payload_json)
            if not isinstance(payload, dict):
                raise RuntimeError("Prototype terminal observation must be an object")

            raw_outcome = payload.get("terminalOutcome")
            outcome = raw_outcome if isinstance(raw_outcome, str) else None
            final_hash = self._backend.exact_hash(state)
            elapsed = time.perf_counter() - started

            return PrototypeRunEvaluation(
                seed=seed,
                outcome=outcome,
                decisions=decisions,
                search_decisions=search_decisions,
                expanded_nodes=expanded_nodes,
                final_state_hash=final_hash,
                elapsed_seconds=elapsed,
            )
        finally:
            self._backend.release_many((state,))

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
