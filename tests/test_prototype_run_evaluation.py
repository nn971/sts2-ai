from __future__ import annotations

import json

from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.evaluation import PrototypeRandomRunEvaluator, PrototypeSearchRunEvaluator
from sts2_ai.strategy_db import SQLiteStrategyStore
from sts2_ai.testing import MockLinearBackend


class OutcomeMockBackend(MockLinearBackend):
    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        base = super().observe(state, policy)
        payload = json.loads(base.payload_json)
        if self.is_terminal(state):
            payload["terminal_outcome"] = "victory"
        return Observation(
            policy_id=base.policy_id,
            payload_json=json.dumps(payload, sort_keys=True),
            observation_hash=base.observation_hash,
        )


class TrackingMockBackend(MockLinearBackend):
    def __init__(self, terminal_at: int = 5) -> None:
        super().__init__(terminal_at=terminal_at)
        self.released: list[str] = []

    def release_many(self, states: tuple[str, ...]) -> int:
        self.released.extend(states)
        return len(states)


def test_multi_seed_evaluator_reaches_terminal_deterministically() -> None:
    first_backend = TrackingMockBackend(terminal_at=6)
    first = PrototypeSearchRunEvaluator(
        first_backend,
        InformationPolicy("fair-test"),
        nodes_per_decision=12,
        rollout_depth=3,
        rollout_batch_size=4,
        search_seed=17,
    ).evaluate(("run-a", "run-b"))

    second_backend = TrackingMockBackend(terminal_at=6)
    second = PrototypeSearchRunEvaluator(
        second_backend,
        InformationPolicy("fair-test"),
        nodes_per_decision=12,
        rollout_depth=3,
        rollout_batch_size=4,
        search_seed=17,
    ).evaluate(("run-a", "run-b"))

    assert len(first.runs) == 2
    assert first.total_decisions > 0
    assert first.total_search_decisions > 0
    assert first.total_expanded_nodes > 0
    assert first.victories == 0
    assert first.defeats == 0
    assert first.unknown_terminal_outcomes == 2

    assert [
        (
            run.seed,
            run.outcome,
            run.decisions,
            run.search_decisions,
            run.expanded_nodes,
            run.final_state_hash,
        )
        for run in first.runs
    ] == [
        (
            run.seed,
            run.outcome,
            run.decisions,
            run.search_decisions,
            run.expanded_nodes,
            run.final_state_hash,
        )
        for run in second.runs
    ]

    assert first.total_decisions == second.total_decisions
    assert first.total_search_decisions == second.total_search_decisions
    assert first.total_expanded_nodes == second.total_expanded_nodes
    assert first_backend.released
    assert second_backend.released


def test_empty_seed_set_returns_zero_summary() -> None:
    summary = PrototypeSearchRunEvaluator(
        MockLinearBackend(),
        InformationPolicy("fair-test"),
    ).evaluate(())

    assert summary.runs == ()
    assert summary.victory_rate == 0.0
    assert summary.nodes_per_second == 0.0
    assert summary.total_decisions == 0
    assert summary.total_expanded_nodes == 0


def test_evaluator_reads_snake_case_terminal_outcome() -> None:
    summary = PrototypeSearchRunEvaluator(
        OutcomeMockBackend(terminal_at=3),
        InformationPolicy("fair-test"),
        nodes_per_decision=4,
        rollout_depth=1,
        rollout_batch_size=2,
        search_seed=3,
    ).evaluate(("terminal-outcome",))

    assert summary.victories == 1
    assert summary.defeats == 0
    assert summary.unknown_terminal_outcomes == 0
    assert summary.victory_rate == 1.0
    assert summary.runs[0].outcome == "victory"


def test_exact_search_cache_reuses_state_evaluations(tmp_path) -> None:
    database = tmp_path / "strategy.sqlite"

    with SQLiteStrategyStore(database) as store:
        first_backend = MockLinearBackend(terminal_at=8)
        first = PrototypeSearchRunEvaluator(
            first_backend,
            InformationPolicy("fair-test"),
            nodes_per_decision=12,
            rollout_depth=3,
            rollout_batch_size=4,
            search_seed=29,
            strategy_store=store,
            game_build="test-build",
        ).evaluate(("cache-a", "cache-b"))

        second_backend = MockLinearBackend(terminal_at=8)
        second = PrototypeSearchRunEvaluator(
            second_backend,
            InformationPolicy("fair-test"),
            nodes_per_decision=12,
            rollout_depth=3,
            rollout_batch_size=4,
            search_seed=29,
            strategy_store=store,
            game_build="test-build",
        ).evaluate(("cache-a", "cache-b"))

    assert first.total_expanded_nodes > 0
    assert second.total_expanded_nodes == 0
    assert second.total_cache_hits == second.total_search_decisions
    assert second.total_cache_hits > 0

    assert [
        (run.decisions, run.final_state_hash)
        for run in first.runs
    ] == [
        (run.decisions, run.final_state_hash)
        for run in second.runs
    ]



def test_search_evaluator_reports_exact_state_recurrence_across_runs() -> None:
    summary = PrototypeSearchRunEvaluator(
        MockLinearBackend(terminal_at=6),
        InformationPolicy("fair-test"),
        nodes_per_decision=8,
        rollout_depth=2,
        rollout_batch_size=2,
        search_seed=11,
    ).evaluate(("same-shape-a", "same-shape-b"))

    assert summary.unique_search_states > 0
    assert summary.repeated_search_states > 0
    assert 0.0 < summary.exact_state_recurrence_rate < 1.0


def test_random_baseline_is_deterministic_under_seed() -> None:
    first = PrototypeRandomRunEvaluator(
        OutcomeMockBackend(terminal_at=7),
        InformationPolicy("fair-test"),
        random_seed=23,
    ).evaluate(("baseline-a", "baseline-b"))

    second = PrototypeRandomRunEvaluator(
        OutcomeMockBackend(terminal_at=7),
        InformationPolicy("fair-test"),
        random_seed=23,
    ).evaluate(("baseline-a", "baseline-b"))

    assert first.victories == 2
    assert first.defeats == 0
    assert [
        (run.seed, run.decisions, run.final_state_hash)
        for run in first.runs
    ] == [
        (run.seed, run.decisions, run.final_state_hash)
        for run in second.runs
    ]
