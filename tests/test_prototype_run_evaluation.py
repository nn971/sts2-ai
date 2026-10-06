from __future__ import annotations

from sts2_ai.emulator import InformationPolicy
from sts2_ai.evaluation import PrototypeSearchRunEvaluator
from sts2_ai.testing import MockLinearBackend


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
