from sts2_ai.search import MonteCarloRolloutSearch, SearchBudget
from sts2_ai.testing import MockLinearBackend


def test_rollout_search_is_deterministic_and_respects_node_budget() -> None:
    backend = MockLinearBackend(terminal_at=5)
    search = MonteCarloRolloutSearch(
        backend,
        seed=17,
        max_rollout_depth=8,
    )

    first = search.search("0", SearchBudget(max_nodes=20))
    second = search.search("0", SearchBudget(max_nodes=20))

    assert first == second
    assert 0 < first.expanded_nodes <= 20
    assert first.search_version == "mc-rollout-v0"
    assert len(first.evaluations) == 2
    assert all(evaluation.visits > 0 for evaluation in first.evaluations)
