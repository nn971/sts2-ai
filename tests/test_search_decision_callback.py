from sts2_ai.emulator import InformationPolicy
from sts2_ai.evaluation import PrototypeSearchRunEvaluator
from sts2_ai.testing import MockLinearBackend


def test_search_decision_callback_records_replay_history() -> None:
    recorded = []
    backend = MockLinearBackend(terminal_at=5)

    PrototypeSearchRunEvaluator(
        backend,
        InformationPolicy("fair-test"),
        nodes_per_decision=4,
        rollout_depth=1,
        rollout_batch_size=2,
        search_seed=5,
        on_search_decision=recorded.append,
    ).evaluate(("callback-run",))

    assert recorded
    first = recorded[0]
    assert first.seed == "callback-run"
    assert first.decision_index == 0
    assert first.action_history == ()
    assert first.state_hash == backend.exact_hash("0")
    assert len(first.evaluations) == 2
