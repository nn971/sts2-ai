from sts2_ai.agents import RandomAgent
from sts2_ai.emulator import InformationPolicy
from sts2_ai.evaluation import play_run
from sts2_ai.testing import MockLinearBackend


def test_full_run_driver_reaches_terminal() -> None:
    backend = MockLinearBackend(terminal_at=6)
    summary = play_run(
        backend,
        RandomAgent(seed=5),
        seed="mock-run",
        policy=InformationPolicy("fair-test"),
    )

    assert summary.decisions > 0
    assert summary.emulator_transitions == summary.decisions
    assert summary.outcome == "unknown"
