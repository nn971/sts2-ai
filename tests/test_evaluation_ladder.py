from sts2_ai.evaluation import BenchmarkRow, markdown_table, summarize_runs
from sts2_ai.evaluation.run import RunSummary


def _run(
    *,
    outcome: str,
    floor: int,
    decisions: int,
    transitions: int,
    wall: float,
) -> RunSummary:
    return RunSummary(
        seed="seed",
        outcome=outcome,
        terminal_act=1,
        terminal_floor=floor,
        decisions=decisions,
        emulator_transitions=transitions,
        wall_seconds=wall,
        agent_compute_seconds=wall / 2,
        hp_trajectory=(70, 60),
    )


def test_summarize_runs_aggregates_common_metrics() -> None:
    row = summarize_runs(
        "MCTS-32",
        [
            _run(outcome="victory", floor=6, decisions=10, transitions=100, wall=2.0),
            _run(outcome="defeat", floor=4, decisions=20, transitions=300, wall=4.0),
        ],
    )

    assert row == BenchmarkRow(
        label="MCTS-32",
        runs=2,
        win_rate=0.5,
        average_terminal_progress=5.0,
        average_decisions=15.0,
        transitions_per_decision=400 / 30,
        average_wall_seconds=3.0,
        average_agent_compute_seconds=1.5,
    )
    table = markdown_table([row])
    assert "MCTS-32" in table
    assert "50.0" in table
