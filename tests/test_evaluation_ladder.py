from sts2_ai.evaluation import (
    BenchmarkRow,
    compare_paired_runs,
    markdown_table,
    paired_markdown_table,
    summarize_runs,
)
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
        frontier_progress=float(floor),
        frontier_enemy_hp=None,
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
        average_frontier_progress=5.0,
        average_defeat_enemy_hp=None,
        average_decisions=15.0,
        transitions_per_decision=400 / 30,
        average_wall_seconds=3.0,
        average_agent_compute_seconds=1.5,
    )
    table = markdown_table([row])
    assert "MCTS-32" in table
    assert "50.0" in table


def test_compare_paired_runs_uses_common_seed_deltas() -> None:
    baseline = [
        RunSummary(
            seed="a",
            outcome="defeat",
            terminal_act=1,
            terminal_floor=3,
            decisions=10,
            emulator_transitions=10,
            wall_seconds=1.0,
            agent_compute_seconds=0.1,
            frontier_progress=2.5,
            frontier_enemy_hp=20,
            hp_trajectory=(70, 30),
        ),
        RunSummary(
            seed="b",
            outcome="defeat",
            terminal_act=1,
            terminal_floor=4,
            decisions=10,
            emulator_transitions=10,
            wall_seconds=1.0,
            agent_compute_seconds=0.1,
            frontier_progress=3.0,
            frontier_enemy_hp=10,
            hp_trajectory=(70, 20),
        ),
    ]
    challenger = [
        RunSummary(
            seed="a",
            outcome="defeat",
            terminal_act=1,
            terminal_floor=4,
            decisions=11,
            emulator_transitions=20,
            wall_seconds=2.0,
            agent_compute_seconds=1.0,
            frontier_progress=3.5,
            frontier_enemy_hp=15,
            hp_trajectory=(70, 35),
        ),
        RunSummary(
            seed="b",
            outcome="defeat",
            terminal_act=1,
            terminal_floor=4,
            decisions=11,
            emulator_transitions=20,
            wall_seconds=2.0,
            agent_compute_seconds=1.0,
            frontier_progress=2.5,
            frontier_enemy_hp=12,
            hp_trajectory=(70, 18),
        ),
    ]

    comparison = compare_paired_runs(
        "heuristic",
        baseline,
        "MCTS-32",
        challenger,
    )

    assert comparison.runs == 2
    assert comparison.mean_frontier_delta == 0.25
    assert comparison.mean_terminal_delta == 0.5
    assert comparison.challenger_ahead == 1
    assert comparison.tied == 0
    assert comparison.baseline_ahead == 1
    table = paired_markdown_table([comparison])
    assert "MCTS-32" in table
    assert "+0.250" in table
