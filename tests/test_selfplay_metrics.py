"""Paired evaluation never reports censored games as defeats."""
from __future__ import annotations

import pytest

from sts2_ai.evaluation import RunSummary
from sts2_ai.evaluation.selfplay_metrics import (
    compare_completed_pairs,
    summarize_completed_runs,
    wilson_win_interval,
)


def _run(seed: str, *, outcome: str, progress: float) -> RunSummary:
    return RunSummary(
        seed=seed, outcome=outcome,
        terminal_act=1, terminal_floor=1, decisions=7,
        emulator_transitions=7, wall_seconds=1.2,
        agent_compute_seconds=0.4,
        frontier_progress=progress, frontier_enemy_hp=None,
        hp_trajectory=(25, 20),
        censored=outcome == "truncated",
        full_game_victory=(
            outcome == "victory" if outcome != "truncated" else None
        ),
    )


def test_wilson_interval_never_equates_zero_trials_to_zero_win_rate() -> None:
    assert wilson_win_interval(0, 0) is None
    lower, upper = wilson_win_interval(0, 6) or (None, None)
    assert lower == 0.0
    assert upper is not None and 0.25 < upper < 0.45
    lower, upper = wilson_win_interval(6, 6) or (None, None)
    assert upper == 1.0
    assert lower is not None and 0.55 < lower < 0.75
    with pytest.raises(ValueError):
        wilson_win_interval(7, 6)


def test_holdout_summary_marks_censorship_and_reports_completed_only() -> None:
    runs = [
        _run("a", outcome="defeat", progress=1.0),
        _run("b", outcome="truncated", progress=14.0),
        _run("c", outcome="victory", progress=17.0),
    ]
    report = summarize_completed_runs(runs)
    assert report["attempted"] == 3
    assert report["completed"] == 2
    assert report["censored"] == 1
    assert report["wins"] == 1
    assert report["win_rate_completed_only"] == 0.5
    assert report["mean_frontier_progress_completed_only"] == 9.0


def test_paired_comparison_strictly_ignores_either_censored_run() -> None:
    left = [
        _run("a", outcome="defeat", progress=2),
        _run("b", outcome="truncated", progress=15),
        _run("c", outcome="defeat", progress=5),
    ]
    right = [
        _run("c", outcome="victory", progress=16),
        _run("b", outcome="victory", progress=18),
        _run("a", outcome="defeat", progress=1),
    ]
    pairs = compare_completed_pairs(left, right)
    assert pairs["paired_completed"] == 2
    assert pairs["paired_excluded_censored"] == 1
    assert pairs["mean_frontier_delta"] == 5.0
    assert pairs["contender_ahead"] == 1
    assert pairs["baseline_ahead"] == 1
    with pytest.raises(ValueError, match="matching seeds"):
        compare_completed_pairs(left, right[:2])
