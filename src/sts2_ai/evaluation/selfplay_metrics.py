"""Conservative paired held-out diagnostics for tiny self-play experiments.

Censored episodes are never defeats. Win-rate confidence intervals use
completed outcomes only; paired frontiers are diagnostic, not proof of skill.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from .run import RunSummary

Z_95 = 1.959963984540054


def wilson_win_interval(wins: int, completed: int) -> tuple[float, float] | None:
    """Two-sided 95% Wilson interval for completed Bernoulli outcomes only."""
    if type(completed) is not int or completed < 0 or not 0 <= wins <= completed:
        raise ValueError("Invalid completed run count / victories")
    if not completed:
        return None
    p = wins / completed
    zz = Z_95 * Z_95
    center = (p + zz / (2 * completed)) / (1 + zz / completed)
    margin = Z_95 * math.sqrt(
        p * (1 - p) / completed + zz / (4 * completed * completed)
    ) / (1 + zz / completed)
    # Exact Bernoulli endpoints should remain exact despite IEEE rounding.
    lower = 0.0 if wins == 0 else max(0.0, center - margin)
    upper = 1.0 if wins == completed else min(1.0, center + margin)
    return lower, upper


def summarize_completed_runs(runs: Sequence[RunSummary]) -> dict[str, Any]:
    valid = [run for run in runs if not run.censored]
    won = sum(run.won for run in valid)
    win_rate = won / len(valid) if valid else None
    return {
        "attempted": len(runs),
        "completed": len(valid),
        "censored": len(runs) - len(valid),
        "wins": won,
        "win_rate_completed_only": win_rate,
        "win_rate_wilson95": wilson_win_interval(won, len(valid)),
        "mean_frontier_progress_completed_only": (
            statistics.fmean(run.frontier_progress for run in valid)
            if valid else None
        ),
        "mean_agent_compute_seconds": (
            statistics.fmean(run.agent_compute_seconds for run in runs)
            if runs else None
        ),
    }


def compare_completed_pairs(
    baseline: Sequence[RunSummary], contender: Sequence[RunSummary]
) -> dict[str, Any]:
    """Include a seed only if BOTH runs reached actual terminal outcomes."""
    if len({row.seed for row in baseline}) != len(baseline):
        raise ValueError("Duplicate baseline seeds")
    if len({row.seed for row in contender}) != len(contender):
        raise ValueError("Duplicate contender seeds")
    if {row.seed for row in baseline} != {row.seed for row in contender}:
        raise ValueError("Paired evaluation requires matching seeds")
    left = {row.seed: row for row in baseline}
    right = {row.seed: row for row in contender}
    deltas = [
        right[seed].frontier_progress - left[seed].frontier_progress
        for seed in sorted(left)
        if not left[seed].censored and not right[seed].censored
    ]
    return {
        "paired_completed": len(deltas),
        "paired_excluded_censored": len(left) - len(deltas),
        "mean_frontier_delta": statistics.fmean(deltas) if deltas else None,
        "contender_ahead": sum(x > 1e-12 for x in deltas),
        "tied": sum(abs(x) <= 1e-12 for x in deltas),
        "baseline_ahead": sum(x < -1e-12 for x in deltas),
    }
