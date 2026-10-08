"""Strict same-seed comparison for saved whole-run evaluation JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _runs(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("runs"), list):
        raise ValueError(f"Malformed evaluation report: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for item in raw["runs"]:
        if not isinstance(item, dict) or not isinstance(item.get("seed"), str):
            raise ValueError(f"Malformed run summary: {path}")
        seed = item["seed"]
        if seed in rows:
            raise ValueError(f"Duplicate run seed {seed!r} in {path}")
        for key in ("frontier_progress", "wall_seconds", "emulator_transitions",
                    "decisions", "outcome"):
            if key not in item:
                raise ValueError(f"Missing {key!r} in {path}")
        rows[seed] = item
    if not rows:
        raise ValueError(f"Empty evaluation report: {path}")
    return raw, rows


def paired_evaluation_report(baseline: Path, contender: Path) -> dict[str, Any]:
    """Pair by run seed; never compare unpaired averages."""
    left, left_rows = _runs(baseline)
    right, right_rows = _runs(contender)
    if left_rows.keys() != right_rows.keys():
        raise ValueError("Paired comparison requires identical run-seed sets")
    if left.get("rollout_depth") != right.get("rollout_depth"):
        raise ValueError("Cannot compare experiments with different rollout depths")
    seeds = sorted(left_rows)
    deltas = [
        float(right_rows[seed]["frontier_progress"])
        - float(left_rows[seed]["frontier_progress"])
        for seed in seeds
    ]
    wins_left = sum(left_rows[s]["outcome"] == "victory" for s in seeds)
    wins_right = sum(right_rows[s]["outcome"] == "victory" for s in seeds)

    def describe(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "wins": sum(rows[s]["outcome"] == "victory" for s in seeds),
            "defeats": sum(rows[s]["outcome"] == "defeat" for s in seeds),
            "truncated": sum(rows[s]["outcome"] == "truncated" for s in seeds),
            "mean_frontier_progress": fmean(
                float(rows[s]["frontier_progress"]) for s in seeds
            ),
            "mean_wall_seconds": fmean(float(rows[s]["wall_seconds"]) for s in seeds),
            "mean_agent_seconds": fmean(float(rows[s]["agent_compute_seconds"])
                                        for s in seeds),
            "mean_emulator_transitions": fmean(
                float(rows[s]["emulator_transitions"]) for s in seeds
            ),
            "mean_decisions": fmean(float(rows[s]["decisions"]) for s in seeds),
        }

    return {
        "seeds": len(seeds),
        "seed_ids": seeds,
        "baseline_agent": left.get("agent"),
        "contender_agent": right.get("agent"),
        "baseline": describe(left_rows),
        "contender": describe(right_rows),
        "paired_mean_frontier_delta": fmean(deltas),
        "paired_ahead": sum(d > 1e-9 for d in deltas),
        "paired_tied": sum(abs(d) <= 1e-9 for d in deltas),
        "paired_behind": sum(d < -1e-9 for d in deltas),
        "paired_win_difference": wins_right - wins_left,
        "warning": (
            "Small or truncated samples do not establish playing strength. "
            "Report search cost and full-run victory separately from frontier progress."
        ),
    }
