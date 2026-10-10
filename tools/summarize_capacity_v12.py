#!/usr/bin/env python3
"""Compare paired width32/64 capacity runs without cherry-picking training seeds."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sts2_ai.evaluation.checkpoint_selection import REPORT_SCHEMA, paired_comparison


def summarize_comparison(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA or report.get("mode") != "comparison":
        raise ValueError("Expected a non-selective certified Act-1 comparison report")
    identity = report["identity"]
    if identity["goal"] != "native-act1-boss-v1":
        raise ValueError("Capacity comparison must use the certified Act-1 goal")
    if report["analysis"].get("selected_label") is not None:
        raise ValueError("Individual checkpoint was selected: invalid capacity comparison")
    runs = report["runs"]
    if len(runs) != identity["seeds"] or not runs:
        raise ValueError("Incomplete evaluation cohort")
    expected = {
        f"h{hidden}-r{rep}" for rep in (1, 2, 3) for hidden in (32, 64)
    }
    if set(identity["model_specs"]) != expected:
        raise ValueError("Expected exactly three paired replicas of widths 32 and 64")
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in sorted(expected)}
    for index, record in enumerate(runs):
        if record["index"] != index:
            raise ValueError("Duplicate or missing comparison index")
        if record["seed"] != f"{identity['seed_prefix']}-{index}":
            raise ValueError("Comparison seed mismatch")
        if set(record["models"]) != expected:
            raise ValueError("Missing models in paired evaluation")
        for label in grouped:
            grouped[label].append(record["models"][label])
    pairs = []
    for replica in range(1, 4):
        left, right = f"h32-r{replica}", f"h64-r{replica}"
        pair = paired_comparison(grouped[right], grouped[left])
        n = len(grouped[left])
        a = sum(row["act1_cleared"] is True for row in grouped[left])
        b = sum(row["act1_cleared"] is True for row in grouped[right])
        pairs.append({
            "training_replica": replica,
            "h32_clears": a, "h64_clears": b,
            "h32_rate": a / n, "h64_rate": b / n,
            "width64_minus_width32": (b - a) / n,
            "paired": pair,
        })
    deltas = [pair["width64_minus_width32"] for pair in pairs]
    return {
        "schema": "sts2-v8-paired-capacity-comparison-v12",
        "cohort_seed_prefix": identity["seed_prefix"],
        "cohort_size": len(runs),
        "training_replicates": 3,
        "architecture": "v8 enemy instances, batched PPO, widths 32 versus 64",
        "per_training_replica": pairs,
        "mean_h32_rate": statistics.fmean(pair["h32_rate"] for pair in pairs),
        "mean_h64_rate": statistics.fmean(pair["h64_rate"] for pair in pairs),
        "mean_paired_width_advantage": statistics.fmean(deltas),
        "replicate_difference_range": [min(deltas), max(deltas)],
        "replicas_favoring_h64": sum(d > 0 for d in deltas),
        "caution": (
            "Training replicas, not evaluation seeds, are the independent "
            "units for assessing architecture robustness. Three paired "
            "training seeds cannot establish a precise population confidence interval."
        ),
    }


def training_times(root: Path) -> dict[str, dict[str, float | int]]:
    rows: dict[str, dict[str, float | int]] = {}
    for replica in range(1, 4):
        for width in (32, 64):
            label = f"h{width}-r{replica}"
            path = root / f"r{replica}" / f"h{width}" / "reports" / "stage-0080.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            rounds = report["rounds"]
            if len(rounds) != 80 or report.get("ppo_backend") != "batched":
                raise ValueError(f"Invalid trained model/report: {path}")
            rows[label] = {
                "rounds": len(rounds),
                "episodes_played": sum(row["played"] for row in rounds),
                "optimizer_seconds": sum(float(row["optimizer_seconds"]) for row in rounds),
                "rollout_seconds": sum(float(row["rollout_seconds"]) for row in rounds),
            }
    return rows


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Width 32 vs 64: paired capacity comparison",
        "",
        f"Fresh held-out seeds: **{summary['cohort_size']}**, "
        f"prefix: \`{summary['cohort_seed_prefix']}\`.",
        "",
        "| Training seed pair | Width 32 | Width 64 | Width-64 advantage |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["per_training_replica"]:
        lines.append(
            f"| r{row['training_replica']} "
            f"| {row['h32_clears']}/{summary['cohort_size']} "
            f"({row['h32_rate']:.1%}) "
            f"| {row['h64_clears']}/{summary['cohort_size']} "
            f"({row['h64_rate']:.1%}) "
            f"| {row['width64_minus_width32']:+.1%} |"
        )
    lines.extend([
        "",
        f"**Mean across the three paired training seeds:** "
        f"{summary['mean_h32_rate']:.1%} (h32) vs "
        f"{summary['mean_h64_rate']:.1%} (h64); "
        f"mean paired difference {summary['mean_paired_width_advantage']:+.1%}.",
        "",
        f"Replicas favoring h64: {summary['replicas_favoring_h64']}/3.",
        "",
        summary["caution"],
        "",
        "Compare optimizer/rollout seconds in the accompanying JSON. "
        "Equal PPO environment interactions do not imply equal wall-clock cost.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--training-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    summary = summarize_comparison(report)
    if args.training_root is not None:
        summary["training_cost"] = training_times(args.training_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(format_markdown(summary), encoding="utf-8")
    print(format_markdown(summary), flush=True)
    print(f"[capacity] JSON: {args.output}; Markdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
