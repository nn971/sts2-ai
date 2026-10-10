#!/usr/bin/env python3
"""Independent training-replica analysis: attention versus equal-width v8 and width64."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from sts2_ai.evaluation.checkpoint_selection import (
    GOAL,
    REPORT_SCHEMA,
    is_certified_clear,
    paired_comparison,
)


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA or report.get("mode") != "comparison":
        raise ValueError("Expected a preregistered non-selective comparison")
    identity = report["identity"]
    if identity.get("goal") != GOAL:
        raise ValueError("Wrong episode goal")
    expected = {f"{prefix}-r{r}" for r in (1, 2, 3) for prefix in ("h32", "h64", "attn")}
    if set(identity["model_specs"]) != expected:
        raise ValueError("Expected all nine matched capacity/attention checkpoints")
    entries = report["runs"]
    if len(entries) != identity["seeds"] or not entries:
        raise ValueError("Incomplete comparison report")
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in expected}
    for i, entry in enumerate(entries):
        if entry["index"] != i or entry["seed"] != f"{identity['seed_prefix']}-{i}":
            raise ValueError("Comparison seed order mismatch")
        if set(entry["models"]) != expected:
            raise ValueError("Incomplete model evaluation")
        for name in expected:
            row = entry["models"][name]
            if row["seed"] != entry["seed"]:
                raise ValueError("Inconsistent per-model evaluation seeds")
            is_certified_clear(row)
            grouped[name].append(row)

    by_replica = []
    for replica in range(1, 4):
        rows: dict[str, Any] = {"replica": replica}
        for arch in ("h32", "h64", "attn"):
            name = f"{arch}-r{replica}"
            selected = grouped[name]
            wins = sum(is_certified_clear(row) for row in selected)
            bosses = [row for row in selected if row["boss_progress"] is not None]
            rows[arch] = {
                "wins": wins,
                "clear_rate": wins / len(selected),
                "boss_encounters": len(bosses),
                "boss_conversion": wins / len(bosses) if bosses else None,
            }
        rows["attention_vs_h32"] = paired_comparison(
            grouped[f"attn-r{replica}"], grouped[f"h32-r{replica}"],
        )
        rows["attention_vs_h64"] = paired_comparison(
            grouped[f"attn-r{replica}"], grouped[f"h64-r{replica}"],
        )
        by_replica.append(rows)

    means = {
        arch: statistics.fmean(row[arch]["clear_rate"] for row in by_replica)
        for arch in ("h32", "h64", "attn")
    }
    advantages = {
        baseline: [
            row["attn"]["clear_rate"] - row[baseline]["clear_rate"]
            for row in by_replica
        ]
        for baseline in ("h32", "h64")
    }
    return {
        "schema": "sts2-v13-attention-vs-width-capacity-v1",
        "seed_prefix": identity["seed_prefix"],
        "seeds": len(entries),
        "emulator_revision": identity["emulator_revision"],
        "replicas": by_replica,
        "mean_clear_rates": means,
        "mean_attention_advantages": {
            key: statistics.fmean(values) for key, values in advantages.items()
        },
        "positive_pair_counts": {
            key: sum(value > 0 for value in values)
            for key, values in advantages.items()
        },
        "note": (
            "Paired inference seeds control evaluation noise; the three TRAINING "
            "replicates are independent evidence about architecture effects. "
            "Do not treat nine policies times hundreds of seeds as independent "
            "training replications, and do not cherry-pick the best checkpoint."
        ),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# v13 enemy attention versus v8 width controls",
        "",
        f"Fresh certified Act 1 seeds: {summary['seeds']} "
        f"({summary['seed_prefix']}).",
        "",
        "| Training replicate | h32 | h64 | Attention h32 | Attention − h32 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["replicas"]:
        lines.append(
            f"| r{item['replica']} | {item['h32']['clear_rate']:.1%} "
            f"| {item['h64']['clear_rate']:.1%} "
            f"| {item['attn']['clear_rate']:.1%} "
            f"| {item['attention_vs_h32']['clear_rate_delta_first_minus_second']:+.1%} |"
        )
    means = summary["mean_clear_rates"]
    lines += [
        "",
        f"Mean: h32 {means['h32']:.1%}, h64 {means['h64']:.1%}, "
        f"attention {means['attn']:.1%}.",
        "",
        f"Attention better than h32 in "
        f"{summary['positive_pair_counts']['h32']}/3 training pairs.",
        "",
        summary["note"], "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    result = summarize(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
