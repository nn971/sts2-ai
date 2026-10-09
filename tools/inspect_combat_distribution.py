#!/usr/bin/env python3
"""Inspect nonparametric on-policy combat-outcome distributions from JSONL.

This is a descriptive baseline, NOT a trained tactical policy or a calibrated
counterfactual predictor. No risk-preference scalar is imposed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.training.combat_distribution import (
    EmpiricalCombatDistribution, load_combat_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path)
    parser.add_argument("--min-group-size", type=int, default=8)
    parser.add_argument("--example-count", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outcomes = load_combat_samples(args.samples)
    distribution = EmpiricalCombatDistribution(
        outcomes, min_group_size=args.min_group_size
    )
    report = {
        "samples": len(outcomes),
        "groups": len(distribution.groups),
        "examples": [
            {
                "entry": outcome["entry"],
                "enemies": outcome["enemy_ids"],
                "distribution": distribution.estimate(outcome),
            }
            for outcome in outcomes[:args.example_count]
        ],
        "limitations": (
            "These are empirical on-policy frequencies from observed runs, "
            "not calibrated predictions for new strategies or unseen fights."
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload, flush=True)


if __name__ == "__main__":
    main()
