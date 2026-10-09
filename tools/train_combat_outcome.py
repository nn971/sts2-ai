#!/usr/bin/env python3
"""Fit a public-only combat outcome predictor from per-round JSONL samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sts2_ai.training.combat_predictor import load_outcome_samples
from sts2_ai.training.combat_predictor_fit import (
    constant_baseline_metrics, evaluation_metrics, fit_outcome_predictor,
    split_by_run_seed,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--dimension", type=int, default=256)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=41)
    args = p.parse_args()
    samples = load_outcome_samples(args.samples)
    train, heldout = split_by_run_seed(samples)
    print(
        f"[combat-fit] samples={len(samples)} train={len(train)} "
        f"heldout={len(heldout)} (disjoint seeds)",
        file=sys.stderr, flush=True,
    )

    def progress(epoch: int, loss: float) -> None:
        print(
            f"[combat-fit] epoch={epoch}/{args.epochs} loss={loss:.6f}",
            file=sys.stderr, flush=True,
        )

    model = fit_outcome_predictor(
        train, dimension=args.dimension, hidden=args.hidden,
        epochs=args.epochs, learning_rate=args.learning_rate,
        seed=args.seed, progress=progress,
    )
    model.save(args.output)
    metrics = evaluation_metrics(model, heldout)
    baseline = constant_baseline_metrics(train, heldout)
    report = {
        "schema": "sts2-combat-outcome-holdout-v1",
        "model_id": model.model_id,
        "train_samples": len(train),
        "heldout_samples": len(heldout),
        "train_runs": len({s.seed for s in train}),
        "heldout_runs": len({s.seed for s in heldout}),
        "heldout": metrics,
        "constant_baseline_heldout": baseline,
        "targets": ["combat_survival", "exit_hp_over_max_hp"],
        "warning": (
            "Small or correlated samples cannot establish generalization. "
            "The outcome predictor is not a potion-value critic or tactical "
            "actor objective."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[combat-fit] heldout={metrics} baseline={baseline} "
        f"model={args.output} report={args.report}",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
