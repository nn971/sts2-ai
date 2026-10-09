#!/usr/bin/env python3
"""Train a PUBLIC combat survival + exit-HP distribution predictor.

This is an auxiliary learning experiment, not the agent's tactical reward.
Keeps defeat probability separate from conditional surviving HP bins, allowing
risk objectives to be chosen and validated later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sts2_ai.training.combat_distributional_fit import (
    constant_distributional_baseline,
    distributional_metrics,
    fit_distributional_predictor,
)
from sts2_ai.training.combat_predictor import load_outcome_samples
from sts2_ai.training.combat_predictor_fit import split_by_run_seed


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
    examples = load_outcome_samples(args.samples)
    training, heldout = split_by_run_seed(examples)
    print(
        f"[distribution] samples={len(examples)} "
        f"train={len(training)} heldout={len(heldout)} "
        f"train_runs={len({x.seed for x in training})} "
        f"heldout_runs={len({x.seed for x in heldout})}",
        file=sys.stderr, flush=True,
    )
    if len(training) < 32:
        print(
            "[distribution] warning: fewer than 32 training combats; "
            "this is only a functionality check, not a reliable distribution",
            file=sys.stderr, flush=True,
        )

    def progress(epoch: int, loss: float) -> None:
        print(
            f"[distribution] epoch={epoch}/{args.epochs} "
            f"training_loss={loss:.5f}",
            file=sys.stderr, flush=True,
        )

    model = fit_distributional_predictor(
        training,
        dimension=args.dimension, hidden=args.hidden,
        epochs=args.epochs, learning_rate=args.learning_rate,
        seed=args.seed, progress=progress,
    )
    model.save(args.output)
    report = {
        "schema": "sts2-combat-distributional-training-report-v1",
        "model_id": model.model_id,
        "train_samples": len(training),
        "heldout_samples": len(heldout),
        "train_runs": sorted({x.seed for x in training}),
        "heldout_runs": sorted({x.seed for x in heldout}),
        "train": distributional_metrics(model, training),
        "heldout": distributional_metrics(model, heldout),
        "constant_baseline_heldout": constant_distributional_baseline(
            training, heldout,
        ),
        "limitations": (
            "Public on-policy combat-entry predictions with a coarse HP "
            "histogram. Not counterfactual, not calibrated without sufficient "
            "heldout evidence, no potions distribution yet. Predictions are "
            "not currently used by the tactical or strategic policy."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[distribution] saved model={args.output} report={args.report}",
        file=sys.stderr, flush=True,
    )
    print(json.dumps({
        "heldout": report["heldout"],
        "constant_baseline_heldout": report["constant_baseline_heldout"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
