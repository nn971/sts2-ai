#!/usr/bin/env python3
"""Aggregate paired Nature PPO retraining seeds and ordinary held-out results.

Expected immutable output layout:
  study/seed-29/nature/{history.json,final-model.json}
  study/seed-29/ordinary/{history.json,final-model.json}
  study/seed-29/holdout/holdout-report.json

Every replica is paired by the *same* restart seed. The report distinguishes
two-axis sampling uncertainty from incomplete computational-budget parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.replicated_study import analyze_replicated_study


def load_replica(root: Path, seed: int) -> dict:
    directory = root / f"seed-{seed}"
    files = {
        "nature_history": directory / "nature" / "history.json",
        "ordinary_history": directory / "ordinary" / "history.json",
        "holdout": directory / "holdout" / "holdout-report.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing genuine completed study artifact: {path}")
    model_hashes = {
        f"{arm}_model_sha256": file_sha256(
            directory / arm / "final-model.json"
        )
        for arm in ("nature", "ordinary")
    }
    return {
        **{key: json.loads(path.read_text(encoding="utf-8"))
           for key, path in files.items()},
        **model_hashes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-root", required=True, type=Path)
    p.add_argument("--training-seeds", default="29,101,211,307,401")
    p.add_argument("--bootstrap-seed", type=int, default=197)
    p.add_argument("--bootstrap-replicates", type=int, default=4000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        seeds = tuple(int(s.strip()) for s in args.training_seeds.split(","))
    except ValueError as exc:
        p.error(f"Invalid training-seeds: {exc}")
    if not seeds or len(set(seeds)) != len(seeds):
        p.error("Training restart seeds must be distinct")
    if args.output.exists():
        p.error("Refusing to replace an existing analysis report")
    reports = [load_replica(args.study_root, seed) for seed in seeds]
    result = analyze_replicated_study(
        reports, bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    if [r["training_seed"] for r in result["replicas"]] != list(seeds):
        p.error("Study directory labels do not match actual restart seeds")
    atomic_write(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"[nature-p0-replicas] seeds={len(seeds)} "
        f"sources={result['heldout_source_run_count']} "
        f"ordinary_delta={result['ordinary_mean_score_delta']:+.4f} "
        f"two_axis_CI95={result['paired_training_seed_and_source_bootstrap_95pct']} "
        f"budget={result['budget_status']} "
        "decision_grade=False reference_law=UNVERIFIED",
        flush=True,
    )


if __name__ == "__main__":
    main()
