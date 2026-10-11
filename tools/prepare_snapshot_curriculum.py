#!/usr/bin/env python3
"""Partition authentic combat snapshots by source run for opt-in v15 training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sts2_ai.evaluation.combat_snapshots import CombatSnapshotRecipe, load_recipes
from sts2_ai.training.snapshot_curriculum import source_key, split_snapshot_sources


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, text: str) -> None:
    """Fail rather than overwrite an immutable experimental partition."""
    with path.open("x", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())


def _rows(items: tuple[CombatSnapshotRecipe, ...]) -> str:
    return "".join(json.dumps(asdict(recipe), sort_keys=True) + "\n"
                   for recipe in items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--salt", default="sts2-v15m-snapshot-v1")
    args = parser.parse_args()
    recipes = load_recipes(args.corpus)
    train, holdout = split_snapshot_sources(
        recipes, holdout_fraction=args.holdout_fraction, salt=args.salt,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "train.jsonl", out / "development.jsonl", out / "manifest.json"]
    if any(path.exists() for path in paths):
        parser.error("Partition output already exists; choose a fresh directory")

    def profile(items: tuple[CombatSnapshotRecipe, ...]) -> dict:
        return {
            "entries": len(items),
            "source_runs": len({source_key(item) for item in items}),
            "progress_by_tier": {
                f"{p}/{t}": n
                for (p, t), n in sorted(Counter(
                    (r.progress, r.tier) for r in items
                ).items())
            },
        }

    _write_exclusive(paths[0], _rows(train))
    _write_exclusive(paths[1], _rows(holdout))
    manifest = {
        "schema": "sts2-v15m-snapshot-development-partition-v1",
        "source_corpus": str(args.corpus),
        "source_sha256": sha256(args.corpus),
        "source_runs": len({source_key(r) for r in recipes}),
        "holdout_fraction": args.holdout_fraction,
        "salt": args.salt,
        "note": "Development partition only. Prior-used source-A is not a fresh final holdout.",
        "train": {**profile(train), "sha256": sha256(paths[0])},
        "development": {**profile(holdout), "sha256": sha256(paths[1])},
    }
    _write_exclusive(paths[2], json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
