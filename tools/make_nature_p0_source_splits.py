#!/usr/bin/env python3
"""Produce auditable, source-group-disjoint P0 route *candidates*.

Every candidate must still finish against the pinned C# source route before
it is treated as a supported combat. No test result is implied by generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.nature.corpus import atomic_write
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_p0 import (
    SPLIT_SCHEMA,
    assert_disjoint_partitions,
    candidate_source_splits,
    read_partition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1731)
    parser.add_argument("--train", type=int, default=12800)
    parser.add_argument("--development", type=int, default=256)
    parser.add_argument("--evaluation", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = candidate_source_splits(
        seed=args.seed, train=args.train,
        development=args.development, evaluation=args.evaluation,
    )
    output = args.output
    for name in rows:
        if (output / f"{name}.jsonl").exists():
            parser.error(f"Refusing to replace existing {name} split")
    if (output / "candidate-manifest.json").exists():
        parser.error("Refusing to replace candidate split manifest")
    output.mkdir(parents=True, exist_ok=True)
    for name, items in rows.items():
        atomic_write(
            output / f"{name}.jsonl",
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in items),
        )
    partitions = {
        name: read_partition(output / f"{name}.jsonl", name) for name in rows
    }
    assert_disjoint_partitions(partitions)
    manifest = {
        "schema": SPLIT_SCHEMA,
        "seed": args.seed,
        "counts": {name: len(items) for name, items in rows.items()},
        "source_seeds_globally_disjoint": True,
        "source_validation": "CANDIDATES_NOT_YET_TESTED_IN_CSHARP",
        "splits_sha256": {name: canonical_sha256(items)
                          for name, items in rows.items()},
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write(
        output / "candidate-manifest.json",
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
    )
    print(
        f"[nature-p0-corpus] generated={sum(len(v) for v in rows.values())} "
        f"groups; heldout={len(rows['evaluation'])} "
        f"status=CANDIDATES_ONLY path={output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
