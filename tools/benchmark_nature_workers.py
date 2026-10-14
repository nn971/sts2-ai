#!/usr/bin/env python3
"""Compare isolated Nature fixture throughput at 1/4/8/15 processes.

Each mode uses exactly the same source fixture manifest and Nature seed.
This checks deterministic *fixture* replay, not full player-combat training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.nature.corpus import atomic_write, file_sha256, load_fixtures
from sts2_ai.nature.worker_corpus import collect_isolated_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 15])
    parser.add_argument("--nature-seed", type=int, default=29)
    parser.add_argument("--nature-mode", choices=("uniform", "first", "last"), default="uniform")
    parser.add_argument("--max-choices", type=int, default=128)
    args = parser.parse_args()
    if not args.emulator_dll.is_file() or not args.fixtures.is_file():
        parser.error("Need real fixture corpus and built emulator DLL")
    if not args.workers or len(set(args.workers)) != len(args.workers) or min(args.workers) <= 0:
        parser.error("Workers must be unique positive counts")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Benchmark output must be empty")
    corpus = load_fixtures(args.fixtures)
    artifact = file_sha256(args.emulator_dll)
    rows = []
    expected: str | None = None
    for workers in args.workers:
        result = collect_isolated_corpus(
            corpus, output_root=args.output_root / f"workers-{workers}",
            emulator_command=("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl"),
            emulator_revision=args.emulator_revision, emulator_artifact_sha256=artifact,
            nature_mode=args.nature_mode, nature_seed=args.nature_seed,
            max_choices=args.max_choices, workers=workers,
        )
        digest = result["records_digest"]
        if expected is not None and expected != digest:
            raise ValueError(f"Nature worker-count divergence at {workers} workers")
        expected = digest
        seconds = result["execution"]["seconds_this_invocation"]
        row = {
            "workers": workers, "seconds": seconds,
            "fixtures": result["fixture_count"],
            "decisions": result["decision_count"],
            "fixtures_per_second": result["fixture_count"] / seconds,
            "worker_processes_observed": result["execution"][
                "worker_processes_observed_this_invocation"
            ],
            "records_digest": digest,
        }
        rows.append(row)
        print(f"[nature-throughput] {json.dumps(row, sort_keys=True)}", flush=True)
    report = {
        "schema": "sts2-ai-nature-isolated-workers-benchmark-v1",
        "emulator_revision": args.emulator_revision,
        "emulator_artifact_sha256": artifact,
        "nature_mode": args.nature_mode,
        "nature_seed": args.nature_seed,
        "worker_count_results": rows,
        "equal_records_across_workers": True,
        "scope": "isolated-combat-entry-only",
    }
    atomic_write(
        args.output_root / "benchmark.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(f"[nature-throughput] report={args.output_root / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
