#!/usr/bin/env python3
"""Collect reproducible multiworker playable Nature coverage and first blockers.

The emulator owns mechanics. This diagnostic uses only the existing LEGACY
indexed controlled fixture, not semantic deferred draw events. Terminal
outcomes are deliberately unclassified until the emulator supplies an
independently validated win/loss result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sts2_ai.nature.corpus import file_sha256, load_fixtures
from sts2_ai.nature.playable_corpus import collect_playable_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nature-mode", choices=("first", "last", "uniform"),
                        default="first")
    parser.add_argument("--nature-seed", type=int, default=29)
    parser.add_argument("--max-boundaries", type=int, default=512)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.emulator_dll.is_file():
        parser.error("Missing emulator DLL; use a separately built integration branch")
    if not args.fixtures.is_file():
        parser.error("Missing source-partitioned playable fixture JSONL")
    result = collect_playable_coverage(
        load_fixtures(args.fixtures),
        output_root=args.output_root,
        emulator_command=("dotnet", str(args.emulator_dll.resolve()),
                          "prototype-ai-jsonl"),
        emulator_revision=args.emulator_revision,
        emulator_artifact_sha256=file_sha256(args.emulator_dll),
        nature_mode=args.nature_mode, nature_seed=args.nature_seed,
        max_boundaries=args.max_boundaries, workers=args.workers,
        progress_every=args.progress_every, resume=args.resume,
    )
    for label, count in result["blocked_sites"].items():
        print(f"[nature-playable-site] {count:>5} {label}", flush=True)
    print(
        f"[nature-playable] coverage={result['status_counts']} "
        f"decisions={result['total_player_decisions']}"
        f"+{result['total_chance_decisions']} "
        f"output={args.output_root}", flush=True,
    )


if __name__ == "__main__":
    main()
