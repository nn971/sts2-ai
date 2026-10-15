#!/usr/bin/env python3
"""Collect deterministic, source-partitioned Nature HP/intent fixtures.

Runs the existing experimental isolated emulator interface. This command
cannot generate playable combat trajectories or native-reference p/q.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sts2_ai.nature.corpus import file_sha256, load_fixtures
from sts2_ai.nature.worker_corpus import collect_isolated_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="Versioned source-run partitioned Nature JSONL recipes")
    parser.add_argument("--emulator-dll", type=Path, required=True,
                        help="Built DLL from a separate v111 RNG integration checkout")
    parser.add_argument("--emulator-revision", required=True,
                        help="Exact emulator Git commit SHA corresponding to DLL")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nature-mode", choices=("first", "last", "uniform"),
                        default="uniform")
    parser.add_argument("--nature-seed", type=int, default=29)
    parser.add_argument("--max-choices", type=int, default=128)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.emulator_dll.is_file():
        parser.error("Emulator DLL does not exist; build a separate integration checkout")
    if not args.fixtures.is_file():
        parser.error("Nature fixture JSONL does not exist")
    output = collect_isolated_corpus(
        load_fixtures(args.fixtures), output_root=args.output_root,
        emulator_command=("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl"),
        emulator_revision=args.emulator_revision,
        emulator_artifact_sha256=file_sha256(args.emulator_dll),
        nature_mode=args.nature_mode, nature_seed=args.nature_seed,
        max_choices=args.max_choices, workers=args.workers,
        progress_every=args.progress_every, resume=args.resume,
    )
    print(
        f"[nature-n3] fixtures={output['fixture_count']} "
        f"decisions={output['decision_count']} "
        f"resume-ready={args.output_root}", flush=True,
    )


if __name__ == "__main__":
    main()
