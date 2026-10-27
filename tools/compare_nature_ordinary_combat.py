#!/usr/bin/env python3
"""Compare frozen-player combats under Nature and ordinary emulator RNG.

A source-route pair shares a pre-controller opening. The comparator commits
hidden shuffle order using its own native-v111 stream and never exposes extra
Nature events. Results are ENGINE-MODEL smoke data, not native-game parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.paired_real_combat import PairedSourceFixture
from sts2_ai.nature.paired_real_workers import paired_batch
from sts2_ai.nature.two_site_opponent import TwoSiteNature

# Each pair is a distinct source run. All five seeds below already appear
# in the emulator's source-route mechanics smoke; no inference of coverage
# for arbitrary seeds or late encounters is implied.
SMOKE_CASES = (
    PairedSourceFixture("overgrowth-default", "nature-p0-weak-real-victory"),
    PairedSourceFixture("overgrowth-beetle", "route-audit-01"),
    PairedSourceFixture("overgrowth-wurm", "route-audit-02"),
    PairedSourceFixture("underdocks-toad", "route-audit-01", "Underdocks"),
    PairedSourceFixture("underdocks-sludge", "route-audit-02", "Underdocks"),
)


def fixtures_from_jsonl(path: Path) -> tuple[PairedSourceFixture, ...]:
    rows = tuple(
        PairedSourceFixture(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows:
        raise ValueError("Empty paired combat source fixtures")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-bindings", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--source-fixtures", type=Path,
                        help="JSONL rows: fixture_id, seed, region")
    parser.add_argument("--player-model", type=Path,
                        help="Immutable real exported v15 model.json")
    parser.add_argument("--player-seed", type=int, default=2026)
    parser.add_argument("--player-temperature", type=float, default=1.0)
    parser.add_argument("--player-sampling", action="store_true")
    parser.add_argument("--player-action-scope",
                        choices=("silent-basic-v1", "all-legal-v1"),
                        default="silent-basic-v1")
    parser.add_argument("--nature-seed", type=int, default=29)
    parser.add_argument("--nature-mix", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-boundaries", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.emulator_dll.is_file():
        parser.error("Build the selected pinned emulator DLL first")
    if not args.emulator_revision or args.workers < 1:
        parser.error("Pinned emulator revision and positive workers required")
    if not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        parser.error("Emulator Python binding path is not present")

    fixtures = (
        fixtures_from_jsonl(args.source_fixtures)
        if args.source_fixtures is not None else SMOKE_CASES
    )
    player = (
        FrozenPlayerSpec.from_file(
            args.player_model, seed=args.player_seed,
            temperature=args.player_temperature,
            greedy=not args.player_sampling,
            action_scope=args.player_action_scope,
        ) if args.player_model is not None else None
    )
    report = paired_batch(
        fixtures, emulator_command=(
            "dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl",
        ),
        emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        nature=TwoSiteNature(seed=args.nature_seed, mix=args.nature_mix),
        player_spec=player, workers=args.workers,
        max_boundaries=args.max_boundaries, progress_every=1,
    )
    report["emulator_dll_sha256"] = file_sha256(args.emulator_dll)
    report["comparator"] = "engine-owned-native-v111-comparator-unverified-v1"
    atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "[nature-paired] DONE "
        f"source_groups={report['source_group_count']} "
        f"nature_victories={report['nature_victories']} "
        f"ordinary_victories={report['ordinary_victories']} "
        f"paired_delta={report['mean_paired_return_delta']:+.4f} "
        f"record_digest={report['records_digest']} "
        f"evidence=integration-smoke output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
