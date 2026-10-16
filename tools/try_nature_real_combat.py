#!/usr/bin/env python3
"""Run actual P0 Silent combat with external Nature over intents and draws.

This is a *mechanical/learning smoke*, not native-RNG player strength
evidence; only source-supported restricted encounter setups are eligible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.emulator.semantic_p0_combat import CombatFixture
from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.real_combat_workers import (
    collect_real_combat_batch,
    train_real_nature_batches,
)
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-bindings", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--mode", choices=("rollout", "nature-smoke"), default="rollout")
    parser.add_argument("--player-model", type=Path,
                        help="Portable frozen phase-split v15 model.json (read only)")
    parser.add_argument("--player-temperature", type=float, default=1.0)
    parser.add_argument("--player-sampling", action="store_true",
                        help="Sample frozen player logits, otherwise greedy")
    parser.add_argument("--player-actor-seed", type=int, default=2026)
    parser.add_argument(
        "--player-action-scope",
        choices=("silent-basic-v1", "all-legal-v1"),
        default="silent-basic-v1",
        help="Project frozen policy to audited basic cards for P0; full menu may abort",
    )
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--nature-mix", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.emulator_dll.is_file():
        parser.error("The pinned emulator CLI DLL must be built first")
    if not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        parser.error("Provide emulator's own checked-out bindings/python directory")
    if args.episodes < 2 or args.workers < 1:
        parser.error("Use at least two fixtures and a positive worker count")
    player_spec = (
        FrozenPlayerSpec.from_file(
            args.player_model, seed=args.player_actor_seed,
            temperature=args.player_temperature,
            greedy=not args.player_sampling,
            action_scope=args.player_action_scope,
        ) if args.player_model is not None else None
    )
    command = (
        "dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl",
    )
    if args.mode == "nature-smoke":
        report = train_real_nature_batches(
            rounds=args.rounds, batch_size=args.episodes, seed=args.seed,
            emulator_command=command, emulator_bindings=args.emulator_bindings,
            emulator_revision=args.emulator_revision, workers=args.workers,
            nature_mix=args.nature_mix, output=args.output,
            player_spec=player_spec,
        )
    else:
        fixtures = tuple(
            CombatFixture(
                fixture_id=f"act1-weak-{i:06d}",
                seed=f"nature-real-{args.seed}-{i}",
                act=1, encounter_id="proto.encounter.shrinker_beetle_weak",
                native_entry="ShrinkerBeetleWeak", total_floor=2,
            ) for i in range(args.episodes)
        )
        report = collect_real_combat_batch(
            fixtures, nature=TwoSiteNature(
                seed=args.seed, mix=args.nature_mix,
            ), emulator_command=command,
            emulator_bindings=args.emulator_bindings,
            emulator_revision=args.emulator_revision,
            workers=args.workers, progress_every=5,
            player_spec=player_spec,
        )
        report["emulator_dll_sha256"] = file_sha256(args.emulator_dll)
        report["evidence_level"] = "smoke-only"
        report["player_model_source"] = (
            "portable-v15-frozen-export" if player_spec is not None else
            "structural-strike-first"
        )
        atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[nature-real] mode={args.mode} "
        f"real_sts2_emulator=True source_law=UNVERIFIED "
        f"native_strength_evaluation=False "
        f"player={player_spec.policy_id if player_spec else 'structural'} "
        f"report={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
