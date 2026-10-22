#!/usr/bin/env python3
"""Paired confirmation-world benchmark of frozen Nature on 20-card A10 bosses.

Evaluates a frozen greedy Silent combat player with reference p and a frozen
previously trained Nature q on *identical conditioned opening worlds*. This is
evaluation ONLY, not training. The prototype floor-6 boss setting is a
research fixture, not native-game floor-16 or native-relic sampling fidelity.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.nature.combat_scenarios import parse_deck_pool, parse_scenario_pool
from sts2_ai.nature.contextual_draw import restore_nature
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.h2_benchmark import balanced_h2_source_seeds
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256
# Direct script invocation ("python tools/...") makes tools/ importable.
from train_draw_only_nature_h1 import (
    collect_paired_holdout,
    paired_holdout_metrics,
)

DEFAULT_BOSSES = "ceremonial-beast-boss-a10,vantom-boss-a10"
DEFAULT_DECKS = "silent-boss-cycle-20-v1,silent-boss-poison-20-v1"
SCHEMA = "sts2-ai-h3-boss-paired-evaluation-v1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-model", type=Path, required=True)
    p.add_argument("--nature-checkpoint", type=Path, required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--scenarios", default=DEFAULT_BOSSES)
    p.add_argument("--decks", default=DEFAULT_DECKS)
    p.add_argument("--split", choices=("selection", "confirmation"),
                   default="confirmation")
    p.add_argument("--eval-episodes", type=int, default=1000)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--reference-seed", type=int, default=17993)
    p.add_argument("--actor-seed", type=int, default=811)
    args = p.parse_args()
    scenarios = parse_scenario_pool(args.scenarios)
    decks = parse_deck_pool(args.decks)
    if not all(x.endswith("-boss-a10") for x in scenarios):
        p.error("This evaluator is restricted to A10 boss research scenarios")
    if not all("-boss-" in x and "-20-" in x for x in decks):
        p.error("This evaluator requires the verified 20-card boss deck presets")
    if (args.eval_episodes < 4 or
            args.eval_episodes % (len(scenarios) * len(decks)) or
            args.workers <= 0 or args.max_boundaries <= 0):
        p.error("Require positive budgets and exactly balanced boss/deck cells")
    binding = args.emulator_bindings/"sts2_emulator"/"__init__.py"
    if not (args.player_model.is_file() and
            args.nature_checkpoint.is_file() and
            args.emulator_dll.is_file() and binding.is_file()):
        p.error("Missing pinned player, Nature, emulator DLL or Python bindings")
    spec = FrozenPlayerSpec.from_file(
        args.player_model, seed=args.actor_seed, greedy=True,
        temperature=1.0, action_scope="all-legal-v1",
    )
    checkpoint = json.loads(args.nature_checkpoint.read_text(encoding="utf-8"))
    nature = restore_nature(checkpoint["nature"])
    if type(nature) is not PpoDrawNature:
        p.error("Boss study requires the checkpoint of a trained PPO Nature")
    seeds = balanced_h2_source_seeds(
        args.split, args.eval_episodes, scenarios, decks,
    )
    config = {
        "schema": SCHEMA,
        "prototype_map_boss_floor": 6,
        "source_native_boss_floor": 16,
        "source_relic_distribution_native_fidelity": "NOT_CLAIMED",
        "scenarios": list(scenarios),
        "decks": list(decks), "split": args.split,
        "episodes_per_arm": args.eval_episodes, "workers": args.workers,
        "max_boundaries": args.max_boundaries,
        "reference_seed": args.reference_seed,
        "actor_seed": args.actor_seed,
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(binding),
        "player_sha256": file_sha256(args.player_model),
        "nature_checkpoint_sha256": file_sha256(args.nature_checkpoint),
        "source_sha256": canonical_sha256(seeds),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root/"report.json"
    if report_path.exists():
        p.error("Immutable boss evaluation already exists in output root")
    common = (
        "dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl",
    )
    print(
        f"[boss-eval] cells={len(scenarios)*len(decks)} "
        f"sources={len(seeds)} player={spec.policy_id} "
        f"nature_revision={nature.revision} split={args.split}",
        flush=True,
    )
    learned, reference = collect_paired_holdout(
        seeds, nature=nature, spec=spec,
        emulator_command=common, emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=args.reference_seed, workers=args.workers,
        max_boundaries=args.max_boundaries,
        scenario_pool=scenarios, deck_pool=decks,
    )
    primary = learned["episodes"]
    ordinary = reference["episodes"]
    metrics = paired_holdout_metrics(
        primary, ordinary, bootstrap_seed=81911,
    )
    # In contrast to pure training metrics, this explicitly quantifies
    # whether the chosen boss fixture is too easy or too hard for v8.
    metrics["mean_reference_controlled_draws"] = statistics.mean(
        e.controlled_draw_count for e in ordinary
    )
    metrics["mean_nature_controlled_draws"] = statistics.mean(
        e.controlled_draw_count for e in primary
    )
    metrics["mean_reference_nontrivial_draws"] = statistics.mean(
        e.nontrivial_draw_count for e in ordinary
    )
    metrics["mean_nature_nontrivial_draws"] = statistics.mean(
        e.nontrivial_draw_count for e in primary
    )
    worlds = []
    for a, b in zip(primary, ordinary, strict=True):
        worlds.append({
            "source_seed": a.source_seed,
            "scenario_id": a.scenario_id,
            "deck_preset_id": a.deck_preset_id,
            "source_opening_sha256": a.source_opening_fingerprint,
            "player_model_id": a.player_model_id,
            "nature_score": a.score,
            "reference_score": b.score,
            "nature_victory": a.outcome == "victory",
            "reference_victory": b.outcome == "victory",
            "nature_exit_hp": a.exit_hp, "reference_exit_hp": b.exit_hp,
            "nature_nontrivial_draws": a.nontrivial_draw_count,
            "reference_nontrivial_draws": b.nontrivial_draw_count,
            "nature_replay_sha256": a.digest,
            "reference_replay_sha256": b.digest,
        })
    atomic_write(
        args.output_root/"paired-worlds.jsonl",
        "".join(json.dumps(w, sort_keys=True)+"\n" for w in worlds),
    )
    report = {
        "config": config, "config_sha256": canonical_sha256(config),
        "nature_revision": nature.revision,
        "paired_worlds_sha256": file_sha256(
            args.output_root/"paired-worlds.jsonl",
        ),
        "metrics": metrics,
        "research_verdict": "EVALUATION_ONLY",
    }
    atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(
        "[boss-eval] DONE "
        f"delta={metrics['paired_nature_minus_reference_score']:+.5f} "
        f"ci95={metrics['paired_source_bootstrap_95pct']} "
        f"ref_win={metrics['reference_victory_rate']:.3f} "
        f"nature_win={metrics['nature_victory_rate']:.3f} "
        f"reference_draws={metrics['mean_reference_nontrivial_draws']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
