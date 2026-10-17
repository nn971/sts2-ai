#!/usr/bin/env python3
"""H1: train draw-only Nature against one genuinely frozen v15 combat player.

A single fixed enemy/deck configuration is replicated across independent
source seeds. Intent choices remain distributed according to the emulator's
Chance Law; only physical draws learn. Training and held-out evaluation use
disjoint source seed namespaces. Default is a READ-ONLY plan. --execute opts
in to actual multiworker C# combat; outputs fail closed on unsupported sites.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import TwoSiteNature

SCHEMA = "sts2-ai-nature-h1-draw-only-fixed-combat-v1"


def source_seed(partition: str, index: int, *, training_seed: int) -> str:
    if partition not in ("train", "evaluation") or type(index) is not int or index < 0:
        raise ValueError("Invalid H1 source split")
    if partition == "evaluation":
        # Evaluation worlds remain identical across independently trained Natures.
        return f"nature-draw-h1-eval-fixed-shrinker-{index:08d}"
    return f"nature-draw-h1-train-{training_seed}-{index:08d}"


def _draw_statistics(episodes) -> dict:
    """Exact per-event categorical entropy/KL to the declared reference law."""
    distributions = [
        choice for episode in episodes for choice in episode.nature_choices
        if choice.operation == "draw_card"
    ]
    if not distributions:
        return {"draw_kl_q_to_p": None, "draw_entropy": None,
                "nature_draw_decision_count": 0}
    kls = []
    entropies = []
    for choice in distributions:
        pairs = zip(choice.probabilities, choice.baseline, strict=True)
        kl = 0.0
        entropy = 0.0
        for q, p in pairs:
            if q <= 0:
                continue
            if p <= 0:
                raise ValueError("Learned Nature escaped reference draw support")
            kl += q * math.log(q / p)
            entropy -= q * math.log(q)
        kls.append(kl)
        entropies.append(entropy)
    return {
        "draw_kl_q_to_p": statistics.mean(kls),
        "draw_entropy": statistics.mean(entropies),
        "nature_draw_decision_count": len(distributions),
    }


def paired_holdout_metrics(nature_episodes, reference_episodes, *,
                           bootstrap_seed: int, replicates: int = 2000) -> dict:
    if (not nature_episodes or len(nature_episodes) != len(reference_episodes)
            or replicates < 100):
        raise ValueError("Incomplete paired H1 holdout")
    gaps = []
    for nature, reference in zip(nature_episodes, reference_episodes, strict=True):
        if (nature.source_seed != reference.source_seed
                or nature.reference_seed != reference.reference_seed
                or nature.source_opening_fingerprint !=
                reference.source_opening_fingerprint
                or nature.player_model_id != reference.player_model_id
                or nature.draw_mode != "learned"
                or reference.draw_mode != "reference"):
            raise ValueError("H1 arms differ in source opening/player/intent seed")
        gaps.append(nature.score - reference.score)
    rng = random.Random(bootstrap_seed)
    draws = sorted(
        statistics.mean(gaps[rng.randrange(len(gaps))] for _ in gaps)
        for _ in range(replicates)
    )
    means = {
        "nature_mean_score": statistics.mean(x.score for x in nature_episodes),
        "reference_mean_score": statistics.mean(x.score for x in reference_episodes),
        "nature_victory_rate": statistics.mean(
            x.outcome == "victory" for x in nature_episodes
        ),
        "reference_victory_rate": statistics.mean(
            x.outcome == "victory" for x in reference_episodes
        ),
        "nature_nontrivial_draws": sum(
            x.nontrivial_draw_count for x in nature_episodes
        ),
        "reference_nontrivial_draws": sum(
            x.nontrivial_draw_count for x in reference_episodes
        ),
    }
    return {
        **means,
        **_draw_statistics(nature_episodes),
        "source_count": len(gaps),
        "paired_nature_minus_reference_score": statistics.mean(gaps),
        "paired_source_bootstrap_95pct": [
            draws[int(0.025 * replicates)],
            draws[min(replicates - 1, int(0.975 * replicates))],
        ],
        "bootstrap_replicates": replicates,
        "training_seed_uncertainty": "NOT_ESTIMATED_WITHIN_SINGLE_RESTART",
        "player_action_sampling_uncertainty": "NOT_SEPARATELY_ESTIMATED",
        "chance_probability_status": "UNVERIFIED",
        "decision_grade": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-model", type=Path, required=True)
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--emulator-bindings", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, default=29)
    parser.add_argument("--reference-seed", type=int, default=2027)
    parser.add_argument("--actor-seed", type=int, default=811)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--episodes-per-round", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--max-boundaries", type=int, default=600)
    parser.add_argument("--nature-mix", type=float, default=0.75)
    parser.add_argument("--nature-lr", type=float, default=0.08)
    parser.add_argument("--player-temperature", type=float, default=0.85)
    parser.add_argument("--player-action-scope",
                        choices=("silent-basic-v1", "all-legal-v1"),
                        default="silent-basic-v1")
    parser.add_argument("--sample-player", action="store_true")
    parser.add_argument("--allow-untrained-smoke", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute C# combat and optimizer steps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.rounds, args.episodes_per_round, args.eval_episodes,
            args.workers, args.max_boundaries) < 1
            or args.episodes_per_round < 2):
        parser.error("Require positive budgets and at least two combats per update")
    if (not 0 <= args.nature_mix < 1 or args.nature_lr <= 0
            or not math.isfinite(args.nature_lr)):
        parser.error("Invalid Nature optimizer parameters")
    if not args.emulator_dll.is_file():
        parser.error("Missing pinned emulator DLL")
    binding_file = args.emulator_bindings / "sts2_emulator" / "__init__.py"
    if not binding_file.is_file():
        parser.error("Missing versioned emulator Python binding")
    spec = FrozenPlayerSpec.from_file(
        args.player_model, seed=args.actor_seed,
        greedy=not args.sample_player,
        temperature=args.player_temperature,
        action_scope=args.player_action_scope,
    )
    model = spec.load_pinned()
    if model.combat_value_objective != "hp_preservation":
        parser.error("Frozen v15 model must use production HP-preservation objective")
    if (not args.allow_untrained_smoke
            and any(word in model.model_id.lower()
                    for word in ("untrained", "test-only", "synthetic"))):
        parser.error("Cannot call a test-only/untrained checkpoint a competent player")
    config = {
        "schema": SCHEMA,
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(binding_file),
        "player_model_sha256": spec.model_sha256,
        "player_model_id": model.model_id,
        "player_action_scope": args.player_action_scope,
        "player_greedy": spec.greedy,
        "player_temperature": spec.temperature,
        "actor_seed": args.actor_seed,
        "training_seed": args.training_seed,
        "reference_seed": args.reference_seed,
        "rounds": args.rounds,
        "episodes_per_round": args.episodes_per_round,
        "eval_episodes": args.eval_episodes,
        "workers": args.workers,
        "max_boundaries": args.max_boundaries,
        "nature_mix": args.nature_mix,
        "nature_lr": args.nature_lr,
        "entry_mode": "seeded-fixed-shrinker",
        "chance_distribution": "emulator-authored-v111-unverified",
        "player_training": False,
        "allow_untrained_smoke": args.allow_untrained_smoke,
        "holdout_namespace": "nature-draw-h1-eval-fixed-shrinker",
    }
    fingerprint = canonical_sha256(config)
    print(
        f"[draw-h1] plan={fingerprint[:16]} train_seed={args.training_seed} "
        f"train_combats={args.rounds*args.episodes_per_round} "
        f"holdout_combats_per_arm={args.eval_episodes} "
        f"workers={args.workers} execute={args.execute}",
        flush=True,
    )
    if not args.execute:
        print(json.dumps(config, sort_keys=True, indent=2), flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "checkpoint.json"
    report_path = args.output_root / "holdout-report.json"
    if report_path.exists():
        parser.error("Immutable completed H1 report already exists")
    if args.resume:
        if not checkpoint.is_file():
            parser.error("No H1 checkpoint found for resume")
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("schema") != SCHEMA or saved.get("config_digest") != fingerprint:
            parser.error("Cannot resume against changed emulator/model/source budget")
        nature = TwoSiteNature.restore(saved["nature"])
        metrics = saved["metrics"]
        if len(metrics) != nature.revision or len(metrics) > args.rounds:
            parser.error("Inconsistent H1 round/Nature revision checkpoint")
        first_round = len(metrics)
    else:
        if any(args.output_root.iterdir()):
            parser.error("H1 output root is not empty; use explicit --resume")
        nature = TwoSiteNature(
            seed=args.training_seed + 100,
            mix=args.nature_mix, learning_rate=args.nature_lr,
        )
        metrics = []
        first_round = 0
    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")
    for round_index in range(first_round, args.rounds):
        source_start = round_index * args.episodes_per_round
        seeds = tuple(
            source_seed("train", source_start + i,
                        training_seed=args.training_seed)
            for i in range(args.episodes_per_round)
        )
        batch = draw_only_batch(
            seeds, nature=nature, spec=spec,
            emulator_command=command, emulator_bindings=args.emulator_bindings,
            emulator_revision=args.emulator_revision,
            reference_seed=args.reference_seed + source_start * 1000003,
            draw_mode="learned", workers=args.workers,
            max_boundaries=args.max_boundaries,
        )
        if batch["nontrivial_draws"] <= 0:
            raise RuntimeError("No nontrivial physical draws: invalid H1 training fixture")
        episodes = batch["episodes"]
        metrics_before = {
            "round": round_index,
            "mean_player_score": statistics.mean(x.score for x in episodes),
            "victory_rate": statistics.mean(x.outcome == "victory" for x in episodes),
            "combat_count": len(episodes),
            "physical_draw_count": batch["total_draws"],
            "nontrivial_draw_count": batch["nontrivial_draws"],
            "reference_intent_count": batch["total_intents"],
            "observed_emulator_process_count": batch["process_count"],
            "wall_seconds": batch["elapsed_seconds"],
            "behavior_nature_revision": nature.revision,
            "source_group_sha256": canonical_sha256(seeds),
            **_draw_statistics(episodes),
        }
        optimization = nature.update(tuple(x.for_nature_training() for x in episodes))
        metrics.append({**metrics_before, **optimization})
        record = {
            "schema": SCHEMA, "config": config, "config_digest": fingerprint,
            "nature": nature.dump(), "metrics": metrics,
            "next_round": round_index + 1,
            "research_verdict": "NOT_ESTABLISHED",
        }
        atomic_write(checkpoint, json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(
            f"[draw-h1] round={round_index+1}/{args.rounds} "
            f"score={metrics_before['mean_player_score']:.4f} "
            f"nontrivial_draws={batch['nontrivial_draws']} "
            f"nature_grad={optimization['nature_gradient_norm']:.5f} "
            f"nature_rev={nature.revision} "
            f"emulator_pids={batch['process_count']}", flush=True,
        )
    evaluation_seeds = tuple(
        source_seed("evaluation", i, training_seed=args.training_seed)
        for i in range(args.eval_episodes)
    )
    primary = draw_only_batch(
        evaluation_seeds, nature=nature, spec=spec,
        emulator_command=command, emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=args.reference_seed + 987654321,
        draw_mode="learned", workers=args.workers,
        max_boundaries=args.max_boundaries,
    )
    control = draw_only_batch(
        evaluation_seeds, nature=nature, spec=spec,
        emulator_command=command, emulator_bindings=args.emulator_bindings,
        emulator_revision=args.emulator_revision,
        reference_seed=args.reference_seed + 987654321,
        draw_mode="reference", workers=args.workers,
        max_boundaries=args.max_boundaries,
    )
    analysis = paired_holdout_metrics(
        primary["episodes"], control["episodes"],
        bootstrap_seed=991 + args.training_seed,
    )
    learned_path = args.output_root / "evaluation-learned.jsonl"
    reference_path = args.output_root / "evaluation-reference.jsonl"
    if learned_path.exists() or reference_path.exists():
        parser.error("Existing evaluation traces are immutable")
    atomic_write(
        learned_path,
        "".join(json.dumps(asdict(row), sort_keys=True) + "\\n"
                for row in primary["episodes"]),
    )
    atomic_write(
        reference_path,
        "".join(json.dumps(asdict(row), sort_keys=True) + "\\n"
                for row in control["episodes"]),
    )
    report = {
        "schema": SCHEMA, "config": config, "config_digest": fingerprint,
        "evaluation_learned_sha256": file_sha256(learned_path),
        "evaluation_reference_sha256": file_sha256(reference_path),
        "final_nature_checkpoint_sha256": canonical_sha256(nature.dump()),
        "evaluation_source_sha256": canonical_sha256(evaluation_seeds),
        "metrics": analysis,
        "train_rounds": len(metrics),
        "training_seed": args.training_seed,
        "research_verdict": "NOT_ESTABLISHED",
        "decision_grade": False,
        "installed_game_law": "UNVERIFIED",
        "nature_evaluation_digests": [
            x.digest for x in primary["episodes"]
        ],
        "reference_evaluation_digests": [
            x.digest for x in control["episodes"]
        ],
    }
    atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[draw-h1] FINISHED TRAINED NATURE versus emulator p: "
        f"delta={analysis['paired_nature_minus_reference_score']:+.5f} "
        f"CI95={analysis['paired_source_bootstrap_95pct']} "
        f"sources={args.eval_episodes} status=UNVERIFIED decision_grade=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
