#!/usr/bin/env python3
"""Train a v15c pooled-object combat-only PPO continuation from a frozen v8 policy.

Collect a reusable public-observation vocabulary on separate training seeds,
then run actual on-policy selfplay PPO. The strategy head is frozen. The
v15c combat head is exported in the phase-split format, compatible with the
v14 combat snapshot evaluator. No hidden game state or new emulator APIs.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import load_recipes
from sts2_ai.emulator.episode_goal import EPISODE_GOALS, NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.pooled_ppo_v15 import (
    POOLED_PPO_VERSION,
    collect_public_vocabulary,
    train_pooled_combat_ppo,
)


def _atomic_json(path: Path, content: dict[str, Any]) -> None:
    """Commit report snapshots without publishing a half-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    try:
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_model(path: Path, model: PhaseSplitNeuralModel) -> None:
    """A named milestone is a complete independently loadable model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    try:
        model.save(pending)
        with pending.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("pooled", "relational"), default="pooled",
        help="v15c pooled control or v15d relation-aware attention",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/ppo-pooled-v15c"))
    parser.add_argument("--vocabulary", type=Path,
                        help="Frozen v15 vocabulary path (default: OUTPUT/vocabulary.json)")
    parser.add_argument("--vocabulary-runs", type=int, default=32)
    parser.add_argument("--vocabulary-seed-prefix", default="v15c-vocab-train")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--seed-prefix", default="v15c-pooled-train")
    parser.add_argument("--temperature-start", type=float, default=0.85)
    parser.add_argument("--temperature-end", type=float, default=0.5)
    parser.add_argument("--temperature-decay-rounds", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-batch-size", type=int, default=32)
    parser.add_argument("--ppo-sample-limit", type=int, default=2048)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-gamma", type=float, default=0.995)
    parser.add_argument("--ppo-gae-lambda", type=float, default=0.98)
    parser.add_argument("--max-decisions", type=int, default=4096)
    parser.add_argument("--episode-goal", choices=EPISODE_GOALS,
                        default=NATIVE_ACT1_BOSS_GOAL)
    parser.add_argument("--build", action="store_true",
                        help="Build pinned .NET emulator before starting")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initialize-from-model", type=Path,
        help="Fresh experiment warm-start from compatible exported v15 weights",
    )
    parser.add_argument(
        "--snapshot-corpus", type=Path,
        help="Opt-in replay-verified combat-only training; use a TRAIN partition",
    )
    parser.add_argument(
        "--snapshot-sampling", choices=("balanced", "natural"),
        default="balanced",
        help="Progress × encounter-tier curriculum, or natural corpus weights",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=20,
        help="Export portable model milestones every N completed rounds; 0 disables",
    )
    args = parser.parse_args()
    if not args.base_model.is_file():
        parser.error("Base model path does not exist")
    if min(args.rounds, args.episodes, args.workers, args.vocabulary_runs) <= 0:
        parser.error("Rounds/episodes/workers/vocabulary-runs must be positive")
    if args.snapshot_every < 0:
        parser.error("--snapshot-every must be nonnegative")
    if args.snapshot_corpus is not None and args.workers != 1:
        parser.error("Snapshot rollout mode currently requires --workers 1")
    if args.snapshot_corpus is not None and not args.snapshot_corpus.is_file():
        parser.error("Snapshot training corpus does not exist")
    if args.initialize_from_model is not None and not args.initialize_from_model.is_file():
        parser.error("Initialization model does not exist")
    snapshot_recipes = (
        tuple(load_recipes(args.snapshot_corpus))
        if args.snapshot_corpus is not None else None
    )

    base = PhaseSplitNeuralModel.load(args.base_model)
    initial = (
        PhaseSplitNeuralModel.load(args.initialize_from_model)
        if args.initialize_from_model is not None else None
    )
    if base.combat_value_objective != "hp_preservation":
        parser.error("Base combat critic must estimate hp_preservation")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    vocab_path = args.vocabulary or output / "vocabulary.json"
    checkpoint = output / "checkpoint.pt"
    if args.resume and not checkpoint.is_file():
        parser.error("--resume requires an existing checkpoint")
    if args.resume and not vocab_path.is_file():
        parser.error("--resume requires the original frozen vocabulary")
    if args.vocabulary is not None and not vocab_path.is_file():
        parser.error("Explicit --vocabulary path does not exist")

    training_settings = {
        "architecture": args.architecture,
        "width": args.width, "seed": args.seed,
        "seed_prefix": args.seed_prefix,
        "episodes_per_round": args.episodes, "workers": args.workers,
        "vocabulary_runs": args.vocabulary_runs,
        "vocabulary_seed_prefix": args.vocabulary_seed_prefix,
        "temperature_start": args.temperature_start,
        "temperature_end": args.temperature_end,
        "temperature_decay_rounds": args.temperature_decay_rounds,
        "learning_rate": args.learning_rate,
        "ppo_epochs": args.ppo_epochs,
        "ppo_batch_size": args.ppo_batch_size,
        "ppo_sample_limit": args.ppo_sample_limit,
        "ppo_clip_epsilon": args.ppo_clip_epsilon,
        "ppo_gamma": args.ppo_gamma,
        "ppo_gae_lambda": args.ppo_gae_lambda,
        "max_decisions": args.max_decisions,
        "episode_goal": args.episode_goal,
    }
    if initial is not None:
        training_settings["initialize_from_model"] = str(args.initialize_from_model)
    if snapshot_recipes is not None:
        training_settings["snapshot_corpus"] = str(args.snapshot_corpus)
        training_settings["snapshot_sampling"] = args.snapshot_sampling
        training_settings["rollout_source"] = "combat_snapshots"
    with JsonlEmulatorBackend(build=args.build) as backend:
        if vocab_path.is_file():
            vocabulary = ObjectVocabulary.from_dict(json.loads(
                vocab_path.read_text(encoding="utf-8")
            ))
            print(f"[v15c] loaded vocabulary from {vocab_path}", flush=True)
        else:
            print(
                f"[v15c] collecting {args.vocabulary_runs} public vocabulary runs "
                f"with frozen base policy", flush=True,
            )
            vocabulary = collect_public_vocabulary(
                backend, base, runs=args.vocabulary_runs,
                seed=args.seed, seed_prefix=args.vocabulary_seed_prefix,
                workers=args.workers, max_decisions=args.max_decisions,
                environment=NATIVE_OVERGROWTH,
                episode_goal_version=args.episode_goal,
                temperature=args.temperature_start,
            )
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            vocab_path.write_text(
                json.dumps(vocabulary.to_dict(), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[v15c] saved frozen vocabulary: {vocab_path}", flush=True)
        print(
            f"[v15c] PPO start: rounds={args.rounds} episodes={args.episodes} "
            f"workers={args.workers} seed={args.seed} goal={args.episode_goal} "
            f"resume={args.resume} frozen_base={args.base_model} "
            f"rollout={'combat_snapshots' if snapshot_recipes is not None else 'full_run'}",
            flush=True,
        )
        model_path = output / "model.json"
        report_path = output / "report.json"

        def make_report(history: tuple[Any, ...]) -> dict[str, Any]:
            return {
                "schema": POOLED_PPO_VERSION,
                "architecture": args.architecture,
                "base_model": str(args.base_model),
                "vocabulary": str(vocab_path),
                "emulator_revision": backend.emulator_revision,
                "combat_objective": base.combat_value_objective,
                "rounds_requested": args.rounds,
                "rounds_completed": len(history),
                "checkpoint": str(checkpoint),
                "snapshot_every": args.snapshot_every,
                "training_settings": training_settings,
                "rows": [asdict(row) for row in history],
            }

        def publish_round(history: tuple[Any, ...],
                          milestone: PhaseSplitNeuralModel | None) -> None:
            # Called only *after* checkpoint.pt is atomically committed.
            # Preserve a live report even when no portable milestone is due.
            _atomic_json(report_path, make_report(history))
            if milestone is not None:
                stage = output / "models" / f"stage-{len(history):04d}.json"
                _atomic_model(stage, milestone)
                _atomic_model(model_path, milestone)
                print(
                    f"[v15-{args.architecture}] exported milestone "
                    f"round={len(history)} model={stage}", flush=True,
                )

        model, rows = train_pooled_combat_ppo(
            backend, base, vocabulary,
            rounds=args.rounds, episodes_per_round=args.episodes,
            width=args.width, workers=args.workers, seed=args.seed,
            seed_prefix=args.seed_prefix, environment=NATIVE_OVERGROWTH,
            episode_goal_version=args.episode_goal,
            max_decisions=args.max_decisions,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            temperature_decay_rounds=args.temperature_decay_rounds,
            learning_rate=args.learning_rate, epochs=args.ppo_epochs,
            batch_size=args.ppo_batch_size,
            sample_limit=args.ppo_sample_limit,
            clip_epsilon=args.ppo_clip_epsilon,
            gamma=args.ppo_gamma, gae_lambda=args.ppo_gae_lambda,
            checkpoint=checkpoint, resume=args.resume,
            architecture=args.architecture,
            snapshot_every=args.snapshot_every,
            snapshot_recipes=snapshot_recipes,
            snapshot_sampling=args.snapshot_sampling,
            initialize_from=initial,
            on_round=publish_round,
        )
        _atomic_model(model_path, model)
        _atomic_json(report_path, make_report(rows))
    print(
        f"[v15-{args.architecture}] complete: model={model_path} report={report_path} "
        f"checkpoint={checkpoint}", flush=True,
    )


if __name__ == "__main__":
    main()
