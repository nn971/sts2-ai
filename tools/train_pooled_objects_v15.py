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
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import EPISODE_GOALS, NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.pooled_ppo_v15 import (
    POOLED_PPO_VERSION,
    collect_public_vocabulary,
    train_pooled_combat_ppo,
)


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
    args = parser.parse_args()
    if not args.base_model.is_file():
        parser.error("Base model path does not exist")
    if min(args.rounds, args.episodes, args.workers, args.vocabulary_runs) <= 0:
        parser.error("Rounds/episodes/workers/vocabulary-runs must be positive")

    base = PhaseSplitNeuralModel.load(args.base_model)
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
            f"resume={args.resume} frozen_base={args.base_model}", flush=True,
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
        )
    model_path = output / "model.json"
    report_path = output / "report.json"
    model.save(model_path)
    report_path.write_text(json.dumps({
        "schema": POOLED_PPO_VERSION,
        "architecture": args.architecture,
        "base_model": str(args.base_model),
        "vocabulary": str(vocab_path),
        "emulator_revision": backend.emulator_revision,
        "combat_objective": model.combat_value_objective,
        "rows": [asdict(row) for row in rows],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[v15-{args.architecture}] complete: model={model_path} report={report_path} "
        f"checkpoint={checkpoint}", flush=True,
    )


if __name__ == "__main__":
    main()
