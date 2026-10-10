#!/usr/bin/env python3
"""Experimental two-head Silent self-play; prints each training round."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents import NeuralGreedyAgent
from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.emulator.episode_goal import (
    EPISODE_GOALS, NATIVE_ACT1_BOSS_GOAL,
)
from sts2_ai.training.phase_split_selfplay import (
    BALANCED_TRAINING_VERSION,
    PPO_TRAINING_VERSION,
    SPLIT_TRAINING_VERSION,
    PhaseSplitRound,
    train_phase_split,
)


def log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--optimizer-method", choices=("reinforce", "ppo"), default="reinforce")
    p.add_argument("--ppo-backend", choices=("reference", "batched"), default="reference",
                   help="Reference per-decision PPO or cached batched PPO")
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-batch-size", type=int, default=128)
    p.add_argument("--ppo-sample-limit", type=int, default=4096)
    p.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    p.add_argument("--ppo-gamma", type=float, default=0.995)
    p.add_argument("--ppo-gae-lambda", type=float, default=0.98)
    p.add_argument(
        "--train-seed-prefix", default="phase-split-v1",
        help="Isolate training seeds from prior experiments; stable across resumes",
    )
    p.add_argument(
        "--loss-normalization",
        choices=("phase_mean", "legacy_episode_sum"),
        default="phase_mean",
        help="Balance the strategy/combat gradient by decisions; legacy reproduces v4",
    )
    p.add_argument("--boundary-weight", type=float, default=0.25)
    p.add_argument(
        "--combat-objective", choices=("hp_preservation", "hp_first", "continuation"),
        default="hp_preservation",
        help="Survival and within-combat HP preserved, or original objective",
    )
    p.add_argument(
        "--combat-advantage-baseline", choices=("critic", "leave_one_run_out"),
        help="HP combat policy baseline (default: leave-one-run-out for preservation)",
    )
    p.add_argument("--hp-monotonic-weight", type=float, default=0.2)
    p.add_argument(
        "--tactical-state-encoding",
        choices=("legacy", "structured", "relational", "relational_damage", "public_resources", "enemy_instances"),
        default="relational",
        help="v5/v6 relational, v7 resources, or v8 species-aware enemy instances",
    )
    p.add_argument("--temperature-start", type=float, default=0.05)
    p.add_argument("--temperature-end", type=float, default=0.035)
    p.add_argument("--temperature-decay-rounds", type=int, default=20)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--win-anneal-threshold", type=int, default=16,
                   help="Number of training victories before auxiliary shaping vanishes")
    p.add_argument("--eval-seeds", type=int, default=16)
    p.add_argument(
        "--episode-goal", choices=EPISODE_GOALS, default=NATIVE_ACT1_BOSS_GOAL,
        help="Certified Act-1 boss clear is terminal victory, or historical prototype goal",
    )
    p.add_argument(
        "--eval-seed-prefix", default="phase-split-heldout-v1",
        help="Stable heldout seed prefix; use a fresh prefix for new experiments",
    )
    p.add_argument("--seed", type=int, default=19)
    p.add_argument(
        "--exclude-train-seed", action="append", default=[],
        help="Explicitly omit unsupported run seeds from training; recorded in report",
    )
    p.add_argument("--warm-start", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--combat-samples-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    if args.win_anneal_threshold < 1:
        p.error("--win-anneal-threshold must be positive")
    if args.eval_seeds < 1:
        p.error("--eval-seeds must be positive")
    if not args.eval_seed_prefix:
        p.error("--eval-seed-prefix must not be empty")
    started = time.perf_counter()
    combat_baseline = args.combat_advantage_baseline or (
        "leave_one_run_out" if args.combat_objective == "hp_preservation"
        else "critic"
    )

    def progress(row: PhaseSplitRound) -> None:
        log(
            f"[split] round {row.round_index + 1}/{args.rounds} | "
            f"runs={row.completed}/{row.played} censored={row.censored} "
            f"wins={row.wins} combats={row.combat_victories}W/"
            f"{row.combat_defeats}L "
            f"actions=strategy:{row.strategic_decisions}/"
            f"combat:{row.tactical_decisions} "
            f"boundary_target={row.mean_combat_boundary_target} "
            f"hp_pairs={row.hp_monotonic_pairs} "
            f"victory_exit_hp_fraction={row.mean_victory_exit_hp_fraction} "
            f"victory_hp_lost_fraction={row.mean_victory_hp_lost_fraction} "
            f"potions_used_per_combat={row.mean_potions_used_per_combat} "
            f"run_target={row.mean_run_return} loss={row.mean_loss} "
            f"phase_diagnostics={row.phase_loss_diagnostics} "
            f"updates={row.optimization_steps} "
            f"rollouts={row.rollout_seconds:.1f}s "
            f"optimizer={row.optimizer_seconds:.1f}s "
            f"elapsed={time.perf_counter() - started:.1f}s"
        )

    log(
        f"[split] start | rounds={args.rounds} episodes={args.episodes} "
        f"workers={args.workers} boundary_weight={args.boundary_weight} "
        f"tactical_state_encoding={args.tactical_state_encoding} "
        f"optimizer_method={args.optimizer_method} ppo_backend={args.ppo_backend} "
        f"loss_normalization={args.loss_normalization} "
        f"combat_objective={args.combat_objective} "
        f"combat_advantage_baseline={combat_baseline} "
        f"resume={args.resume} goal={args.episode_goal} "
        f"win_anneal_threshold={args.win_anneal_threshold} "
        f"warm_start={args.warm_start or 'none'}"
    )
    with JsonlEmulatorBackend(build=args.build) as backend:
        model, rows = train_phase_split(
            backend, rounds=args.rounds, episodes_per_round=args.episodes,
            workers=args.workers, dimension=args.dimension, hidden=args.hidden,
            learning_rate=args.learning_rate,
            optimizer_method=args.optimizer_method,
            ppo_backend=args.ppo_backend,
            ppo_epochs=args.ppo_epochs,
            ppo_batch_size=args.ppo_batch_size,
            ppo_sample_limit=args.ppo_sample_limit,
            ppo_clip_epsilon=args.ppo_clip_epsilon,
            ppo_gamma=args.ppo_gamma,
            ppo_gae_lambda=args.ppo_gae_lambda,
            seed_prefix=args.train_seed_prefix,
            boundary_weight=args.boundary_weight,
            hp_monotonic_weight=args.hp_monotonic_weight,
            tactical_state_encoding=args.tactical_state_encoding,
            loss_normalization=args.loss_normalization,
            combat_objective=args.combat_objective,
            combat_advantage_baseline=combat_baseline,
            temperature_start=args.temperature_start,
            temperature_end=args.temperature_end,
            temperature_decay_rounds=args.temperature_decay_rounds,
            max_decisions=args.max_decisions,
            seed=args.seed, warm_start=args.warm_start,
            checkpoint=args.checkpoint, resume=args.resume,
            combat_samples_dir=args.combat_samples_dir,
            environment=NATIVE_OVERGROWTH, progress=progress,
            excluded_training_seeds=frozenset(args.exclude_train_seed),
            episode_goal_version=args.episode_goal,
            win_anneal_threshold=args.win_anneal_threshold,
        )
        model.save(args.output)
        log(f"[split] model exported: {args.output}")
        agent = PhaseSplitGreedyAgent(model)
        evaluations = []
        for index in range(args.eval_seeds):
            item = play_run(
                backend, agent,
                seed=f"{args.eval_seed_prefix}-{index}",
                policy=InformationPolicy(FAIR_POLICY_ID),
                max_decisions=args.max_decisions,
                environment=NATIVE_OVERGROWTH,
                episode_goal_version=args.episode_goal,
            )
            evaluations.append({
                "seed": item.seed,
                "outcome": item.outcome,
                "act": item.terminal_act,
                "floor": item.terminal_floor,
                "progress": item.frontier_progress,
                "censored": item.censored,
                "act1_cleared": item.act1_cleared,
                "full_game_victory": item.full_game_victory,
                "episode_goal_version": item.episode_goal_version,
                "boss": (
                    asdict(item.boss_progress)
                    if item.boss_progress is not None else None
                ),
            })
            log(
                f"[split-eval] {index + 1}/{args.eval_seeds} "
                f"act={item.terminal_act} floor={item.terminal_floor} "
                f"outcome={item.outcome}"
            )
        baseline_evaluations = []
        if args.warm_start is not None:
            raw = json.loads(args.warm_start.read_text())
            baseline_agent = (
                PhaseSplitGreedyAgent.load(args.warm_start)
                if raw.get("format") == "sts2-phase-split-policy-value-v1"
                else NeuralGreedyAgent.load(args.warm_start)
            )
            for index in range(args.eval_seeds):
                item = play_run(
                    backend, baseline_agent,
                    seed=f"{args.eval_seed_prefix}-{index}",
                    policy=InformationPolicy(FAIR_POLICY_ID),
                    max_decisions=args.max_decisions,
                    environment=NATIVE_OVERGROWTH,
                    episode_goal_version=args.episode_goal,
                )
                baseline_evaluations.append({
                    "seed": item.seed,
                    "outcome": item.outcome,
                    "act": item.terminal_act,
                    "floor": item.terminal_floor,
                    "progress": item.frontier_progress,
                    "censored": item.censored,
                    "act1_cleared": item.act1_cleared,
                    "full_game_victory": item.full_game_victory,
                    "episode_goal_version": item.episode_goal_version,
                    "boss": (
                        asdict(item.boss_progress)
                        if item.boss_progress is not None else None
                    ),
                })
                log(
                    f"[baseline-eval] {index + 1}/{args.eval_seeds} "
                    f"act={item.terminal_act} floor={item.terminal_floor} "
                    f"outcome={item.outcome}"
                )
        paired = (
            [
                new["progress"] - old["progress"]
                for new, old in zip(evaluations, baseline_evaluations, strict=True)
                if not new["censored"] and not old["censored"]
            ]
            if baseline_evaluations else []
        )
        report = {
            "schema": "sts2-phase-split-experiment-v1",
            "training_version": (
                PPO_TRAINING_VERSION if args.optimizer_method == "ppo"
                else BALANCED_TRAINING_VERSION
                if args.loss_normalization == "phase_mean"
                else SPLIT_TRAINING_VERSION
            ),
            "optimizer_method": args.optimizer_method,
            "ppo_backend": args.ppo_backend,
            "episode_goal_version": args.episode_goal,
            "win_anneal_threshold": args.win_anneal_threshold,
            "train_seed_prefix": args.train_seed_prefix,
            "excluded_training_seeds": sorted(set(args.exclude_train_seed)),
            "ppo_hyperparameters": (
                {
                    "epochs": args.ppo_epochs,
                    "batch_size": args.ppo_batch_size,
                    "sample_limit": args.ppo_sample_limit,
                    "clip_epsilon": args.ppo_clip_epsilon,
                    "gamma": args.ppo_gamma,
                    "gae_lambda": args.ppo_gae_lambda,
                } if args.optimizer_method == "ppo" else None
            ),
            "loss_normalization": args.loss_normalization,
            "combat_objective": args.combat_objective,
            "combat_advantage_baseline": combat_baseline,
            "model_id": model.model_id,
            "emulator_revision": backend.emulator_revision,
            "environment": NATIVE_OVERGROWTH,
            "rounds": [asdict(r) for r in rows],
            "heldout": evaluations,
            "warm_start_heldout": baseline_evaluations,
            "paired_completed_only": {
                "pairs": len(paired),
                "mean_frontier_delta": (
                    sum(paired) / len(paired) if paired else None
                ),
                "improved": sum(x > 0 for x in paired),
                "worsened": sum(x < 0 for x in paired),
                "tied": sum(x == 0 for x in paired),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "boundary_weight": args.boundary_weight,
            "combat_samples_dir": (
                str(args.combat_samples_dir)
                if args.combat_samples_dir else None
            ),
            "hp_monotonic_weight": args.hp_monotonic_weight,
            "tactical_state_encoding": args.tactical_state_encoding,
            "eval_seed_prefix": args.eval_seed_prefix,
            "warm_start": str(args.warm_start) if args.warm_start else None,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        log(
            f"[split] done | report={args.report} "
            f"act1_clears={sum(x['act1_cleared'] is True for x in evaluations)}"
        )


if __name__ == "__main__":
    main()
