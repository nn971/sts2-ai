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
from sts2_ai.training.phase_split_selfplay import (
    BALANCED_TRAINING_VERSION,
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
        "--tactical-state-encoding", choices=("legacy", "structured"),
        default="structured",
        help="v4 structured tactical observation features, or legacy v3 for ablation",
    )
    p.add_argument("--temperature-start", type=float, default=0.05)
    p.add_argument("--temperature-end", type=float, default=0.035)
    p.add_argument("--temperature-decay-rounds", type=int, default=20)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--eval-seeds", type=int, default=16)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--warm-start", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--combat-samples-dir", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    if args.eval_seeds < 1:
        p.error("--eval-seeds must be positive")
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
        f"loss_normalization={args.loss_normalization} "
        f"combat_objective={args.combat_objective} "
        f"combat_advantage_baseline={combat_baseline} "
        f"resume={args.resume} warm_start={args.warm_start or 'none'}"
    )
    with JsonlEmulatorBackend(build=args.build) as backend:
        model, rows = train_phase_split(
            backend, rounds=args.rounds, episodes_per_round=args.episodes,
            workers=args.workers, dimension=args.dimension, hidden=args.hidden,
            learning_rate=args.learning_rate,
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
        )
        model.save(args.output)
        log(f"[split] model exported: {args.output}")
        agent = PhaseSplitGreedyAgent(model)
        evaluations = []
        for index in range(args.eval_seeds):
            item = play_run(
                backend, agent,
                seed=f"phase-split-heldout-v1-{index}",
                policy=InformationPolicy(FAIR_POLICY_ID),
                max_decisions=args.max_decisions,
                environment=NATIVE_OVERGROWTH,
            )
            evaluations.append({
                "seed": item.seed,
                "outcome": item.outcome,
                "act": item.terminal_act,
                "floor": item.terminal_floor,
                "progress": item.frontier_progress,
                "censored": item.censored,
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
                    seed=f"phase-split-heldout-v1-{index}",
                    policy=InformationPolicy(FAIR_POLICY_ID),
                    max_decisions=args.max_decisions,
                    environment=NATIVE_OVERGROWTH,
                )
                baseline_evaluations.append({
                    "seed": item.seed,
                    "outcome": item.outcome,
                    "act": item.terminal_act,
                    "floor": item.terminal_floor,
                    "progress": item.frontier_progress,
                    "censored": item.censored,
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
                BALANCED_TRAINING_VERSION if args.loss_normalization == "phase_mean"
                else SPLIT_TRAINING_VERSION
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
            "warm_start": str(args.warm_start) if args.warm_start else None,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        log(
            f"[split] done | report={args.report} "
            f"act1_clears={sum(x['act'] is not None and x['act'] >= 2 for x in evaluations)}"
        )


if __name__ == "__main__":
    main()
