#!/usr/bin/env python3
"""Train and evaluate a first teacher-free neural Silent agent.

The actor sees only public observations and legal menus. The emulator
privately handles game RNG, and no exact-state search or posterior
calculation is used. Early decision caps produce censored runs and
NEVER fake defeat/terminal training labels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents import HeuristicAgent, NeuralGreedyAgent, RandomAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import (
    ENVIRONMENTS,
    LEGACY,
    NATIVE_OVERGROWTH,
    require_environment,
)
from sts2_ai.evaluation import RunSummary, play_run
from sts2_ai.evaluation.selfplay_metrics import (
    compare_completed_pairs,
    summarize_completed_runs,
)
from sts2_ai.training.selfplay import SELFPLAY_VERSION, TrainingRound, train_selfplay
from sts2_ai.training.rollout_failure import PublicRolloutFailure


def _progress(message: str) -> None:
    # Flush even when redirected or piped through tee. stderr keeps the
    # machine-readable final stdout JSON unaffected.
    print(message, file=sys.stderr, flush=True)


class _TrainingProgress:
    def __init__(self, total_rounds: int, episodes_per_round: int) -> None:
        self.total_rounds = total_rounds
        self.episodes_per_round = episodes_per_round
        self.started = time.perf_counter()
        self.round_started = self.started
        self.completed_this_invocation = 0
        self.wins_this_invocation = 0
        self.first_round_index: int | None = None

    def start(self, round_index: int, total_rounds: int, temperature: float) -> None:
        if self.first_round_index is None:
            self.first_round_index = round_index
            if round_index > 0:
                _progress(
                    f"[train] resumed at round {round_index + 1}/{total_rounds}; "
                    f"{round_index} rounds already in checkpoint"
                )
        self.round_started = time.perf_counter()
        _progress(
            f"[train] round {round_index + 1}/{total_rounds} starting | "
            f"episodes={self.episodes_per_round} | temperature={temperature:.5g}"
        )

    def complete(self, row: TrainingRound) -> None:
        self.completed_this_invocation += 1
        self.wins_this_invocation += row.wins
        elapsed = time.perf_counter() - self.started
        duration = time.perf_counter() - self.round_started
        remaining = self.total_rounds - row.round_index - 1
        eta = elapsed / self.completed_this_invocation * remaining
        progress = (
            f"{row.mean_progress:.1%}" if row.mean_progress is not None else "n/a"
        )
        loss = f"{row.mean_loss:.4f}" if row.mean_loss is not None else "n/a"
        _progress(
            f"[train] round {row.round_index + 1}/{self.total_rounds} complete | "
            f"completed={row.completed}/{row.played} censored={row.censored} "
            f"wins={row.wins} new_wins={self.wins_this_invocation} | "
            f"progress={progress} loss={loss} "
            f"updates={row.update_steps} decisions={row.decision_samples} | "
            f"round={duration:.1f}s elapsed={elapsed:.1f}s eta={eta:.0f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment", choices=ENVIRONMENTS, default=NATIVE_OVERGROWTH,
        help="Native Overgrowth REQUIRES an advertised emulator JSONL capability; "
             "legacy-prototype explicitly opts back into six-floor research runs",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--temperature-start", type=float, default=1.0)
    parser.add_argument("--temperature-end", type=float, default=1.0)
    parser.add_argument("--temperature-decay-rounds", type=int, default=1)
    parser.add_argument(
        "--initialize-from-model", type=Path,
        help="Initialize a NEW on-policy run from portable neural JSON; "
             "the optimizer and all training metrics start fresh",
    )
    parser.add_argument("--auxiliary-weight", type=float, default=0.4)
    parser.add_argument(
        "--win-anneal-threshold", type=int, default=16,
        help="Completed training victories required to anneal auxiliary return to zero",
    )
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Isolated .NET rollout processes; 15 is aggressive on 8-core/16-thread, 24GB hosts",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Atomic Torch checkpoint saved after each completed training round",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Restore model/AdamW/round metrics from --checkpoint (same config)",
    )
    parser.add_argument("--evaluate-seeds", type=int, default=3)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.evaluate_seeds < 1:
        parser.error("--evaluate-seeds must be positive")
    _progress(
        f"[train] opening emulator | environment={args.environment} "
        f"rounds={args.rounds} episodes/round={args.episodes} "
        f"workers={args.workers} seed={args.seed} "
        f"resume={args.resume} warm_start={args.initialize_from_model or 'none'}"
    )
    with JsonlEmulatorBackend(build=args.build) as backend:
        # Refuse before Torch initialization, game generation or checkpoint
        # writes if the pinned bridge lacks the requested native mode.
        require_environment(backend, args.environment)
        started_at = time.perf_counter()
        progress = _TrainingProgress(args.rounds, args.episodes)
        _progress(
            f"[train] emulator ready: revision={backend.emulator_revision} | "
            f"temperature={args.temperature_start:g}->{args.temperature_end:g} "
            f"over {args.temperature_decay_rounds} rounds"
        )
        try:
            trained = train_selfplay(
                backend,
                rounds=args.rounds, episodes_per_round=args.episodes,
                max_decisions=args.max_decisions,
                dimension=args.dimension, hidden=args.hidden,
                learning_rate=args.learning_rate,
                entropy_weight=args.entropy_weight,
                sampling_temperature_start=args.temperature_start,
                sampling_temperature_end=args.temperature_end,
                temperature_decay_rounds=args.temperature_decay_rounds,
                initialize_from_model=args.initialize_from_model,
                auxiliary_weight=args.auxiliary_weight,
                win_anneal_threshold=args.win_anneal_threshold,
                seed=args.seed,
                workers=args.workers,
                checkpoint_path=args.checkpoint,
                resume=args.resume,
                environment=args.environment,
                on_round_start=progress.start,
                on_round_complete=progress.complete,
            )
        except PublicRolloutFailure as failure:
            # Broken emulator transitions are not valid censored episodes.
            # Persist a replayable public history before aborting the cohort;
            # prior completed-round checkpoints remain intact.
            artifact = args.report.with_name(args.report.stem + ".failure.json")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            diagnostic = {
                **failure.report,
                "emulator_revision": backend.emulator_revision,
                "training_version": SELFPLAY_VERSION,
                "learner_seed": args.seed,
                "episodes_per_round": args.episodes,
                "dimension": args.dimension,
                "hidden": args.hidden,
                "checkpoint_path": (
                    str(args.checkpoint) if args.checkpoint is not None else None
                ),
            }
            artifact.write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
            )
            raise SystemExit(
                f"{failure}\nPublic replay saved: {artifact}\n"
                "Training stopped without assigning a reward to this episode. "
                "Completed-round checkpoints remain intact; an emulator "
                "revision change requires a fresh training experiment."
            ) from failure
        training_wall_seconds = time.perf_counter() - started_at
        _progress(
            f"[train] training complete | rounds={len(trained.rounds)} "
            f"total_episodes={sum(x.played for x in trained.rounds)} "
            f"wins={sum(x.wins for x in trained.rounds)} "
            f"elapsed_this_invocation={training_wall_seconds:.1f}s"
        )
        trained.model.save(args.output)
        _progress(
            f"[train] exported model={args.output} | "
            f"evaluating {args.evaluate_seeds} held-out seeds per agent"
        )
        contenders = {
            "random": RandomAgent(seed=args.seed + 999),
            "heuristic": HeuristicAgent(),
            "initial_neural_greedy": NeuralGreedyAgent(
                trained.initial_model, content_hash=trained.initial_model.model_id
            ),
            "neural_greedy": NeuralGreedyAgent(
                trained.model, content_hash=trained.model.model_id
            ),
        }
        raw_runs: dict[str, list[RunSummary]] = {}
        evaluations: dict[str, list[dict[str, object]]] = {}
        for name, agent in contenders.items():
            _progress(f"[eval] {name}: starting {args.evaluate_seeds} seeds")
            eval_started = time.perf_counter()
            raw_runs[name] = []
            evaluations[name] = []
            for index in range(args.evaluate_seeds):
                episode = play_run(
                    backend, agent, seed=f"selfplay-heldout-{index}",
                    policy=InformationPolicy(FAIR_POLICY_ID),
                    max_decisions=args.max_decisions,
                    environment=args.environment,
                )
                raw_runs[name].append(episode)
                if (index + 1) % min(args.evaluate_seeds, 16) == 0:
                    _progress(
                        f"[eval] {name}: {index + 1}/{args.evaluate_seeds} | "
                        f"elapsed={time.perf_counter() - eval_started:.1f}s"
                    )
                evaluations[name].append({
                    "seed": episode.seed,
                    "outcome": episode.outcome,
                    "censored": episode.censored,
                    "decisions": episode.decisions,
                    "act": episode.terminal_act,
                    "floor": episode.terminal_floor,
                    "frontier_progress": episode.frontier_progress,
                })
        completed_summaries = {
            name: summarize_completed_runs(runs)
            for name, runs in raw_runs.items()
        }
        for name, row in completed_summaries.items():
            _progress(
                f"[eval] {name}: completed={row['completed']} "
                f"censored={row['censored']} wins={row['wins']} "
                f"mean_frontier={row['mean_frontier_progress_completed_only']}"
            )
        paired_diagnostics = {
            f"neural_greedy_vs_{name}": compare_completed_pairs(
                runs, raw_runs["neural_greedy"]
            )
            for name, runs in raw_runs.items()
            if name != "neural_greedy"
        }
        report = {
            "schema": "sts2-onpolicy-neural-training-smoke-v2",
            "training_version": SELFPLAY_VERSION,
            "game_prior": (
                "native-structure-overgrowth-prototype-rng"
                if args.environment == NATIVE_OVERGROWTH
                else "ordinary-prototype-emulator-reset"
            ),
            "training_environment": args.environment,
            "policy_information": FAIR_POLICY_ID,
            "emulator_revision": backend.emulator_revision,
            "model_id": trained.model.model_id,
            "initial_model_id": trained.initial_model.model_id,
            "curriculum_win_anneal_threshold": args.win_anneal_threshold,
            "sampling_temperature_start": args.temperature_start,
            "sampling_temperature_end": args.temperature_end,
            "temperature_decay_rounds": args.temperature_decay_rounds,
            "entropy_weight": args.entropy_weight,
            "initialized_from_model": (
                str(args.initialize_from_model)
                if args.initialize_from_model is not None else None
            ),
            "rollout_workers": args.workers,
            "checkpoint_path": str(args.checkpoint) if args.checkpoint is not None else None,
            "resumed": args.resume,
            "training_wall_seconds_current_invocation": training_wall_seconds,
            "rounds": [asdict(row) for row in trained.rounds],
            "train_completed": sum(row.completed for row in trained.rounds),
            "train_censored": sum(row.censored for row in trained.rounds),
            "gradient_updates": sum(row.update_steps for row in trained.rounds),
            "gradient_decision_samples": sum(row.decision_samples for row in trained.rounds),
            "heldout_evaluation": evaluations,
            "heldout_completed_only": completed_summaries,
            "paired_completed_only": paired_diagnostics,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    _progress(f"[train] report saved: {args.report}")
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
