#!/usr/bin/env python3
"""H6: on-policy macro PPO on native-structure Silent A10 Overgrowth runs.

Only the semantic strategy head is optimized. Every player combat action is
chosen by the *greedy frozen combat backbone*. The emulator supplies all
outcomes; native RNG/fidelity inspection is not a training prerequisite.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.frozen_macro import FrozenCombatMacroPPO, combat_digest
from sts2_ai.training.h6_sources import H6_SOURCE_SCHEMA, run_sources, source_digest
from sts2_ai.training.parallel_rollouts import collect_parallel, open_worker_pool
from sts2_ai.training.selfplay import Episode, _bounded_return
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.protocol import canonical_sha256

SCHEMA = "sts2-h6-macro-native-overgrowth-training-v1"


def _atomic_torch(path: Path, payload: dict[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".pending", delete=False
        ) as stream:
            pending = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def summarize(episodes: tuple[Episode, ...]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("Empty H6 run cohort")
    completed = [ep for ep in episodes if ep.completed]
    censored = [ep for ep in episodes if not ep.completed]
    if any(ep.outcome != "truncated" for ep in censored):
        raise RuntimeError("H6 nonterminal run reported as completed outcome")
    return {
        "runs": len(episodes),
        "completed": len(completed),
        "censored": len(censored),
        "wins": sum(ep.won for ep in completed),
        "completion_fraction": len(completed) / len(episodes),
        "win_rate_completed": (
            sum(ep.won for ep in completed) / len(completed)
            if completed else None
        ),
        "win_fraction_all": sum(ep.won for ep in completed) / len(episodes),
        "mean_floor_completed": (
            statistics.mean(ep.floor or 0 for ep in completed)
            if completed else None
        ),
        "mean_hp_fraction_completed": (
            statistics.mean(ep.hp_ratio for ep in completed)
            if completed else None
        ),
        "strategy_decisions": sum(
            dec.phase == "strategy" for ep in episodes for dec in ep.decisions
        ),
        "combat_decisions": sum(
            dec.phase == "combat" for ep in episodes for dec in ep.decisions
        ),
        "combat_victories": sum(
            segment.result == "victory"
            for ep in episodes for segment in ep.combat_outcomes
        ),
        "combat_defeats": sum(
            segment.result == "defeat"
            for ep in episodes for segment in ep.combat_outcomes
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player-model", type=Path, required=True,
                   help="Frozen H4/H5 two-head JSON export")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--emulator-root", type=Path, default=Path("emulator"))
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--episodes-per-round", type=int, default=32)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--study-seed", type=int, default=43)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--player-lr", type=float, default=0.0003)
    p.add_argument("--strategy-temperature", type=float, default=0.85)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-batch-size", type=int, default=128)
    p.add_argument("--ppo-sample-limit", type=int, default=4096)
    p.add_argument("--defeat-progress-weight", type=float, default=0.35)
    p.add_argument("--progress-anneal-wins", type=int, default=80)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (min(args.rounds, args.episodes_per_round, args.workers,
            args.max_decisions, args.eval_every, args.eval_episodes,
            args.ppo_epochs, args.ppo_batch_size, args.ppo_sample_limit,
            args.progress_anneal_wins) < 1
            or args.study_seed < 0
            or args.player_lr <= 0 or args.player_lr >= 1
            or args.strategy_temperature <= 0
            or not 0 <= args.defeat_progress_weight <= 1):
        p.error("Invalid H6 macro settings")

    initial = PhaseSplitNeuralModel.load(args.player_model)
    emulator_root = args.emulator_root.resolve()
    repo_root = args.repo_root.resolve()
    cli = emulator_root/"src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj"
    if not cli.is_file():
        p.error("Missing emulator checkout or semantic CLI project")
    config = {
        "schema": SCHEMA,
        "source_schema": H6_SOURCE_SCHEMA,
        "player_model_sha256": file_sha256(args.player_model),
        "frozen_combat_sha256": combat_digest(initial),
        "emulator_revision": args.emulator_revision,
        "environment": NATIVE_OVERGROWTH,
        "goal": NATIVE_ACT1_BOSS_GOAL,
        "ascension": 10,
        "draw_and_intent_law": "emulator-reference",
        "combat_action_rule": "frozen-greedy",
        "strategy_action_rule_train": "on-policy-temperature",
        "strategy_action_rule_eval": "greedy",
        "rounds": args.rounds,
        "episodes_per_round": args.episodes_per_round,
        "workers": args.workers,
        "max_decisions": args.max_decisions,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "study_seed": args.study_seed,
        "actor_seed": args.actor_seed,
        "player_lr": args.player_lr,
        "strategy_temperature": args.strategy_temperature,
        "ppo_epochs": args.ppo_epochs,
        "ppo_batch_size": args.ppo_batch_size,
        "ppo_sample_limit": args.ppo_sample_limit,
        "defeat_progress_weight": args.defeat_progress_weight,
        "progress_anneal_wins": args.progress_anneal_wins,
    }
    fingerprint = canonical_sha256(config)
    print(
        f"[h6] config={fingerprint[:16]} runs="
        f"{args.rounds*args.episodes_per_round} workers={args.workers} "
        f"native-overgrowth=A10 execute={args.execute}",
        flush=True,
    )
    if not args.execute:
        print(json.dumps(config, indent=2, sort_keys=True))
        return

    import torch
    torch.set_num_threads(1)
    torch.manual_seed(args.study_seed)
    learner = FrozenCombatMacroPPO(
        initial, torch=torch, learning_rate=args.player_lr,
        temperature=args.strategy_temperature,
        epochs=args.ppo_epochs, batch_size=args.ppo_batch_size,
        sample_limit=args.ppo_sample_limit,
    )
    checkpoint = args.output_root/"macro-checkpoint.pt"
    models_dir = args.output_root/"player_models"
    rows: list[dict[str, Any]] = []
    if args.resume:
        if not checkpoint.is_file():
            p.error("H6 resume requires existing macro-checkpoint.pt")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("schema") != SCHEMA or saved.get("fingerprint") != fingerprint:
            p.error("H6 checkpoint incompatible with pinned model/config")
        learner.restore(saved["learner"])
        rows = saved["rows"]
        if learner.revision != len(rows) or len(rows) > args.rounds:
            p.error("H6 resume revision/row mismatch")
        torch.set_rng_state(saved["torch_rng_state"])
    elif checkpoint.exists() or (
        models_dir.is_dir() and any(models_dir.iterdir())
    ):
        p.error("H6 output already contains training; use --resume")
    args.output_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    backend = JsonlEmulatorBackend(
        repo_root=repo_root, emulator_root=emulator_root,
        expected_emulator_revision=args.emulator_revision, build=False,
    )
    wins_so_far = sum(row["train"]["wins"] for row in rows)
    try:
        with ExitStack() as scope:
            pool = scope.enter_context(open_worker_pool(backend, args.workers))
            for index in range(len(rows), args.rounds):
                model = learner.model()
                behavior_path = models_dir/f"player-r{index:04d}.json"
                atomic_write(
                    behavior_path,
                    json.dumps(model.to_dict(), sort_keys=True)+"\n",
                )
                seeds = run_sources(
                    "train", args.episodes_per_round,
                    study_seed=args.study_seed, round_index=index,
                )
                started = time.perf_counter()
                episodes = collect_parallel(
                    pool, model, seeds, base_seed=args.actor_seed,
                    max_decisions=args.max_decisions,
                    policy_id="prototype-fair-v0",
                    environment=NATIVE_OVERGROWTH,
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                    temperature=args.strategy_temperature,
                    ascension=10, combat_greedy=True,
                )
                elapsed = time.perf_counter() - started
                completed = [ep for ep in episodes if ep.completed]
                alpha = args.defeat_progress_weight * max(
                    0.0, 1.0 - wins_so_far / args.progress_anneal_wins,
                )
                targets = [
                    (ep, _bounded_return(ep, alpha)) for ep in completed
                ]
                updated = learner.update(
                    targets, seed=args.study_seed*1000003+index,
                )
                if learner.revision != index+1:
                    raise RuntimeError("H6 macro revision drift")
                wins_so_far += sum(ep.won for ep in completed)
                row = {
                    "round": index+1,
                    "behavior_player_sha256": file_sha256(behavior_path),
                    "frozen_combat_sha256": learner.frozen_sha256,
                    "source_digest": source_digest(seeds),
                    "defeat_progress_coefficient": alpha,
                    "train": summarize(episodes),
                    "rollout_seconds": elapsed,
                    **updated,
                }
                if (index+1) % args.eval_every == 0 or index+1 == args.rounds:
                    current = learner.model()
                    current_path = models_dir/f"player-r{index+1:04d}.json"
                    atomic_write(
                        current_path,
                        json.dumps(current.to_dict(), sort_keys=True)+"\n",
                    )
                    eval_seeds = run_sources(
                        "selection", args.eval_episodes,
                        study_seed=args.study_seed,
                    )
                    started = time.perf_counter()
                    evaluation = collect_parallel(
                        pool, current, eval_seeds,
                        base_seed=args.actor_seed,
                        max_decisions=args.max_decisions,
                        policy_id="prototype-fair-v0",
                        environment=NATIVE_OVERGROWTH,
                        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                        temperature=1.0, ascension=10,
                        combat_greedy=True, strategy_greedy=True,
                    )
                    row["selection"] = summarize(evaluation)
                    row["selection"]["source_digest"] = source_digest(eval_seeds)
                    row["selection"]["seconds"] = time.perf_counter()-started
                rows.append(row)
                _atomic_torch(checkpoint, {
                    "schema": SCHEMA,
                    "config": config,
                    "fingerprint": fingerprint,
                    "learner": learner.dump(),
                    "rows": rows,
                    "torch_rng_state": torch.get_rng_state(),
                    "native_fidelity": "INDEPENDENT_NOT_TRAINING_GATE",
                }, torch)
                atomic_write(
                    args.output_root/"progress.json",
                    json.dumps(
                        {"config": config, "rows": rows},
                        indent=2, sort_keys=True,
                    )+"\n",
                )
                status = row["train"]
                evaluation = row.get("selection")
                print(
                    f"[h6] round={index+1}/{args.rounds} "
                    f"completed={status['completed']}/{status['runs']} "
                    f"wins={status['wins']} "
                    f"strategy_actions={status['strategy_decisions']} "
                    f"combat_actions={status['combat_decisions']} "
                    f"macro_ppo_steps={updated['ppo_steps']} "
                    + (
                        f"selection={evaluation['wins']}/{evaluation['completed']} "
                        f"censored={evaluation['censored']}"
                        if evaluation else ""
                    ),
                    flush=True,
                )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
