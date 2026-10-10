#!/usr/bin/env python3
"""Stage, log and resume a large solo Overgrowth PPO run on pinned emulator."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean

from sts2_ai.emulator.episode_goal import (
    EPISODE_GOALS, NATIVE_ACT1_BOSS_GOAL, PROTOTYPE_THREE_ACT_GOAL,
)

PINNED_EMULATOR = "6328a62989014f080abed06684e4b5f7ea1f0af6"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True,
    ).strip()


def _check_revision(root: Path) -> None:
    pin = _git(root, "rev-parse", "HEAD:emulator")
    actual = _git(root / "emulator", "rev-parse", "HEAD")
    if pin != PINNED_EMULATOR or actual != pin:
        raise SystemExit(
            f"Refusing training on wrong emulator: branch pin={pin}, "
            f"checkout={actual}, expected={PINNED_EMULATOR}"
        )


def summarize(report: dict) -> dict[str, int | float | None]:
    rows = report.get("heldout", [])
    completed = [r for r in rows if not r.get("censored")]
    bosses = [r["boss"] for r in completed if r.get("boss") is not None]
    return {
        "evaluated": len(completed),
        "act1_clears": sum(
            r.get("act1_cleared") is True or (
                "act1_cleared" not in r
                and r.get("act") is not None and r["act"] >= 2
            ) for r in completed
        ),
        "mean_frontier": fmean(r["progress"] for r in completed)
        if completed else None,
        "boss_entries": len(bosses),
        "mean_boss_fraction_removed": fmean(
            (b["initial_hp"] - b["remaining_hp"]) / b["initial_hp"]
            for b in bosses if b["initial_hp"] > 0
        ) if bosses else None,
        "training_wins": sum(r["wins"] for r in report["rounds"]),
        "optimizer_steps": sum(r["optimization_steps"] for r in report["rounds"]),
        "rounds": len(report["rounds"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds", type=int, default=320)
    p.add_argument("--stage-size", type=int, default=40)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--eval-seeds", type=int, default=128)
    p.add_argument("--win-anneal-threshold", type=int, default=16)
    p.add_argument(
        "--episode-goal", choices=EPISODE_GOALS, default=NATIVE_ACT1_BOSS_GOAL,
    )
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--temperature-start", type=float, default=0.85)
    p.add_argument("--temperature-end", type=float, default=0.5)
    p.add_argument("--temperature-decay-rounds", type=int, default=200)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--ppo-batch-size", type=int, default=128)
    p.add_argument("--ppo-sample-limit", type=int, default=4096)
    p.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    p.add_argument("--ppo-gamma", type=float, default=0.995)
    p.add_argument("--ppo-gae-lambda", type=float, default=0.98)
    p.add_argument("--warm-start", type=Path, required=True)
    p.add_argument(
        "--tactical-state-encoding", choices=("public_resources", "relational_damage"),
        default="public_resources",
        help="v7 public-resource combat features, or v6 control with same emulator",
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/ppo-public-resources-v7"))
    p.add_argument("--save-combat-samples", action="store_true")
    p.add_argument("--train-seed-prefix", default="ppo-public-resources-v7-train")
    p.add_argument(
        "--exclude-train-seed", action="append", default=[],
        help="Explicitly omit an unsupported training episode seed (repeatable; logged)",
    )
    p.add_argument("--eval-seed-prefix", default="ppo-public-resources-v7-eval")
    args = p.parse_args()

    if args.win_anneal_threshold < 1:
        p.error("--win-anneal-threshold must be positive")
    if min(args.rounds, args.stage_size, args.episodes, args.workers) <= 0:
        p.error("rounds/stage/episodes/workers must be positive")
    if args.rounds % args.stage_size:
        p.error("--rounds must be a multiple of --stage-size")
    if not args.warm_start.is_file():
        p.error(f"Warm-start model does not exist: {args.warm_start}")
    root = Path(__file__).resolve().parents[1]
    _check_revision(root)
    output = args.output_dir.resolve()
    (output / "reports").mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    history: list[dict] = []
    for target_round in range(args.stage_size, args.rounds + 1, args.stage_size):
        report = output / "reports" / f"stage-{target_round:04d}.json"
        model = output / "models" / f"stage-{target_round:04d}.json"
        log_file = output / "logs" / f"stage-{target_round:04d}.log"
        if report.exists():
            existing = json.loads(report.read_text())
            if (
                existing["emulator_revision"] != PINNED_EMULATOR
                or existing.get("episode_goal_version", PROTOTYPE_THREE_ACT_GOAL)
                    != args.episode_goal
                or existing.get("win_anneal_threshold", 16)
                    != args.win_anneal_threshold
                or len(existing["rounds"]) != target_round
                or existing["optimizer_method"] != "ppo"
            ):
                raise SystemExit(f"Existing stage report mismatch: {report}")
            summary = summarize(existing)
            history.append({"stage": target_round, **summary})
            print(f"[large] stage {target_round} already completed: {summary}", flush=True)
            continue

        cmd = [
            sys.executable, "-u", "tools/train_phase_split.py",
            "--rounds", str(target_round),
            "--episodes", str(args.episodes),
            "--workers", str(args.workers),
            "--dimension", str(args.dimension),
            "--hidden", str(args.hidden),
            "--learning-rate", str(args.learning_rate),
            "--optimizer-method", "ppo",
            "--ppo-epochs", str(args.ppo_epochs),
            "--ppo-batch-size", str(args.ppo_batch_size),
            "--ppo-sample-limit", str(args.ppo_sample_limit),
            "--ppo-clip-epsilon", str(args.ppo_clip_epsilon),
            "--ppo-gamma", str(args.ppo_gamma),
            "--ppo-gae-lambda", str(args.ppo_gae_lambda),
            "--tactical-state-encoding", args.tactical_state_encoding,
            "--combat-objective", "hp_preservation",
            "--combat-advantage-baseline", "critic",
            "--hp-monotonic-weight", "0",
            "--loss-normalization", "phase_mean",
            "--temperature-start", str(args.temperature_start),
            "--temperature-end", str(args.temperature_end),
            "--temperature-decay-rounds", str(args.temperature_decay_rounds),
            "--train-seed-prefix", args.train_seed_prefix,
            "--episode-goal", args.episode_goal,
            "--win-anneal-threshold", str(args.win_anneal_threshold),
            "--eval-seed-prefix", args.eval_seed_prefix,
            "--eval-seeds", str(args.eval_seeds),
            "--seed", str(args.seed),
            "--warm-start", str(args.warm_start.resolve()),
            "--checkpoint", str(checkpoint),
            "--output", str(model),
            "--report", str(report),
        ]
        if checkpoint.exists():
            cmd.append("--resume")
        else:
            cmd.append("--build")
        for excluded_seed in sorted(set(args.exclude_train_seed)):
            cmd.extend(["--exclude-train-seed", excluded_seed])
        if args.save_combat_samples:
            cmd.extend(["--combat-samples-dir", str(output / "combat-samples")])
        print(
            f"[large] starting/recovering through round {target_round}/"
            f"{args.rounds}, full progress: {log_file}", flush=True,
        )
        with log_file.open("w", encoding="utf-8") as log:
            with subprocess.Popen(
                cmd, cwd=root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = proc.wait()
        if returncode:
            raise SystemExit(
                f"Training stage {target_round} failed with exit code {returncode}; "
                f"checkpoint is reusable. Inspect {log_file}"
            )
        record = json.loads(report.read_text())
        if (record["emulator_revision"] != PINNED_EMULATOR or
                record.get("episode_goal_version", PROTOTYPE_THREE_ACT_GOAL)
                != args.episode_goal or
                record.get("win_anneal_threshold", 16)
                != args.win_anneal_threshold):
            raise SystemExit("Unexpected emulator revision in completed training report")
        summary = summarize(record)
        history.append({"stage": target_round, **summary})
        print(f"[large] stage {target_round} completed: {summary}", flush=True)
        (output / "learning-curve.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    print(f"[large] training complete; learning curve: {output / 'learning-curve.json'}")


if __name__ == "__main__":
    main()
