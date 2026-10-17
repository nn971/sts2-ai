#!/usr/bin/env python3
"""Plan or execute paired multi-seed Nature-versus-ordinary v15 P0 PPO.

Run locally with --execute after supplying a genuinely PRETRAINED v15 portable
checkpoint and source-route candidate partitions. The source route must
actually replay in C#; a random candidate seed alone is not certification.
Default is a non-destructive plan. Heavy runs use 15 emulator processes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.research_p0 import (
    assert_disjoint_partitions,
    read_partition,
)

TRAIN_SCHEMA = "sts2-ai-p0-real-combat-v15-ppo-training-v1"
PLAN_SCHEMA = "sts2-ai-p0-replicated-train-plan-v1"


def parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError("Training seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require distinct independent training restart seeds")
    return seeds


def make_commands(
    *, seeds: tuple[int, ...], rounds: int, episodes_per_round: int,
    workers: int, initial_model: Path, train_corpus: Path,
    evaluation_corpus: Path, emulator_dll: Path,
    emulator_bindings: Path, emulator_revision: str, root: Path,
    temperature: float, learning_rate: float, epochs: int,
    batch_size: int, sample_limit: int,
) -> list[dict]:
    """Single source of the symmetric per-seed training/evaluation recipe."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate or empty training restart seeds")
    if min(rounds, episodes_per_round, workers, epochs, batch_size, sample_limit) < 1:
        raise ValueError("Positive training and compute budgets required")
    if not (0 < temperature and 0 < learning_rate):
        raise ValueError("Invalid training temperature/learning rate")
    scripts = Path(__file__).resolve().parent
    result: list[dict] = []
    for seed in seeds:
        directory = root / f"seed-{seed}"
        for arm in ("nature", "ordinary"):
            cmd = [
                sys.executable, "-u", str(scripts / "train_nature_two_site_player_ppo.py"),
                "--initial-model", str(initial_model),
                "--source-fixtures", str(train_corpus),
                "--emulator-dll", str(emulator_dll),
                "--emulator-bindings", str(emulator_bindings),
                "--emulator-revision", emulator_revision,
                "--rounds", str(rounds),
                "--episodes-per-round", str(episodes_per_round),
                "--workers", str(workers), "--seed", str(seed),
                "--temperature", str(temperature),
                "--learning-rate", str(learning_rate),
                "--epochs", str(epochs), "--batch-size", str(batch_size),
                "--sample-limit", str(sample_limit),
                "--training-arm", arm, "--output-root", str(directory / arm),
            ]
            if arm == "nature":
                cmd.append("--nature-update")
            result.append({
                "training_seed": seed, "phase": "train", "arm": arm,
                "output": str(directory / arm), "argv": cmd,
            })
        result.append({
            "training_seed": seed, "phase": "holdout", "arm": "ordinary-primary",
            "output": str(directory / "holdout"),
            "argv": [
                sys.executable, "-u", str(scripts / "evaluate_nature_p0_player_ppo.py"),
                "--nature-trained", str(directory / "nature" / "final-model.json"),
                "--ordinary-trained", str(directory / "ordinary" / "final-model.json"),
                "--train-corpus", str(train_corpus),
                "--holdout-corpus", str(evaluation_corpus),
                "--emulator-dll", str(emulator_dll),
                "--emulator-bindings", str(emulator_bindings),
                "--emulator-revision", emulator_revision,
                "--workers", str(workers),
                "--actor-seed", str(900000 + seed),
                "--temperature", str(temperature),
                "--output-root", str(directory / "holdout"),
            ],
        })
    return result


def _run_job(job: dict, *, rounds: int, seed: int) -> None:
    directory = Path(job["output"])
    if job["phase"] == "train":
        history = directory / "history.json"
        model = directory / "final-model.json"
        if history.exists():
            state = json.loads(history.read_text(encoding="utf-8"))
            config = state.get("config", {})
            if (state.get("schema") != TRAIN_SCHEMA
                    or config.get("seed") != seed
                    or config.get("arm") != job["arm"]):
                raise ValueError("Existing training output has wrong restart/arm")
            if len(state.get("metrics", [])) == rounds and model.exists():
                print(f"[nature-p0-replicas] already complete {job['output']}", flush=True)
                return
        cmd = list(job["argv"])
        checkpoint = directory / "trainer-state.pt"
        if checkpoint.exists():
            cmd.append("--resume")
        elif directory.exists() and any(directory.iterdir()):
            raise ValueError("Partial training directory lacks resumable checkpoint")
    else:
        target = directory / "holdout-report.json"
        if target.is_file():
            raise ValueError(
                "Existing holdout output is immutable; verify it separately "
                "or choose a new study root"
            )
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Incomplete holdout output; choose a new study root")
        cmd = job["argv"]
    print(
        f"[nature-p0-replicas] seed={seed} phase={job['phase']} "
        f"arm={job['arm']}", flush=True,
    )
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial-model", type=Path, required=True)
    p.add_argument("--train-corpus", type=Path, required=True)
    p.add_argument("--evaluation-corpus", type=Path, required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--seeds", default="29,101,211,307,401")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--episodes-per-round", type=int, default=128)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample-limit", type=int, default=2048)
    p.add_argument("--execute", action="store_true", help="Launch heavy subprocess jobs")
    p.add_argument("--allow-untrained-smoke", action="store_true",
                   help="Only for engineering tests with synthetic checkpoint")
    args = p.parse_args()
    try:
        seeds = parse_seeds(args.seeds)
        train = read_partition(args.train_corpus, "train")
        evaluation = read_partition(args.evaluation_corpus, "evaluation")
        assert_disjoint_partitions({"train": train, "evaluation": evaluation})
        if len(train) < args.rounds * args.episodes_per_round:
            raise ValueError("Too few distinct candidate training source routes")
        if not args.emulator_dll.is_file():
            raise FileNotFoundError("Pinned emulator CLI DLL missing")
        bindings_file = args.emulator_bindings / "sts2_emulator" / "__init__.py"
        if not bindings_file.is_file():
            raise FileNotFoundError("Pinned emulator Python bindings missing")
        initial = PhaseSplitNeuralModel.load(args.initial_model)
        if initial.combat_value_objective != "hp_preservation":
            raise ValueError("Pretrained v15 requires hp_preservation value objective")
        if (not args.allow_untrained_smoke
                and any(word in initial.model_id.lower()
                        for word in ("test-only", "untrained"))):
            raise ValueError(
                "Refusing test-only/untrained checkpoint for research replicas"
            )
        jobs = make_commands(
            seeds=seeds, rounds=args.rounds, episodes_per_round=args.episodes_per_round,
            workers=args.workers, initial_model=args.initial_model.resolve(),
            train_corpus=args.train_corpus.resolve(),
            evaluation_corpus=args.evaluation_corpus.resolve(),
            emulator_dll=args.emulator_dll.resolve(),
            emulator_bindings=args.emulator_bindings.resolve(),
            emulator_revision=args.emulator_revision, root=args.study_root.resolve(),
            temperature=args.temperature, learning_rate=args.learning_rate,
            epochs=args.epochs, batch_size=args.batch_size,
            sample_limit=args.sample_limit,
        )
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    fingerprint_inputs = {
        "schema": PLAN_SCHEMA,
        "model_sha256": file_sha256(args.initial_model),
        "train_sha256": file_sha256(args.train_corpus),
        "evaluation_sha256": file_sha256(args.evaluation_corpus),
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_bindings_sha256": file_sha256(bindings_file),
        "emulator_revision": args.emulator_revision,
        "seeds": list(seeds), "rounds": args.rounds,
        "episodes_per_round": args.episodes_per_round,
        "workers": args.workers, "temperature": args.temperature,
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "batch_size": args.batch_size, "sample_limit": args.sample_limit,
        "allow_untrained_smoke": args.allow_untrained_smoke,
        "planned_study_jobs": jobs,
    }
    plan = {
        **fingerprint_inputs,
        "fingerprint": canonical_sha256(fingerprint_inputs),
        "source_status": "CANDIDATE_ROUTES_UNVERIFIED",
        "matched_compute": "PLANNED_SAME_PLAYER_SETTINGS_NOT_CERTIFIED_TOTAL_COMPUTE",
        "research_verdict": "NOT_ESTABLISHED",
    }
    print(
        f"[nature-p0-replicas] plan={plan['fingerprint'][:16]} "
        f"seeds={len(seeds)} train_jobs={2 * len(seeds)} "
        f"holdout_jobs={len(seeds)} "
        f"training_episodes_per_arm={args.rounds * args.episodes_per_round} "
        f"execute={args.execute}", flush=True,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
        return
    args.study_root.mkdir(parents=True, exist_ok=True)
    stored = args.study_root / "study-plan.json"
    if stored.exists():
        if json.loads(stored.read_text(encoding="utf-8")) != plan:
            p.error("Cannot resume study under changed immutable experiment plan")
    else:
        if any(args.study_root.iterdir()):
            p.error("Study root already contains files but has no matching study plan")
        atomic_write(stored, json.dumps(plan, sort_keys=True, indent=2) + "\n")
    for job in jobs:
        _run_job(job, rounds=args.rounds, seed=job["training_seed"])
    report = args.study_root / "replicated-analysis.json"
    if report.exists():
        p.error("Refusing to replace existing replicated analysis")
    subprocess.run([
        sys.executable, "-u",
        str(Path(__file__).resolve().parent / "analyze_nature_p0_replicas.py"),
        "--study-root", str(args.study_root),
        "--training-seeds", ",".join(str(seed) for seed in seeds),
        "--output", str(report),
    ], check=True)


if __name__ == "__main__":
    main()
