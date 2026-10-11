"""Launch a controlled two-arm v15p importance-priority ablation.

Both arms use the *same target distribution* (beta), frozen base, vocabulary,
source corpus, seed, temperature, architecture, and number of combat episodes.
Only proposal mixing lambda differs: 0 vs 0.7. Separate checkpoints are
mandatory. Results are NOT considered final until paired held-out evaluation.

Example:
  python -u tools/run_importance_ablation_v15p.py \
    --base-model results/ppo-v8-batched-40/models/stage-0120.json \
    --vocabulary results/ppo-v15j-relational-cpu64/vocabulary.json \
    --snapshot-corpus results/v15o-snapshots/partition/train.jsonl \
    --output-root results/v15p-ablation --rounds 40 --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def manifest_for(args: argparse.Namespace) -> dict:
    paths = {
        "base_model": args.base_model,
        "vocabulary": args.vocabulary,
        "snapshot_corpus": args.snapshot_corpus,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    return {
        "schema": "sts2-v15p-importance-ablation-v1",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "common_settings": {
            "rounds": args.rounds, "episodes": args.episodes,
            "workers": args.workers, "seed": args.seed, "width": args.width,
            "target_balance": args.target_balance,
            "ppo_epochs": args.ppo_epochs, "ppo_batch_size": args.ppo_batch_size,
            "ppo_sample_limit": args.ppo_sample_limit,
            "max_decisions": args.max_decisions,
        },
        "arms": {"target": 0.0, "priority": 0.7},
    }


def command_for(args: argparse.Namespace, *, mix: float, arm: str) -> list[str]:
    command = [
        sys.executable, "-u", "tools/train_pooled_objects_v15.py",
        "--base-model", str(args.base_model),
        "--vocabulary", str(args.vocabulary),
        "--snapshot-corpus", str(args.snapshot_corpus),
        "--architecture", "relational",
        "--width", str(args.width),
        "--snapshot-sampling", "importance",
        "--snapshot-target-balance", str(args.target_balance),
        "--snapshot-priority-mix", str(mix),
        "--output-dir", str(args.output_root / arm),
        "--rounds", str(args.rounds),
        "--episodes", str(args.episodes),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--max-decisions", str(args.max_decisions),
        "--ppo-batch-size", str(args.ppo_batch_size),
        "--ppo-epochs", str(args.ppo_epochs),
        "--ppo-sample-limit", str(args.ppo_sample_limit),
        "--snapshot-every", str(args.snapshot_every),
        "--diagnose-ppo",
    ]
    if args.resume and (args.output_root / arm / "checkpoint.pt").is_file():
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--snapshot-corpus", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--target-balance", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--ppo-batch-size", type=int, default=64)
    parser.add_argument("--ppo-sample-limit", type=int, default=4096)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(
        args.rounds, args.episodes, args.workers, args.width,
        args.ppo_epochs, args.ppo_batch_size, args.ppo_sample_limit,
        args.max_decisions, args.snapshot_every,
    ) <= 0:
        parser.error("All simulation/optimization sizes must be positive")
    if not math.isfinite(args.target_balance) or not 0 <= args.target_balance <= 1:
        parser.error("Target balance must lie in [0,1]")
    manifest = manifest_for(args)
    path = args.output_root / "ablation-manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Ablation manifest mismatch; refusing to reuse experiment")
        if not args.resume and not args.plan_only and any(
            (args.output_root / arm / "checkpoint.pt").exists()
            or (args.output_root / f"{arm}.log").exists()
            for arm in manifest["arms"]
        ):
            raise FileExistsError(
                "Existing ablation requires --resume; use new --output-root"
            )
    elif args.resume:
        raise FileNotFoundError("Cannot resume without original ablation manifest")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    for arm, mix in manifest["arms"].items():
        cmd = command_for(args, mix=mix, arm=arm)
        print(f"[v15p ablation] {arm} lambda={mix}: {' '.join(cmd)}", flush=True)
        if args.plan_only:
            continue
        log_path = args.output_root / f"{arm}.log"
        with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            ) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{arm}] {line}", end="", flush=True)
                code = process.wait()
                if code:
                    raise RuntimeError(f"Ablation arm {arm} exited with {code}")
    print(f"[v15p ablation] complete: {path}", flush=True)


if __name__ == "__main__":
    main()
