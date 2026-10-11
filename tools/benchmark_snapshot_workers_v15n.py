#!/usr/bin/env python3
"""Benchmark serial versus parallel authentic combat snapshot collection.

All worker modes use one frozen model, identical selected entries, and the
same independent per-episode actor seeds. Abort if exact public rollout
fingerprints differ. No PPO update is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import load_recipes
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.parallel_rollouts import open_worker_pool
from sts2_ai.training.snapshot_curriculum import (
    collect_snapshot_cohort,
    sample_snapshot_curriculum,
)


def digest_cohort(episodes: tuple) -> str:
    """Fingerprint complete public decisions and resolved combat boundaries."""
    records = []
    for episode in episodes:
        records.append({
            "seed": episode.seed,
            "outcome": episode.outcome,
            "completed": episode.completed,
            "decisions": [{
                "public": decision.observation.payload_json,
                "menu": [
                    (action.action_id, action.kind, action.payload_json)
                    for action in decision.legal_actions
                ],
                "choice": decision.chosen_index,
            } for decision in episode.decisions],
            "combats": [asdict(outcome) for outcome in episode.combat_outcomes],
        })
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--sampling", choices=("balanced", "natural"),
                        default="balanced")
    parser.add_argument("--seed", type=int, default=9917)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if (not args.corpus.is_file() or not args.model.is_file()
            or min(args.workers) <= 0 or len(set(args.workers)) != len(args.workers)):
        parser.error("Require existing inputs and unique positive worker counts")
    if args.report.exists():
        parser.error("Report exists; choose a fresh output path")
    recipes = load_recipes(args.corpus)
    selected = sample_snapshot_curriculum(
        recipes, count=args.episodes, seed=args.seed, mode=args.sampling,
    )
    model = PhaseSplitNeuralModel.load(args.model)
    timings: list[dict] = []
    expected: str | None = None
    with JsonlEmulatorBackend(build=args.build) as backend:
        for worker_count in args.workers:
            print(
                f"[snapshot-bench] workers={worker_count} entries={len(selected)} "
                f"emulator={backend.emulator_revision}", file=sys.stderr,
                flush=True,
            )
            started = time.perf_counter()
            manager = (
                open_worker_pool(backend, worker_count)
                if worker_count > 1 else nullcontext(None)
            )
            with manager as pool:
                episodes = collect_snapshot_cohort(
                    backend, model, selected,
                    round_index=args.round_index,
                    actor_seed=args.seed, temperature=args.temperature,
                    max_decisions=args.max_decisions,
                    pool=pool, workers=worker_count,
                )
            elapsed = time.perf_counter() - started
            digest = digest_cohort(episodes)
            if expected is None:
                expected = digest
            elif digest != expected:
                raise ValueError(
                    f"Parallel snapshot result mismatch for workers={worker_count}"
                )
            resolved = sum(ep.completed for ep in episodes)
            wins = sum(ep.won for ep in episodes)
            row = {
                "workers": worker_count, "seconds": elapsed,
                "episodes": len(episodes), "resolved": resolved, "combat_wins": wins,
                "decisions": sum(len(ep.decisions) for ep in episodes),
                "episode_digest": digest,
                "episodes_per_second": len(episodes) / elapsed,
            }
            timings.append(row)
            print(f"[snapshot-bench] {json.dumps(row, sort_keys=True)}",
                  file=sys.stderr, flush=True)
        report = {
            "schema": "sts2-v15n-parallel-snapshot-throughput-v1",
            "model_sha256": sha256(args.model),
            "corpus_sha256": sha256(args.corpus),
            "emulator_revision": backend.emulator_revision,
            "episodes": args.episodes,
            "sampling": args.sampling,
            "seed": args.seed,
            "round_index": args.round_index,
            "temperature": args.temperature,
            "max_decisions": args.max_decisions,
            "rows": timings,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[snapshot-bench] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
