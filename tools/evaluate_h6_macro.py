#!/usr/bin/env python3
"""Independent paired full-run H6 evaluation with an identical frozen combat head.

Each worker owns a private semantic emulator; seeds are shared across macro
models but follow different action-dependent stochastic trajectories. Censored
runs are not counted as losses. This is evaluative, never a training gate.
"""
from __future__ import annotations

import argparse
import atexit
import json
import random
import statistics
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, require_environment
from sts2_ai.evaluation import play_run
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.protocol import canonical_sha256

SCHEMA = "sts2-ai-h6-paired-greedy-macro-a10-v1"
_WORKER: JsonlEmulatorBackend | None = None
_AGENT: PhaseSplitGreedyAgent | None = None


def _init(repo_root: str, emulator_root: str, revision: str,
          model_json: dict[str, Any]) -> None:
    global _WORKER, _AGENT
    _WORKER = JsonlEmulatorBackend(
        repo_root=Path(repo_root), emulator_root=Path(emulator_root),
        expected_emulator_revision=revision, build=False,
    )
    require_environment(_WORKER, NATIVE_OVERGROWTH)
    _AGENT = PhaseSplitGreedyAgent(PhaseSplitNeuralModel.from_dict(model_json))
    atexit.register(_WORKER.close)


def _evaluate(seed: str, max_decisions: int) -> dict:
    if _WORKER is None or _AGENT is None:
        raise RuntimeError("Uninitialized H6 evaluation worker")
    result = play_run(
        _WORKER, _AGENT, seed=seed, ascension=10,
        policy=InformationPolicy(FAIR_POLICY_ID),
        environment=NATIVE_OVERGROWTH,
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        max_decisions=max_decisions,
    )
    return {
        "seed": result.seed, "outcome": result.outcome,
        "censored": result.censored,
        "act1_cleared": result.act1_cleared,
        "act": result.terminal_act, "floor": result.terminal_floor,
        "frontier_progress": result.frontier_progress,
        "decisions": result.decisions,
        "wall_seconds": result.wall_seconds,
        "final_hp": result.hp_trajectory[-1] if result.hp_trajectory else None,
    }


def _summarize(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot evaluate an empty run population")
    finished = [r for r in rows if not r["censored"]]
    wins = sum(r["act1_cleared"] is True for r in finished)
    return {
        "runs": len(rows), "completed": len(finished),
        "censored": len(rows) - len(finished),
        "act1_clears": wins,
        "completion_rate": len(finished) / len(rows),
        "win_rate_lower_bound": wins / len(rows),
        "win_rate_upper_bound": (wins + len(rows) - len(finished)) / len(rows),
        "completed_only_win_rate": (
            wins / len(finished) if finished else None
        ),
        "mean_completed_frontier_progress": (
            statistics.mean(r["frontier_progress"] for r in finished)
            if finished else None
        ),
    }


def _paired(baseline: Sequence[dict], candidate: Sequence[dict]) -> dict:
    if len(baseline) != len(candidate) or (
        [x["seed"] for x in baseline] != [x["seed"] for x in candidate]
    ):
        raise ValueError("Unmatched macro evaluation source worlds")
    pairs = [(a, b) for a, b in zip(baseline, candidate, strict=True)
             if not a["censored"] and not b["censored"]]
    source_rng = random.Random(61026)
    differences = [
        int(b["act1_cleared"] is True) - int(a["act1_cleared"] is True)
        for a, b in pairs
    ]
    bootstrap = sorted(
        statistics.mean(
            differences[source_rng.randrange(len(differences))]
            for _ in range(len(differences))
        )
        for _ in range(2000)
    ) if differences else []
    return {
        "completed_pair_win_delta_bootstrap_95pct": (
            [bootstrap[49], bootstrap[1949]] if bootstrap else None
        ),
        "pairs_jointly_completed": len(pairs),
        "pairs_with_censoring": len(baseline) - len(pairs),
        "completed_pair_win_delta": (
            statistics.mean(int(b["act1_cleared"] is True)
                            - int(a["act1_cleared"] is True) for a, b in pairs)
            if pairs else None
        ),
        "completed_pair_progress_delta": (
            statistics.mean(b["frontier_progress"] - a["frontier_progress"]
                            for a, b in pairs) if pairs else None
        ),
        "same_source_id_not_same_stochastic_path": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", action="append", required=True,
                   help="Unique label=exported-phase-split-model.json")
    p.add_argument("--split", choices=("selection", "confirmation"),
                   default="confirmation")
    p.add_argument("--runs", type=int, default=160)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--seed-prefix", default="h6-v1")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.runs, args.workers, args.max_decisions) <= 0 or not args.seed_prefix:
        p.error("Invalid evaluation sizes or seed prefix")

    paths: dict[str, Path] = {}
    for raw in args.player:
        label, sep, path = raw.partition("=")
        if not sep or not label or label in paths or not Path(path).is_file():
            p.error("Expected unique label=existing-model.json")
        paths[label] = Path(path)
    if len(paths) < 2:
        p.error("Comparison requires at least two macro checkpoints")
    models = {label: PhaseSplitNeuralModel.load(path)
              for label, path in sorted(paths.items())}
    combat = next(iter(models.values())).combat.to_dict()
    if any(m.combat.to_dict() != combat for m in models.values()):
        p.error("H6 comparison requires identical frozen combat model weights")

    seeds = tuple(f"{args.seed_prefix}-{args.split}-a10-{i:06d}"
                  for i in range(args.runs))
    with JsonlEmulatorBackend(build=args.build) as backend:
        require_environment(backend, NATIVE_OVERGROWTH)
        config = {
            "schema": SCHEMA, "ascension": 10,
            "goal": NATIVE_ACT1_BOSS_GOAL, "environment": NATIVE_OVERGROWTH,
            "player_models": {label: file_sha256(path)
                              for label, path in sorted(paths.items())},
            "combat_model_sha256": canonical_sha256(combat),
            "emulator_revision": backend.emulator_revision,
            "source_digest": canonical_sha256(seeds), "split": args.split,
            "runs": len(seeds), "workers": args.workers,
            "max_decisions": args.max_decisions,
        }
        if args.resume:
            if not args.output.is_file():
                p.error("--resume requires saved output")
            prior = json.loads(args.output.read_text(encoding="utf-8"))
            if prior.get("config") != config:
                p.error("Cannot resume H6 comparison with changed config")
            worlds: dict[str, list[dict]] = prior["worlds"]
        else:
            if args.output.exists():
                p.error("Report already exists; use --resume or new output")
            worlds = {}
        print("[h6-eval] "+json.dumps(config, sort_keys=True), flush=True)

        def persist(complete: bool = False) -> None:
            summaries = {label: _summarize(rows)
                         for label, rows in worlds.items()}
            anchor = sorted(models)[0]
            comparisons = {
                label + "-minus-" + anchor: _paired(worlds[anchor], rows)
                for label, rows in worlds.items()
                if label != anchor and anchor in worlds
            }
            atomic_write(args.output, json.dumps({
                "config": config, "worlds": worlds, "summaries": summaries,
                "paired": comparisons, "complete": complete,
            }, indent=2, sort_keys=True) + "\n")

        for label, model in models.items():
            if label in worlds:
                if tuple(r["seed"] for r in worlds[label]) != seeds:
                    p.error("Saved macro evaluation seed mismatch")
                print(f"[h6-eval] reused {label}", flush=True)
                continue
            results = []
            if args.workers == 1:
                agent = PhaseSplitGreedyAgent(model)
                for seed in seeds:
                    obj = play_run(
                        backend, agent, seed=seed, ascension=10,
                        policy=InformationPolicy(FAIR_POLICY_ID),
                        environment=NATIVE_OVERGROWTH,
                        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                        max_decisions=args.max_decisions,
                    )
                    results.append({
                        "seed": obj.seed, "outcome": obj.outcome,
                        "censored": obj.censored,
                        "act1_cleared": obj.act1_cleared,
                        "act": obj.terminal_act, "floor": obj.terminal_floor,
                        "frontier_progress": obj.frontier_progress,
                        "decisions": obj.decisions,
                        "wall_seconds": obj.wall_seconds,
                        "final_hp": obj.hp_trajectory[-1]
                        if obj.hp_trajectory else None,
                    })
                    if len(results) % max(1, len(seeds)//8) == 0:
                        print(f"[h6-eval] {label} {len(results)}/{len(seeds)}",
                              flush=True)
            else:
                with ProcessPoolExecutor(
                    max_workers=args.workers,
                    mp_context=get_context("spawn"),
                    initializer=_init,
                    initargs=(
                        str(backend._repo_root), str(backend._emulator_root),
                        backend.emulator_revision, model.to_dict(),
                    ),
                ) as pool:
                    for item in pool.map(
                        _evaluate, seeds,
                        [args.max_decisions] * len(seeds),
                    ):
                        results.append(item)
                        if len(results) % max(1, len(seeds)//8) == 0:
                            print(f"[h6-eval] {label} "
                                  f"{len(results)}/{len(seeds)}",
                                  flush=True)
            if tuple(r["seed"] for r in results) != seeds:
                raise RuntimeError("H6 evaluator returned unmatched sources")
            worlds[label] = results
            persist()
            print(f"[h6-eval] {label} {_summarize(results)}", flush=True)
        persist(complete=True)


if __name__ == "__main__":
    main()
