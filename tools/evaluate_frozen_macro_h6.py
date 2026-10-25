#!/usr/bin/env python3
"""H6 paired macro-policy evaluation with an identical frozen combat backbone.

Only completed native Act-1 terminal outcomes enter paired victory analysis.
Truncated runs are censored, *never silently relabeled as defeats*.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

from train_frozen_macro_h6 import summarize

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.training.frozen_macro import combat_digest
from sts2_ai.training.h6_sources import H6_SOURCE_SCHEMA, run_sources, source_digest
from sts2_ai.training.parallel_rollouts import collect_parallel, open_worker_pool
SCHEMA = "sts2-h6-frozen-macro-confirmation-v1"


def paired_victories(a: list[dict], b: list[dict]) -> dict:
    if [x["source_seed"] for x in a] != [x["source_seed"] for x in b]:
        raise ValueError("Paired H6 runs have mismatched seeds")
    paired = [
        (int(y["victory"]) - int(x["victory"]))
        for x, y in zip(a, b, strict=True)
        if x["completed"] and y["completed"]
    ]
    if not paired:
        return {
            "paired_completed_runs": 0,
            "victory_delta": None,
            "bootstrap_ci95": None,
        }
    rng = random.Random(60606)
    n = len(paired)
    boots = sorted(
        statistics.mean(paired[rng.randrange(n)] for _ in range(n))
        for _ in range(2000)
    )
    return {
        "paired_completed_runs": n,
        "victory_delta": statistics.mean(paired),
        "bootstrap_ci95": [boots[49], boots[1949]],
        "different_completion_statuses": sum(
            x["completed"] != y["completed"]
            for x, y in zip(a, b, strict=True)
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", action="append", required=True,
                   help="Unique label=portable-phase-split-json")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--emulator-root", type=Path, default=Path("emulator"))
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("selection", "confirmation"),
                   default="confirmation")
    p.add_argument("--eval-episodes", type=int, default=120)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--study-seed", type=int, default=43)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    players: dict[str, Path] = {}
    for raw in args.player:
        label, separator, path = raw.partition("=")
        if not separator or not label or label in players:
            p.error("Require unique --player label=path arguments")
        players[label] = Path(path)
    if (min(args.eval_episodes, args.max_decisions) <= 0
            or args.workers < 2 or args.study_seed < 0):
        p.error("Invalid H6 evaluation budgets or worker count")
    if any(not path.is_file() for path in players.values()):
        p.error("Missing H6 evaluation model file")
    models = {
        label: PhaseSplitNeuralModel.load(path)
        for label, path in players.items()
    }
    digests = {label: combat_digest(model) for label, model in models.items()}
    if len(set(digests.values())) != 1:
        p.error("H6 evaluation requires byte-identical frozen combat weights")
    seeds = run_sources(
        args.split, args.eval_episodes, study_seed=args.study_seed,
    )
    config = {
        "schema": SCHEMA,
        "source_schema": H6_SOURCE_SCHEMA,
        "split": args.split,
        "source_digest": source_digest(seeds),
        "models": {
            label: file_sha256(path)
            for label, path in sorted(players.items())
        },
        "frozen_combat_digest": next(iter(digests.values())),
        "emulator_revision": args.emulator_revision,
        "environment": NATIVE_OVERGROWTH,
        "goal": NATIVE_ACT1_BOSS_GOAL,
        "ascension": 10,
        "policy": "greedy-strategy-and-greedy-frozen-combat",
        "runs_per_model": args.eval_episodes,
        "workers": args.workers,
        "max_decisions": args.max_decisions,
        "actor_seed": args.actor_seed,
        "study_seed": args.study_seed,
    }
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if args.resume:
        if not args.output.is_file():
            p.error("H6 resume requires a partial evaluation report")
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if saved["config"] != config:
            p.error("H6 evaluation inputs changed during resume")
        worlds = saved["worlds"]
        summaries = saved["summaries"]
    else:
        if args.output.exists():
            p.error("H6 report already exists; use --resume")
        worlds = {}
        summaries = {}

    def save(complete: bool = False) -> None:
        comparisons: dict[str, dict] = {}
        baseline = sorted(players)[0]
        if baseline in worlds:
            for label in sorted(worlds):
                if label != baseline:
                    comparisons[f"{label}-minus-{baseline}"] = paired_victories(
                        worlds[baseline], worlds[label],
                    )
        atomic_write(args.output, json.dumps({
            "config": config,
            "worlds": worlds,
            "summaries": summaries,
            "comparisons": comparisons,
            "complete": complete,
        }, indent=2, sort_keys=True)+"\n")

    repo_root = args.repo_root.resolve()
    emulator_root = args.emulator_root.resolve()
    backend = JsonlEmulatorBackend(
        repo_root=repo_root, emulator_root=emulator_root,
        expected_emulator_revision=args.emulator_revision, build=False,
    )
    try:
        with ExitStack() as stack:
            pool = stack.enter_context(open_worker_pool(backend, args.workers))
            for label in sorted(players):
                if label in worlds:
                    if [r["source_seed"] for r in worlds[label]] != list(seeds):
                        p.error("H6 resume source worlds changed")
                    print(f"[h6-eval] reused {label}", flush=True)
                    continue
                started = time.perf_counter()
                episodes = collect_parallel(
                    pool, models[label], seeds,
                    base_seed=args.actor_seed,
                    max_decisions=args.max_decisions,
                    policy_id="prototype-fair-v0",
                    environment=NATIVE_OVERGROWTH,
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                    temperature=1.0, ascension=10,
                    combat_greedy=True, strategy_greedy=True,
                )
                worlds[label] = [{
                    "source_seed": ep.seed,
                    "completed": ep.completed,
                    "outcome": ep.outcome,
                    "victory": ep.won,
                    "act": ep.act,
                    "floor": ep.floor,
                    "hp_fraction": ep.hp_ratio,
                    "strategy_decisions": sum(
                        d.phase == "strategy" for d in ep.decisions
                    ),
                    "combat_decisions": sum(
                        d.phase == "combat" for d in ep.decisions
                    ),
                } for ep in episodes]
                summaries[label] = summarize(episodes)
                summaries[label]["seconds"] = time.perf_counter()-started
                save()
                summary = summaries[label]
                print(
                    f"[h6-eval] {label} complete={summary['completed']}/"
                    f"{summary['runs']} wins={summary['wins']} "
                    f"censored={summary['censored']}",
                    flush=True,
                )
            save(complete=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
