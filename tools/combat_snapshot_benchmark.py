#!/usr/bin/env python3
"""Collect reachable native Act-1 combat entries or compare two policies on them."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.evaluation.combat_snapshots import (
    SCHEMA,
    SnapshotCollector,
    build_report,
    evaluate_recipes,
    load_recipes,
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def collect(args: argparse.Namespace) -> None:
    if args.runs <= 0 or args.max_run_decisions <= 0:
        raise ValueError("Runs and max-run-decisions must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Corpus exists: {args.output}; use --overwrite explicitly")

    agent = PhaseSplitGreedyAgent.load(args.collector_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".incomplete")
    # Always overwrite stale incomplete collection; a finished corpus is atomic.
    total = 0
    try:
        with JsonlEmulatorBackend(build=args.build) as backend, temp.open("w") as output:
            for run in range(args.runs):
                seed = f"{args.seed_prefix}-{run}"
                recorder = SnapshotCollector(
                    backend, seed=seed, source_policy=agent.policy_id,
                    ascension=args.ascension,
                )
                result = play_run(
                    backend, agent, seed=seed, policy=InformationPolicy(FAIR_POLICY_ID),
                    ascension=args.ascension, max_decisions=args.max_run_decisions,
                    environment=NATIVE_OVERGROWTH,
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                    decision_observer=recorder.on_decision,
                )
                for recipe in recorder.recipes:
                    output.write(json.dumps(asdict(recipe), sort_keys=True) + "\n")
                output.flush()
                total += len(recorder.recipes)
                log(f"[snapshots] run={run+1}/{args.runs} seed={seed} "
                    f"outcome={result.outcome} captures={len(recorder.recipes)} total={total}")
        temp.replace(args.output)
    except BaseException:
        log(f"[snapshots] interrupted; untrusted partial corpus at {temp}")
        raise
    log(f"[snapshots] collected {total} authentic entries schema={SCHEMA} "
        f"output={args.output}")


def evaluate(args: argparse.Namespace) -> None:
    recipes = load_recipes(args.corpus)
    if not recipes:
        raise ValueError("Empty combat snapshot corpus")
    baseline = PhaseSplitGreedyAgent.load(args.baseline_model)
    candidate = PhaseSplitGreedyAgent.load(args.candidate_model)
    with JsonlEmulatorBackend(build=args.build) as backend:
        log(f"[snapshots] evaluating {len(recipes)} entries on pinned "
            f"emulator {backend.emulator_revision[:12]}")
        results = evaluate_recipes(
            backend, recipes, baseline, candidate,
            max_combat_decisions=args.max_combat_decisions,
        )
        report = build_report(
            results, emulator_revision=backend.emulator_revision,
            corpus=str(args.corpus), baseline_model=str(args.baseline_model),
            candidate_model=str(args.candidate_model),
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"[snapshots] completed {len(results)} pairs; "
        f"overall={report['summary']} report={args.report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    a = cmd.add_parser("collect", help="Record authentic reachable combat recipes")
    a.add_argument("--collector-model", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--seed-prefix", default="combat-snapshot-v14-collection")
    a.add_argument("--runs", type=int, default=100)
    a.add_argument("--ascension", type=int, default=0)
    a.add_argument("--max-run-decisions", type=int, default=4096)
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--build", action="store_true")
    b = cmd.add_parser("evaluate", help="Replay paired combat from exact forked entries")
    b.add_argument("--corpus", type=Path, required=True)
    b.add_argument("--baseline-model", type=Path, required=True)
    b.add_argument("--candidate-model", type=Path, required=True)
    b.add_argument("--report", type=Path, required=True)
    b.add_argument("--max-combat-decisions", type=int, default=256)
    b.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
