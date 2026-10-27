#!/usr/bin/env python3
"""Evaluate a saved phase-split model on fresh fair solo Overgrowth seeds."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import EPISODE_GOALS, NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=256)
    parser.add_argument("--seed-prefix", default="ppo-pr51-final-unseen-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=4096)
    parser.add_argument(
        "--episode-goal", choices=EPISODE_GOALS, default=NATIVE_ACT1_BOSS_GOAL,
        help="Explicit native Act-1 certified boss clear (default) or historical full-run mode",
    )
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or not args.seed_prefix:
        parser.error("Expected positive runs and a nonempty seed prefix")
    agent = PhaseSplitGreedyAgent.load(args.model)
    with JsonlEmulatorBackend(build=args.build) as backend:
        rows = []
        for index in range(args.runs):
            seed = f"{args.seed_prefix}-{index}"
            result = play_run(
                backend, agent, seed=seed,
                policy=InformationPolicy(FAIR_POLICY_ID),
                max_decisions=args.max_decisions, environment=NATIVE_OVERGROWTH,
                episode_goal_version=args.episode_goal,
            )
            rows.append(result)
            print(
                f"[final-eval] {index + 1}/{args.runs} "
                f"outcome={result.outcome} act={result.terminal_act} "
                f"floor={result.terminal_floor}", file=sys.stderr, flush=True,
            )
        completed = [r for r in rows if not r.censored]
        bosses = [r.boss_progress for r in completed if r.boss_progress is not None]
        summary = {
            "completed": len(completed),
            "censored": len(rows) - len(completed),
            "act1_clears": sum(r.act1_cleared is True for r in completed),
            "mean_frontier_progress": fmean(
                r.frontier_progress for r in completed
            ) if completed else None,
            "boss_entries": len(bosses),
            "mean_boss_damage_fraction": fmean(
                b.damage_fraction for b in bosses
            ) if bosses else None,
        }
        report = {
            "schema": "sts2-phase-split-fair-evaluation-v1",
            "model": str(args.model),
            "emulator_revision": backend.emulator_revision,
            "environment": NATIVE_OVERGROWTH,
            "episode_goal_version": args.episode_goal,
            "seed_prefix": args.seed_prefix,
            "summary": summary,
            "runs": [asdict(r) for r in rows],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[final-eval] {summary}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
