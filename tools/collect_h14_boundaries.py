#!/usr/bin/env python3
"""Collect H14 public combat-boundary contract samples (heuristic integration smoke).

For H9/H14 training, use a pinned combat policy in the same observer interface.
This script is intentionally a source-fair integration probe, not model evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.training.h14_boundaries import (
    H14CombatBoundaryTrace,
    TRANSITION_SCHEMA,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--ascension", type=int, default=10)
    parser.add_argument("--max-decisions", type=int, default=2048)
    parser.add_argument("--seed-prefix", default="h14-boundary-smoke")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.max_decisions < 1:
        parser.error("--runs and --max-decisions must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with JsonlEmulatorBackend(build=args.build) as backend:
        with args.output.open("w", encoding="utf-8") as out:
            for i in range(args.runs):
                seed = f"{args.seed_prefix}-{i}"
                trace = H14CombatBoundaryTrace(
                    source_run_seed=seed, combat_policy_id="heuristic-agent-v1",
                )
                summary = play_run(
                    backend, HeuristicAgent(), seed=seed,
                    policy=InformationPolicy(FAIR_POLICY_ID),
                    ascension=args.ascension,
                    environment=NATIVE_OVERGROWTH,
                    max_decisions=args.max_decisions,
                    combat_boundary_observer=trace,
                )
                combats = trace.finish(censored=summary.censored)
                report: dict[str, Any] = {
                    "schema": "sts2-h14-public-combat-run-v1",
                    "transition_schema": TRANSITION_SCHEMA,
                    "provenance": {
                        "source_run_seed": seed,
                        "emulator_revision": backend.emulator_revision,
                        "observation_policy": FAIR_POLICY_ID,
                        "combat_policy_id": "heuristic-agent-v1",
                        "ascension": args.ascension,
                        "environment": NATIVE_OVERGROWTH,
                    },
                    "outcome": summary.outcome,
                    "censored": summary.censored,
                    "decisions": summary.decisions,
                    "act1_cleared": summary.act1_cleared,
                    "combats": combats,
                }
                out.write(json.dumps(report, sort_keys=True) + "\n")
                print(json.dumps({
                    "seed": seed, "outcome": summary.outcome,
                    "censored": summary.censored, "combats": len(combats),
                }), flush=True)


if __name__ == "__main__":
    main()
