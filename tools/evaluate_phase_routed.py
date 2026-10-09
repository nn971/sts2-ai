#!/usr/bin/env python3
"""Evaluate an explicit combat-vs-strategy policy pair on common public seeds."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sts2_ai.agents import NeuralGreedyAgent
from sts2_ai.agents.phase_routed_neural import PhaseRoutedNeuralAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, require_environment
from sts2_ai.evaluation.run import play_run
from sts2_ai.evaluation.selfplay_metrics import (
    compare_completed_pairs, summarize_completed_runs,
)
from sts2_ai.models.neural import NeuralPolicyValueModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-model", type=Path, required=True)
    parser.add_argument("--combat-model", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--seed-prefix", default="phase-routed-heldout")
    parser.add_argument("--max-decisions", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    strategy = NeuralPolicyValueModel.load(args.strategy_model)
    combat = NeuralPolicyValueModel.load(args.combat_model)
    router = PhaseRoutedNeuralAgent(strategy=strategy, tactical=combat)
    baseline = NeuralGreedyAgent(
        strategy, content_hash=strategy.model_id
    )
    with JsonlEmulatorBackend(build=args.build) as backend:
        require_environment(backend, NATIVE_OVERGROWTH)
        results = {}
        for name, agent in (("strategy_only", baseline), ("phase_routed", router)):
            rows = []
            for i in range(args.runs):
                seed = f"{args.seed_prefix}-{i}"
                result = play_run(
                    backend, agent, seed=seed,
                    policy=InformationPolicy(FAIR_POLICY_ID),
                    max_decisions=args.max_decisions,
                    environment=NATIVE_OVERGROWTH,
                )
                rows.append(result)
                print(
                    f"[{name}] {i + 1}/{args.runs} {seed} "
                    f"{result.outcome} act={result.terminal_act} "
                    f"floor={result.terminal_floor}",
                    flush=True, file=sys.stderr,
                )
            results[name] = rows
        payload = {
            "schema": "sts2-phase-routed-comparison-v1",
            "emulator_revision": backend.emulator_revision,
            "strategy_model_id": strategy.model_id,
            "combat_model_id": combat.model_id,
            "runs": args.runs,
            "summary": {
                name: summarize_completed_runs(rows)
                for name, rows in results.items()
            },
            "paired": compare_completed_pairs(
                results["strategy_only"], results["phase_routed"]
            ),
            "trajectories": {
                name: [asdict(row) for row in rows]
                for name, rows in results.items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[eval] comparison saved: {args.output}", flush=True, file=sys.stderr)


if __name__ == "__main__":
    main()
