#!/usr/bin/env python3
"""Compare greedy and sampled neural gameplay, with public Act-1 boss traces.

Run after training has exported a portable model JSON. Does not resume or
modify the training checkpoint. All policies consume only player-visible
observations and legal-action menus; chance stays inside the emulator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import ENVIRONMENTS, NATIVE_OVERGROWTH
from sts2_ai.evaluation.neural_temperature import diagnose_neural_temperatures
from sts2_ai.models.neural import NeuralPolicyValueModel


def _parse_temperatures(raw: list[str]) -> tuple[float | None, ...]:
    temperatures: list[float | None] = []
    for item in raw:
        try:
            temperatures.append(None if item == "greedy" else float(item))
        except ValueError as exc:
            raise ValueError(f"Invalid temperature {item!r}; use positive T or greedy") from exc
    return tuple(temperatures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=64)
    parser.add_argument("--seed-prefix", default="selfplay-heldout")
    parser.add_argument("--temperatures", nargs="+", default=["1", "0.5", "0.25", "greedy"])
    parser.add_argument("--environment", choices=ENVIRONMENTS, default=NATIVE_OVERGROWTH)
    parser.add_argument("--max-decisions", type=int, default=4096)
    parser.add_argument("--max-boss-actions", type=int, default=48)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")
    try:
        values = _parse_temperatures(args.temperatures)
        with JsonlEmulatorBackend(build=args.build) as backend:
            result = diagnose_neural_temperatures(
                backend, NeuralPolicyValueModel.load(args.model),
                seeds=[f"{args.seed_prefix}-{index}" for index in range(args.runs)],
                temperatures=values,
                environment=args.environment,
                max_decisions=args.max_decisions,
                max_boss_actions=args.max_boss_actions,
            )
    except ValueError as exc:
        parser.error(str(exc))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(args.report),
        "schema": result["schema"],
        "model_id": result["model_id"],
        "emulator_revision": result["emulator_revision"],
        "summary": result["summary"],
        "paired_completed_only": result["paired_completed_only"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
