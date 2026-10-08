#!/usr/bin/env python3
"""Smoke test actual native Overgrowth gameplay through the AI's public API.

No teacher, hidden-state reads, state mutation or direct C# factory calls.
Records whether a legitimate public-only heuristic reaches card rewards and
whether any card reward is taken, alongside observed persistent deck growth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sts2_ai.agents import Decision, HeuristicAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=2048)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.max_decisions < 1:
        parser.error("--runs and --max-decisions must be positive")

    results: list[dict[str, Any]] = []
    with JsonlEmulatorBackend(build=args.build) as backend:
        for run_index in range(args.runs):
            counters = {
                "reward_menus": 0,
                "card_offers_seen": 0,
                "card_rewards_taken": 0,
                "card_rewards_skipped": 0,
                "observed_max_deck_size": 13,
                "combat_decisions": 0,
                "neow_choices": 0,
            }

            def record_decision(
                _state: str,
                observation: Any,
                actions: Any,
                decision: Decision,
            ) -> None:
                public = json.loads(observation.payload_json)
                deck = public.get("deck")
                if isinstance(deck, list):
                    counters["observed_max_deck_size"] = max(
                        counters["observed_max_deck_size"], len(deck)
                    )
                kinds = {action.kind for action in actions}
                if "take_reward_card" in kinds:
                    counters["reward_menus"] += 1
                    counters["card_offers_seen"] += sum(
                        action.kind == "take_reward_card" for action in actions
                    )
                if decision.action.kind == "take_reward_card":
                    counters["card_rewards_taken"] += 1
                if decision.action.kind == "skip_reward_card":
                    counters["card_rewards_skipped"] += 1
                if decision.action.kind == "play_card":
                    counters["combat_decisions"] += 1
                if decision.action.kind == "event_choice" and public.get("floor") == 0:
                    counters["neow_choices"] += 1

            seed = f"native-overgrowth-ai-smoke-{run_index}"
            summary = play_run(
                backend, HeuristicAgent(), seed=seed,
                policy=InformationPolicy(FAIR_POLICY_ID),
                environment=NATIVE_OVERGROWTH,
                max_decisions=args.max_decisions,
                decision_observer=record_decision,
            )
            results.append({
                "seed": seed,
                "outcome": summary.outcome,
                "censored": summary.censored,
                "terminal_act": summary.terminal_act,
                "terminal_floor": summary.terminal_floor,
                "decisions": summary.decisions,
                "act1_cleared": summary.act1_cleared,
                "frontier_progress": summary.frontier_progress,
                **counters,
            })
        report = {
            "schema": "sts2-native-overgrowth-public-run-smoke-v1",
            "emulator_revision": backend.emulator_revision,
            "environment": NATIVE_OVERGROWTH,
            "runs": results,
            "total_card_reward_menus": sum(row["reward_menus"] for row in results),
            "total_card_rewards_taken": sum(row["card_rewards_taken"] for row in results),
            "any_deck_growth_observed": any(
                row["observed_max_deck_size"] > 13 for row in results
            ),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["total_card_reward_menus"] == 0:
        raise SystemExit(
            "No native Overgrowth card-reward menu encountered: "
            "inspect run progression before attempting large neural training."
        )
    if report["total_card_rewards_taken"] == 0:
        raise SystemExit(
            "Heuristic did not take any card reward: "
            "review decision policy before large neural training."
        )


if __name__ == "__main__":
    main()
