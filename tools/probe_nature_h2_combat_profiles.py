#!/usr/bin/env python3
"""Real C# terminal/replay probe of explicitly pinned harder Nature contexts."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from sts2_ai.emulator.semantic_p0_combat import strike_first_player
from sts2_ai.nature.combat_scenarios import SCENARIOS, parse_scenario_pool
from sts2_ai.nature.contextual_draw import ContextualDrawNature, restore_nature
from sts2_ai.nature.draw_only_episode import play_draw_only_episode
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", required=True, type=Path)
    parser.add_argument("--emulator-bindings", required=True, type=Path)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument(
        "--scenarios", default="shrinker-a0,shrinker-a10,nibbits-a0,nibbits-a10"
    )
    parser.add_argument("--max-boundaries", type=int, default=600)
    parser.add_argument("--nature-model", choices=("tabular", "contextual"),
                        default="tabular")
    args = parser.parse_args()
    scenarios = parse_scenario_pool(args.scenarios)
    sys.path.insert(0, str(args.emulator_bindings.resolve()))
    Session = importlib.import_module("sts2_emulator").Session
    with Session(["dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl"]) as session:
        contextual = (ContextualDrawNature(seed=1)
                      if args.nature_model == "contextual" else None)
        training_episodes = []
        for key in scenarios:
            scenario = SCENARIOS[key]
            options = dict(
                source_seed="nature-h2-scenario-replay-" + key,
                reference_seed=779,
                nature=contextual if contextual is not None else TwoSiteNature(seed=1),
                emulator_revision=args.emulator_revision,
                player_choose=strike_first_player,
                player_model_id="test-only-strike-first",
                entry_mode="seeded-fixed-shrinker",
                draw_mode=("learned" if args.nature_model == "contextual"
                           else "reference"),
                max_boundaries=args.max_boundaries,
                scenario=scenario,
            )
            actual = play_draw_only_episode(session, **options)
            replayed = play_draw_only_episode(
                session, **{**options, "replay": actual.decisions}
            )
            if actual.digest != replayed.digest or not actual.terminal_replay_verified:
                raise RuntimeError(f"Terminal replay differs for {key}")
            if actual.nontrivial_draw_count <= 0:
                raise RuntimeError(f"No nontrivial draw choices for {key}")
            if actual.scenario_id != key:
                raise RuntimeError("Scenario provenance lost during replay")
            if contextual is not None:
                training_episodes.append(actual.for_nature_training())
            print(json.dumps({
                "scenario_id": key,
                "nature_model": args.nature_model,
                "outcome": actual.outcome,
                "score": actual.score,
                "exit_hp": actual.exit_hp,
                "nontrivial_draws": actual.nontrivial_draw_count,
                "reference_intents": actual.reference_intent_count,
                "replay_verified": True,
            }, sort_keys=True), flush=True)
        if contextual is not None:
            if len(training_episodes) < 2:
                raise RuntimeError("Contextual update smoke requires two real combats")
            update = contextual.update(tuple(training_episodes))
            restored = restore_nature(contextual.dump())
            if restored.revision != contextual.revision:
                raise RuntimeError("Contextual checkpoint failed to restore revision")
            repeated = dict(options)
            repeated["nature"] = restored
            repeated["source_seed"] = "nature-h2-post-update-replay"
            learned = play_draw_only_episode(session, **repeated)
            again = play_draw_only_episode(
                session, **{**repeated, "replay": learned.decisions}
            )
            if learned.digest != again.digest:
                raise RuntimeError("Post-update contextual Nature replay mismatch")
            print(json.dumps({
                "training_smoke": "PASS",
                "nature_gradient_norm": update["nature_gradient_norm"],
                "nature_revision": restored.revision,
                "post_update_replay": True,
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
