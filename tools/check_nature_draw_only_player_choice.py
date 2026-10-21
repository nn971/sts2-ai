#!/usr/bin/env python3
"""Exercise real engine player_choice and exact replay using the Survivor fixture."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sts2_ai.emulator.semantic_p0_combat import strike_first_player
from sts2_ai.nature.draw_only_episode import play_draw_only_episode
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-dll", required=True, type=Path)
    parser.add_argument("--emulator-bindings", required=True, type=Path)
    parser.add_argument("--emulator-revision", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.emulator_bindings.resolve()))
    from sts2_emulator import Session  # noqa: PLC0415

    class SurvivorFixture:
        def __init__(self, session):
            self._session = session
            self.capabilities = session.capabilities

        def begin_seeded_semantic_combat(self, seed, **kwargs):
            return self._session.begin_seeded_semantic_combat(
                seed, fixture_selection_card_id="proto.silent.survivor",
                **kwargs,
            )

        def load_semantic_snapshot(self, snap):
            return self._session.load_semantic_snapshot(snap)

        def release_semantic_many(self, states):
            return self._session.release_semantic_many(states)

    used_survivor = False

    def choose(observation, actions):
        nonlocal used_survivor
        hand = {
            card["instance_id"]: card["card_id"]
            for card in observation.get("combat", {}).get("hand", [])
        }
        if not used_survivor:
            for action in actions:
                if action.get("kind") != "play_card":
                    continue
                card_id = json.loads(action["payloadJson"]).get("CardInstanceId")
                if hand.get(card_id) == "proto.silent.survivor":
                    used_survivor = True
                    return action["actionId"]
        return strike_first_player(observation, actions)

    with Session(["dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl"]) as source:
        session = SurvivorFixture(source)
        options = dict(
            source_seed="nature-h1-player-choice-survivor",
            reference_seed=129,
            nature=TwoSiteNature(seed=17),
            emulator_revision=args.emulator_revision,
            player_model_id="test-only-survivor-fair-choice",
            entry_mode="seeded-fixed-shrinker", draw_mode="reference",
            max_boundaries=600,
        )
        one = play_draw_only_episode(session, player_choose=choose, **options)
        if not used_survivor:
            raise RuntimeError("Survivor fixture was not selected by the fair player")
        player_choices = tuple(s for s in one.decisions if s.phase == "player_choice")
        if not player_choices:
            raise RuntimeError("A Survivor card did not reach a real player_choice")
        two = play_draw_only_episode(session, replay=one.decisions, **options)
        if one.digest != two.digest or not one.terminal_replay_verified:
            raise RuntimeError("Real Survivor player_choice did not replay exactly")
        print(
            f"[h1-player-choice] PASS choices={len(player_choices)} "
            f"combat={one.outcome} exit_hp={one.exit_hp} "
            f"draws={one.nontrivial_draw_count} "
            f"source_replay_identical={one.digest == two.digest}",
            flush=True,
        )


if __name__ == "__main__":
    main()
