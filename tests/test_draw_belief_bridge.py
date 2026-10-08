"""Proof of inventory consistency on real pinned prototype opening frames.

The source event (directly stepping into a new normal combat) is the
certificate. No attempt is made to assert native RNG's exact permutation law.
"""
from __future__ import annotations

import json
import random
import shutil

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.draw_belief import certified_opening_draw_belief


def test_real_pinned_opening_combat_known_deck_inventory() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Requires pinned .NET emulator bridge")
    frames_checked = 0
    with JsonlEmulatorBackend(build=True) as backend:
        policy = InformationPolicy(FAIR_POLICY_ID)
        for index in range(28):
            handle = backend.reset(f"draw-belief-audit:{index}")
            try:
                actions = tuple(backend.legal_actions(handle))
                assert len(actions) == 1 and actions[0].kind == "start_run"
                transition = backend.step(handle, actions[0])
                backend.release_many([handle])
                handle = transition.child

                # Select the initial map node by the actual public legal menu,
                # not by looking ahead through hidden RNG or exact-state expansion.
                map_actions = tuple(
                    action for action in backend.legal_actions(handle)
                    if action.kind == "choose_map_node"
                )
                assert map_actions
                next_step = backend.step(handle, map_actions[0])
                backend.release_many([handle])
                handle = next_step.child
                obs = backend.observe(handle, policy)
                state = json.loads(obs.payload_json)
                if not isinstance(state.get("combat"), dict):
                    continue
                combat = state["combat"]
                if combat.get("discard_pile") != [] or combat.get("exhaust_pile") != []:
                    continue

                # Exact source-of-frame certificate: the immediately preceding
                # transition was a map-room entry with no player combat action.
                belief = certified_opening_draw_belief(
                    obs, opening_frame_certified=True
                )
                assert belief.remaining == combat["draw_pile_count"]
                assert len(belief.sample_ordered(
                    min(2, belief.remaining), search_rng=random.Random(index)
                )) == min(2, belief.remaining)
                assert sum(x.probability for x in belief.ordered_distribution(1)) == pytest.approx(1)
                frames_checked += 1
            finally:
                backend.release_many([handle])
    assert frames_checked >= 6
