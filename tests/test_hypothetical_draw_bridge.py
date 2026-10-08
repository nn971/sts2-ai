"""Experimental full-state draw-order realization; *not* fair RL labels."""
from __future__ import annotations

import json
import random
import shutil

import pytest

from sts2_ai.emulator import (
    FAIR_POLICY_ID,
    InformationPolicy,
    JsonlBridgeError,
    JsonlEmulatorBackend,
)
from sts2_ai.emulator.draw_belief import certified_opening_draw_belief


def test_pinned_experimental_draw_injection_is_hypothetical_only() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned integration requires dotnet")
    with JsonlEmulatorBackend(build=True) as backend:
        assert backend.hypothetical_draw_order_schema == (
            "prototype-hypothetical-draw-order-v1"
        )
        policy = InformationPolicy(FAIR_POLICY_ID)

        live = backend.reset("real-private-game")
        try:
            with pytest.raises(JsonlBridgeError, match="hypothetical"):
                backend.hypothetical_draw_order(live, [])
        finally:
            backend.release_many([live])

        successful = False
        for search_seed in range(1, 28):
            root = backend.reset_hypothetical(search_seed)
            handles = [root]
            try:
                assert backend.hypothetical_draw_order_schema is not None
                start = backend.legal_actions(root)[0]
                map_state = backend.step(root, start).child
                handles.append(map_state)

                # Follow the first available *public* map choice, exactly
                # like the established pinned opening-draw integration test.
                # This avoids depending on action payload casing/room enums.
                choice = next(
                    (
                        action for action in backend.legal_actions(map_state)
                        if action.kind == "choose_map_node"
                    ),
                    None,
                )
                if choice is None:
                    continue
                combat_state = backend.step(map_state, choice).child
                handles.append(combat_state)
                before = backend.observe(combat_state, policy)
                if json.loads(before.payload_json).get("combat") is None:
                    continue
                belief = certified_opening_draw_belief(
                    before, opening_frame_certified=True
                )
                drawn_tokens = belief.sample_ordered(
                    belief.remaining, search_rng=random.Random(101)
                )
                cards = [tuple(json.loads(token)) for token in drawn_tokens]
                assert all(isinstance(card_id, str) and isinstance(upgrade, int)
                           for card_id, upgrade in cards)
                branched = backend.hypothetical_draw_order(combat_state, cards)
                handles.append(branched)
                after = backend.observe(branched, policy)
                assert before == after
                assert backend.legal_actions(branched) == backend.legal_actions(combat_state)

                with pytest.raises(JsonlBridgeError, match="permutation"):
                    backend.hypothetical_draw_order(
                        combat_state, [("NO_SUCH_CARD", 0)] * belief.remaining
                    )

                end_turn = next(
                    item for item in backend.legal_actions(branched)
                    if item.kind == "end_turn"
                )
                advanced = backend.step(branched, end_turn).child
                handles.append(advanced)
                assert backend.observe(advanced, policy).policy_id == FAIR_POLICY_ID
                with pytest.raises(JsonlBridgeError, match="fresh-combat"):
                    backend.hypothetical_draw_order(advanced, cards)
                successful = True
                break
            finally:
                backend.release_many(handles)
        assert successful, "Could not locate a clean certified opening combat"
