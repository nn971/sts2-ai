"""H16 deck multiplicity without losing public card distinctions."""
from __future__ import annotations

import math

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import encode_public_decision
from sts2_ai.training.h16_public_codec import encode_h16_decision


def make_state(deck: list[dict]) -> dict:
    return {
        "act": 1, "floor": 5, "hp": 70, "max_hp": 70,
        "deck": deck, "combat": None,
        "relics": [], "potions": [], "map": [],
    }


ACTIONS = (LegalAction("pass", "skip", "{}"),)


def test_thousand_identical_clones_become_one_inventory_token():
    deck = [
        {"instance_id": i, "card_id": "demo.strike", "upgrade_level": 0,
         "persistent_state": {"times_used": 0}}
        for i in range(1000)
    ]
    encoded = encode_h16_decision(make_state(deck), ACTIONS)
    assert len(encoded.objects) == 3  # global + deck class + one persistent leaf
    clone = encoded.objects[1]
    assert clone.observed[14] == 1  # count is represented, not discarded
    assert clone.numeric[14] == pytest.approx(
        math.tanh(math.log1p(1000) / 10), abs=1e-7,
    )
    # H7's legacy codec still rejects its old 256-object cap instead of
    # silently dropping potentially action-referenced objects.
    with pytest.raises(ValueError, match="Public object cap"):
        encode_public_decision(make_state(deck), ACTIONS)


def test_public_modifications_are_part_of_group_identity():
    deck = [
        {"instance_id": 101, "card_id": "demo.strike", "upgrade_level": 0,
         "persistent_state": {"counter": 1}},
        {"instance_id": 102, "card_id": "demo.strike", "upgrade_level": 0,
         "persistent_state": {"counter": 2}},
        {"instance_id": 103, "card_id": "demo.strike", "upgrade_level": 1,
         "persistent_state": {"counter": 1}},
        {"instance_id": 104, "card_id": "demo.strike", "upgrade_level": 0,
         "persistent_state": {"counter": 1}},
    ]
    encoded = encode_h16_decision(make_state(deck), ACTIONS)
    assert len(encoded.objects) == 7  # global + 3 deck classes + 3 persistent leaves
    assert encoded.objects[1].numeric[14] == pytest.approx(
        math.tanh(math.log1p(2) / 10), abs=1e-7,
    )


def test_no_silent_truncation_for_many_distinct_cards():
    deck = [
        {"instance_id": i, "card_id": f"demo.card.{i}", "upgrade_level": 0}
        for i in range(300)
    ]
    with pytest.raises(ValueError, match="Public object cap"):
        encode_h16_decision(make_state(deck), ACTIONS)


def test_thousand_identical_selectable_cards_use_one_pooled_card_object():
    deck = [{"instance_id": i, "card_id": "demo.strike"} for i in range(1000)]
    public = make_state(deck)
    public["combat"] = {
        "turn": 1, "hand": [], "discard_pile": deck,
        "exhaust_pile": [], "selection_candidates": deck,
        "enemies": [], "player_powers": [],
    }
    selections = (
        LegalAction("finish", "select_cards", '{"CardInstanceIds":[]}'),
        LegalAction("choose-one", "select_cards", '{"CardInstanceIds":[0]}'),
    )
    encoded = encode_h16_decision(public, selections)
    assert len(encoded.objects) == 4  # global/player, discard class, deck class
    assert len(set(encoded.actions)) == 2
    assert encoded.actions[1].members == (2,)


def test_distinct_internal_variants_sharing_one_public_card_class_share_token():
    # The emulator can distinguish internal combat-card variants while the
    # public card projection remains identical. Retain all legal choices.
    deck = [{"instance_id": i, "card_id": "demo.strike"} for i in range(3)]
    public = make_state(deck)
    public["combat"] = {
        "turn": 1, "hand": [], "discard_pile": deck,
        "exhaust_pile": [], "selection_candidates": deck, "enemies": [],
    }
    choices = tuple(
        LegalAction(str(i), "select_cards", '{"CardInstanceIds":[' + str(i) + ']}')
        for i in range(3)
    )
    encoded = encode_h16_decision(public, choices)
    assert len(encoded.actions) == 3  # Preserve full emulator branching.
    assert len(set(encoded.actions)) == 1  # Identical public tokens.
    assert all(a.members == (2,) for a in encoded.actions)
    reversed_encoded = encode_h16_decision(public, choices[::-1])
    assert reversed_encoded.actions == encoded.actions[::-1]
