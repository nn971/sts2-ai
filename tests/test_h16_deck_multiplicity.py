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
    assert len(encoded.objects) == 2  # global state + one deck multiset
    clone = encoded.objects[-1]
    assert clone.observed[13] == 1  # count is represented, not discarded
    assert clone.numeric[13] == pytest.approx(
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
    assert len(encoded.objects) == 4  # global + three visibly distinct classes
    assert encoded.objects[1].numeric[13] == pytest.approx(
        math.tanh(math.log1p(2) / 10), abs=1e-7,
    )


def test_no_silent_truncation_for_many_distinct_cards():
    deck = [
        {"instance_id": i, "card_id": f"demo.card.{i}", "upgrade_level": 0}
        for i in range(300)
    ]
    with pytest.raises(ValueError, match="Public object cap"):
        encode_h16_decision(make_state(deck), ACTIONS)
