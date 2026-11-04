"""H16 select_cards aliases must preserve public-card multiset semantics."""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator import LegalAction
from sts2_ai.training.h16_public_codec import encode_h16_decision


def choice(ids: list[int], *, extra: dict | None = None) -> LegalAction:
    payload = {"CardInstanceIds": ids, **(extra or {})}
    return LegalAction(
        "pick-" + "-".join(map(str, ids)) + json.dumps(extra or {}),
        "select_cards", json.dumps(payload, sort_keys=True),
    )


def selection_state() -> dict:
    return {
        "floor": 1, "hp": 70, "max_hp": 70, "deck": [],
        "combat": {
            "turn": 1, "energy": 3, "hand": [],
            "discard_pile": [], "exhaust_pile": [],
            "enemies": [],
            # 2 and 5 are the *same complete public card class*.
            # 11 is distinct; 7 differs by enchantment.
            "selection_candidates": [
                {"instance_id": 11, "card_id": "demo.anchor", "cost": 1},
                {"instance_id": 5, "card_id": "demo.copy", "cost": 0},
                {"instance_id": 2, "card_id": "demo.copy", "cost": 0},
                {"instance_id": 7, "card_id": "demo.copy", "cost": 0,
                 "enchantments": [{"id": "demo.enchanted"}]},
            ],
        },
    }


def test_h16_two_card_selection_public_alias_keeps_all_legal_choices():
    choices = [
        choice([11, 5]), choice([11, 2]), choice([5, 11]),
        choice([11, 7]), choice([5, 2]),
    ]
    encoded = encode_h16_decision(selection_state(), choices)
    assert len(encoded.actions) == len(choices)
    assert len(encoded.actions[0].members) == 2
    assert encoded.actions[0] == encoded.actions[1]
    assert encoded.actions[0] == encoded.actions[2]
    assert encoded.actions[0] != encoded.actions[3]  # enchantment matters
    assert encoded.actions[0] != encoded.actions[4]  # multiset matters
    assert len(encoded.objects) < 256


def test_h16_three_card_selection_alias_keeps_multiplicities():
    state = selection_state()
    state["combat"]["selection_candidates"].append(
        {"instance_id": 3, "card_id": "demo.copy", "cost": 0},
    )
    encoded = encode_h16_decision(
        state, [choice([11, 5, 2]), choice([11, 2, 3]),
                choice([11, 2, 7])],
    )
    assert len(encoded.actions[0].members) == 3
    assert encoded.actions[0] == encoded.actions[1]
    assert encoded.actions[0] != encoded.actions[2]


def test_h16_same_multiset_with_different_other_payload_is_rejected():
    with pytest.raises(ValueError, match="action representation collision"):
        encode_h16_decision(
            selection_state(),
            [choice([11, 5], extra={"Destination": "hand"}),
             choice([11, 2], extra={"Destination": "exhaust"})],
        )


def test_h16_combat_selection_missing_candidate_remains_rejected():
    with pytest.raises(ValueError, match="candidate missing from public"):
        encode_h16_decision(selection_state(), [choice([11, 999])])


def test_h16_select_cards_illegal_duplicate_instance_remains_rejected():
    with pytest.raises(ValueError, match="Malformed select_cards"):
        encode_h16_decision(selection_state(), [choice([5, 5])])
