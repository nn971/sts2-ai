"""Canonical prototype-ai-v2 enemy power representation acceptance.

H21 v5 is opt-in. V1 H19/H21 encodings retain their original semantics.
"""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import KINDS, ZONES
from sts2_ai.training.h19_resource_codec import (
    EMULATOR_PUBLIC_SCHEMA_V1,
    EMULATOR_PUBLIC_SCHEMA_V2,
    encode_h19_resource_decision,
)
from sts2_ai.training.h21_macro_offer_codec import (
    H21_OBJECT_SCHEMA,
    H21_POWER_OBJECT_SCHEMA,
    encode_h21_macro_decision,
)


def _state():
    return {
        "hp": 55, "max_hp": 70, "gold": 90,
        "act": 1, "floor": 3, "deck": [], "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3, "map": [], "current_map_node_id": None,
        "combat": {
            "turn": 1, "energy": 3, "player_block": 0,
            "hand": [], "draw_pile_count": 0,
            "discard_pile": [], "exhaust_pile": [], "play_pile": [],
            "enemies": [{
                "instance_id": 11, "enemy_id": "proto.enemy.crawler",
                "hp": 30, "block": 0, "move_id": "attack",
                "statuses": {},
                "powers": [
                    {"power_id": "proto.power.poison", "stacks": 3},
                    {"power_id": "proto.power.weak", "stacks": 2},
                    {"power_id": "proto.power.strength", "stacks": 1},
                ],
            }],
            "player_powers": [], "pending_choice_id": None,
            "history": {
                "cards_played_this_turn": 0,
                "attacks_played_this_turn": 0,
                "skills_played_this_turn": 0,
                "attacks_played_last_turn": 0,
                "cards_played_this_combat": 0,
                "manual_cards_played_this_turn": 0,
            },
        },
    }


def _actions():
    return [LegalAction("end_turn:{}", "end_turn", json.dumps({}))]


def _encode(state, schema=EMULATOR_PUBLIC_SCHEMA_V2):
    return encode_h21_macro_decision(
        state, _actions(), public_schema_id=schema)


def test_v2_is_new_schema_with_one_power_concept_per_enemy_effect():
    frame = _encode(_state())
    assert frame.schema == H21_POWER_OBJECT_SCHEMA
    enemies = [(i, x) for i, x in enumerate(frame.objects)
               if x.kind == KINDS.index("enemy")
               and x.zone == ZONES.index("enemy")]
    assert len(enemies) == 1
    index, _ = enemies[0]
    powers = [x for x in frame.objects
              if x.kind == KINDS.index("power") and x.owner == index]
    assert len(powers) == 3
    assert all(x.observed[7] == 1 for x in powers)
    assert not any(x.kind == KINDS.index("status") for x in frame.objects)


def test_v1_encoding_and_h19_rejection_are_preserved():
    state = _state()
    state["combat"]["enemies"][0]["statuses"]["proto.status.weak"] = 1
    legacy = _encode(state, EMULATOR_PUBLIC_SCHEMA_V1)
    assert legacy.schema == H21_OBJECT_SCHEMA
    assert any(x.kind == KINDS.index("status") for x in legacy.objects)
    with pytest.raises(ValueError, match="prototype-ai-v1"):
        encode_h19_resource_decision(
            _state(), _actions(), public_schema_id=EMULATOR_PUBLIC_SCHEMA_V2)


@pytest.mark.parametrize("mutate,pattern", [
    (lambda s: s["combat"]["enemies"][0]["statuses"].update(
        {"proto.status.weak": 2}), "legacy enemy statuses"),
    (lambda s: s["combat"]["enemies"][0]["powers"].append(
        {"power_id": "proto.power.poison", "stacks": 2}),
     "Duplicate canonical enemy"),
    (lambda s: s["combat"]["enemies"][0]["powers"][0].update(
        {"stacks": 0}), "Invalid canonical enemy"),
    (lambda s: s.update({"combat": None}), "requires a combat"),
])
def test_v2_rejects_unmerged_or_invalid_enemy_effects(mutate, pattern):
    state = copy.deepcopy(_state())
    mutate(state)
    with pytest.raises(ValueError, match=pattern):
        _encode(state)


def test_v2_keeps_legacy_combat_targets_and_action_arity():
    state = _state()
    state["combat"]["enemies"].append({
        "instance_id": 12, "enemy_id": "proto.enemy.raider",
        "hp": 12, "block": 0, "move_id": "attack",
        "statuses": {}, "powers": [],
    })
    actions = [LegalAction(
        "use_potion", "use_potion",
        json.dumps({"Slot": 0, "TargetEnemyId": 12}),
    )]
    state["potions"] = [{"slot": 0, "potion_id": "proto.potion.test",
                         "state": {}}]
    state["potion_slots_occupied"] = 1
    state["potion_slots_empty"] = 2
    encoded = encode_h21_macro_decision(
        state, actions, public_schema_id=EMULATOR_PUBLIC_SCHEMA_V2)
    assert len(encoded.actions) == 1
    assert encoded.actions[0].target >= 0
    assert encoded.objects[encoded.actions[0].target].kind == KINDS.index("enemy")
