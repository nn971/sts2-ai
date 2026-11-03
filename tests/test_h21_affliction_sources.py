"""H21 affliction-source pointers are fair, typed and ID-renaming invariant."""
from __future__ import annotations

import copy

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import KINDS, ZONES
from sts2_ai.training.h19_resource_codec import (
    EMULATOR_PUBLIC_SCHEMA_V1,
    encode_h19_resource_decision,
)
from sts2_ai.training.h21_macro_offer_codec import (
    AFFLICTION_SOURCE,
    encode_h21_macro_decision,
)


def visible_combat():
    card = lambda ident, source: {
        "instance_id": ident, "card_id": "silent.strike",
        "upgrade_level": 0, "state": {},
        "affliction": {
            "kind": "Bound", "amount": 1,
            "source_enemy_instance_id": source,
        },
    }
    return {
        "hp": 53, "max_hp": 70, "gold": 40, "act": 1, "floor": 4,
        "deck": [], "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3, "map": [], "current_map_node_id": None,
        "combat": {
            "turn": 2, "energy": 3, "player_block": 0,
            "draw_pile_count": 2,
            "hand": [card(100, 11), card(101, 22)],
            "discard_pile": [card(102, 11), card(103, 22)],
            "exhaust_pile": [],
            "play_pile": [card(104, 11)],
            "selection_candidates": [],
            "selection_selected_cards": [],
            "enemies": [
                {"instance_id": 11, "enemy_id": "slime", "hp": 18,
                 "block": 0, "powers": [], "statuses": {}},
                {"instance_id": 22, "enemy_id": "slime", "hp": 18,
                 "block": 0, "powers": [], "statuses": {}},
            ],
            "history": {"cards_played_this_turn": 1},
            "player_powers": [],
        },
    }


ACTIONS = (LegalAction("end", "end_turn", "{}"),)


def encode(state):
    return encode_h21_macro_decision(
        state, ACTIONS, public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1,
    )


def edges(decision):
    return {(edge.source, edge.target) for edge in decision.relations
            if edge.relation_type == AFFLICTION_SOURCE}


def test_sources_bind_hand_discard_and_play_pile_to_visible_enemies():
    encoded = encode(visible_combat())
    assert len(edges(encoded)) == 5
    enemies = [i for i, obj in enumerate(encoded.objects)
               if obj.kind == KINDS.index("enemy")]
    assert len(enemies) == 2
    assert {dst for _, dst in edges(encoded)} == set(enemies)
    hand = [i for i, obj in enumerate(encoded.objects)
            if obj.kind == KINDS.index("card")
            and obj.zone == ZONES.index("hand")]
    discard = [i for i, obj in enumerate(encoded.objects)
               if obj.kind == KINDS.index("card")
               and obj.zone == ZONES.index("discard")]
    assert len(hand) == len(discard) == 2
    assert {(hand[0], enemies[0]), (hand[1], enemies[1])} <= edges(encoded)
    assert {(discard[0], enemies[0]), (discard[1], enemies[1])} <= edges(encoded)
    played = [src for src, _ in edges(encoded) if src not in hand + discard]
    assert len(played) == 1
    assert encoded.objects[played[0]].zone == ZONES.index("event")


def test_consistent_renaming_of_all_opaque_enemy_ids_changes_no_embeddings():
    source = visible_combat()
    before = encode(source)
    renamed = copy.deepcopy(source)
    mapping = {11: 2001, 22: 2002}
    for enemy in renamed["combat"]["enemies"]:
        enemy["instance_id"] = mapping[enemy["instance_id"]]
    for zone in ("hand", "discard_pile", "play_pile"):
        for card in renamed["combat"][zone]:
            affliction = card["affliction"]
            affliction["source_enemy_instance_id"] = mapping[
                affliction["source_enemy_instance_id"]]
    assert encode(renamed) == before


def test_reassigning_affliction_source_changes_relation_not_raw_attributes():
    source = visible_combat()
    before = encode(source)
    changed = copy.deepcopy(source)
    changed["combat"]["hand"][0]["affliction"]["source_enemy_instance_id"] = 22
    after = encode(changed)
    assert before != after
    hand = [i for i, obj in enumerate(before.objects)
            if obj.kind == KINDS.index("card") and obj.zone == ZONES.index("hand")]
    assert before.objects[hand[0]] == after.objects[hand[0]]
    assert edges(before) != edges(after)


def test_unseen_enemy_source_and_legacy_h19_both_fail_closed():
    state = visible_combat()
    with pytest.raises(ValueError, match="explicit relation"):
        encode_h19_resource_decision(
            state, ACTIONS, public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1,
        )
    state["combat"]["hand"][0]["affliction"]["source_enemy_instance_id"] = 999
    with pytest.raises(ValueError, match="non-visible source enemy"):
        encode(state)


def test_selection_pool_source_relationship_is_preserved():
    state = visible_combat()
    candidate = copy.deepcopy(state["combat"]["hand"][0])
    candidate["instance_id"] = 605
    state["combat"]["selection_candidates"] = [candidate]
    encoded = encode(state)
    reward_cards = [i for i, obj in enumerate(encoded.objects)
                    if obj.kind == KINDS.index("card")
                    and obj.zone == ZONES.index("reward")]
    enemies = [i for i, obj in enumerate(encoded.objects)
               if obj.kind == KINDS.index("enemy")]
    assert len(reward_cards) == 1
    assert (reward_cards[0], enemies[0]) in edges(encoded)
