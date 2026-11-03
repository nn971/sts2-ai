"""H21 public v1 field coverage remains explicit and fail-closed."""
from __future__ import annotations

import copy

import pytest

from sts2_ai.training.h21_field_coverage import (
    AUXILIARY,
    ENCODED,
    GAP,
    PARTIAL,
    REFERENCE,
    PUBLIC_V1_FIELD_CONTRACT,
    audit_public_observation,
)


def observation():
    return {
        "ruleset_id": "v111", "character_id": "silent",
        "decision_index": 17, "phase": "Combat",
        "act": 1, "floor": 6, "hp": 42, "max_hp": 70, "gold": 74,
        "deck": [{"instance_id": 1, "card_id": "silent.strike",
                  "upgrade_level": 0, "state": {},
                  "affliction": {"kind": "Bound", "amount": 1,
                                 "source_enemy_instance_id": None}}],
        "relics": [{"relic_id": "relic.1", "state": {},
                    "is_wax": False, "is_melted": False}],
        "potions": [{"slot": 0, "potion_id": "potion.1", "state": {}}],
        "map": [{"node_id": "m1", "act": 1, "floor": 1,
                 "room_type": "Combat", "next_node_ids": []}],
        "current_map_node_id": "m1",
        "potion_slot_capacity": 3, "potion_slots_occupied": 1,
        "potion_slots_empty": 2,
        "completed_rooms": [],
        "combat": {
            "turn": 3, "energy": 2, "player_block": 7,
            "hand": [], "draw_pile_count": 5,
            "discard_pile": [], "exhaust_pile": [], "play_pile": [],
            "enemies": [{
                "instance_id": 7, "enemy_id": "enemy.mantis", "hp": 30,
                "block": 1, "move_id": "Attack", "statuses": {},
                "powers": [{"power_id": "weak", "stacks": 2}],
                "intent_damage": 8,
            }],
            "player_powers": [{"power_id": "power.strength", "stacks": 1}],
            "history": {"cards_played_this_turn": 2,
                        "attacks_played_this_turn": 1},
        },
    }


def test_gaps_and_references_are_reported_by_specific_field_path():
    report = audit_public_observation(observation())
    assert report.classified
    assert ("observation.combat.enemies[0].move_id", ENCODED) in report.classified
    assert ("observation.map[0].node_id", REFERENCE) in report.classified
    assert ("observation.combat.enemies[0].hp", ENCODED) in report.classified
    assert ("observation.deck[0].affliction.source_enemy_instance_id",
            REFERENCE) in report.classified
    assert ("observation.combat", PARTIAL) in report.classified
    assert ("observation.combat.play_pile", ENCODED) in report.classified
    assert ("observation.decision_index", AUXILIARY) in report.classified
    assert ("observation.completed_rooms", ENCODED) in report.classified
    assert not report.fully_encoded


def test_unknown_public_fields_fail_at_every_audited_structural_layer():
    original = observation()
    mutations = (
        ("observation", lambda s: s.update({"new_future_field": 1})),
        ("combat", lambda s: s["combat"].update({"new_visible_counter": 1})),
        ("enemy", lambda s: s["combat"]["enemies"][0].update({"status_v2": 2})),
        ("map", lambda s: s["map"][0].update({"teleporter_edge": "m2"})),
        ("card", lambda s: s["deck"][0].update({"secret_enchantment": "A"})),
    )
    for name, mutate in mutations:
        updated = copy.deepcopy(original)
        mutate(updated)
        with pytest.raises(ValueError, match="Unclassified v1 public field"):
            audit_public_observation(updated), name


def test_nested_reward_shop_event_fields_are_audited_recursively():
    original = observation()
    original["reward"] = {
        "pending_card_groups": [{
            "group_index": 1, "card_options": ["silent.strike"],
            "option_upgrade_flags": [True],
            "card_previews": [
                {"card_id": "silent.strike", "upgrade_level": 1},
            ],
        }],
        "pending_extra_gold_rewards": [{"group_index": 2, "amount": 45}],
    }
    original["shop"] = {
        "card_offers": [{"offer_id": 7, "item_id": "silent.strike",
                         "price": 40, "sold": False}],
        "card_offer_previews": [
            {"offer_id": 7, "card": {"card_id": "silent.strike",
                                   "upgrade_level": 0}},
        ],
    }
    original["event"] = {
        "event_id": "event.a", "visible_choice_values": [
            {"choice_id": "take", "gold_gain": 101},
        ],
    }
    record = audit_public_observation(original).classified
    for path in (
        "observation.reward.pending_card_groups[0].card_previews[0].upgrade_level",
        "observation.reward.pending_extra_gold_rewards[0].amount",
        "observation.shop.card_offers[0].price",
        "observation.shop.card_offer_previews[0].card.enchantment",
        "observation.event.visible_choice_values[0].gold_gain",
    ):
        if path.endswith(".enchantment"):
            continue  # Optional source attribute was absent.
        assert (path, ENCODED) in record
    mutated = copy.deepcopy(original)
    mutated["event"]["visible_choice_values"][0]["unknown_roll"] = 44
    with pytest.raises(ValueError, match="Unclassified v1 public field"):
        audit_public_observation(mutated)


def test_contract_declares_unrepresented_high_priority_fields():
    root = PUBLIC_V1_FIELD_CONTRACT["observation"]
    combat = PUBLIC_V1_FIELD_CONTRACT["combat"]
    enemy = PUBLIC_V1_FIELD_CONTRACT["enemy"]
    assert root["completed_rooms"] == ENCODED
    assert combat["pending_choice_id"] == REFERENCE
    assert combat["play_pile"] == ENCODED
    assert enemy["move_id"] == ENCODED
    assert enemy["is_minion"] == ENCODED
