"""H21 public pending-choice continuation and offer eligibility regressions."""
from __future__ import annotations

import copy
import json

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import KINDS, ZONES
from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from sts2_ai.training.h21_macro_offer_codec import (
    GROUP_OPTION,
    encode_h21_macro_decision,
)


def card(instance_id, card_id, upgrade_level=0):
    return {
        "instance_id": instance_id, "card_id": card_id,
        "upgrade_level": upgrade_level, "state": {},
    }


def state():
    return {
        "act": 1, "floor": 7, "hp": 45, "max_hp": 70, "gold": 95,
        "deck": [
            card(101, "proto.silent.strike"),
            card(102, "proto.silent.strike"),
            card(103, "proto.silent.defend"),
            card(104, "proto.silent.strike", 1),
        ],
        "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3, "map": [],
        "current_map_node_id": None, "combat": None,
    }


def action(kind, **payload):
    return LegalAction(
        kind + ":" + json.dumps(payload, sort_keys=True), kind,
        json.dumps(payload, sort_keys=True),
    )


def encode(state_value, actions):
    return encode_h21_macro_decision(
        state_value, actions, public_schema_id=EMULATOR_PUBLIC_SCHEMA_V1,
    )


@pytest.mark.parametrize(("context", "choice_kind"), [
    ("reward", "choose_relic_deck_card"),
    ("shop", "choose_shop_relic_deck_card"),
    ("event", "choose_event_deck_card"),
])
def test_pending_deck_candidates_are_owned_grounded_and_finishable(
    context, choice_kind,
):
    observation = state()
    pending = {
        "kind": 1, "remaining_selections": 1,
        "minimum_selections_remaining": 0,
        "source_relic_id": "relic.decks",
        "candidate_card_instance_ids": [101, 102, 104],
    }
    if context == "event":
        pending.update({"choice_id": "event-choice",
                        "upgrade_transformed_cards": False})
        observation[context] = {
            "event_id": "native.event", "pending_deck_choice": pending,
        }
    else:
        observation[context] = {"pending_deck_choice": pending}
    actions = [
        action(choice_kind, CardInstanceId=101),
        action(choice_kind, CardInstanceId=102),
        action(choice_kind, CardInstanceId=104),
        action("finish_relic_deck_choice"),
    ]
    encoded = encode(observation, actions)
    left, duplicate, upgraded, finish = encoded.actions
    assert left == duplicate  # Publicly equivalent copies share a card class.
    assert left != upgraded
    assert left.source >= 0 and left.target >= 0
    assert encoded.objects[left.source].kind == KINDS.index("event_choice")
    assert encoded.objects[left.target].zone == ZONES.index("deck")
    assert finish.target == left.source
    assert finish.kind != left.kind
    assert any(e.relation_type == GROUP_OPTION
               and e.target == left.target for e in encoded.relations)
    assert sum(1 for obj in encoded.objects
               if obj.kind == KINDS.index("event_choice")
               and obj.owner == left.source) == 2  # two distinct card classes

    changed_pool = copy.deepcopy(observation)
    changed_pool[context]["pending_deck_choice"][
        "candidate_card_instance_ids"] = [101, 104]
    assert encode(changed_pool, actions[:1]) != encode(observation, actions[:1])
    with pytest.raises(ValueError, match="not a public candidate"):
        encode(changed_pool, [actions[1]])

    missing = copy.deepcopy(observation)
    missing[context]["pending_deck_choice"][
        "candidate_card_instance_ids"] = [101, 999]
    with pytest.raises(ValueError, match="Unbound public deck card"):
        encode(missing, [actions[0]])

    renamed = copy.deepcopy(observation)
    for deck_card in renamed["deck"]:
        deck_card["instance_id"] += 1000
    renamed[context]["pending_deck_choice"][
        "candidate_card_instance_ids"] = [1101, 1102, 1104]
    renamed_actions = [
        action(choice_kind, CardInstanceId=1101),
        action(choice_kind, CardInstanceId=1102),
        action(choice_kind, CardInstanceId=1104),
        action("finish_relic_deck_choice"),
    ]
    assert encode(renamed, renamed_actions) == encoded

    stricter = copy.deepcopy(observation)
    stricter[context]["pending_deck_choice"][
        "minimum_selections_remaining"] = 1
    assert encode(stricter, actions[:1]) != encode(observation, actions[:1])


def test_pending_choice_candidate_references_fail_closed():
    observation = state()
    observation["reward"] = {
        "pending_deck_choice": {
            "kind": 1, "source_relic_id": "relic.a",
            "remaining_selections": 1,
            "candidate_card_instance_ids": [101, 101],
        },
    }
    with pytest.raises(ValueError, match="Duplicate pending"):
        encode(observation, [action("choose_relic_deck_card", CardInstanceId=101)])
    observation["reward"]["pending_deck_choice"]["candidate_card_instance_ids"] = [101]
    with pytest.raises(ValueError, match="not a public candidate"):
        encode(observation, [action("choose_relic_deck_card", CardInstanceId=103)])
    del observation["reward"]["pending_deck_choice"]
    with pytest.raises(ValueError, match="no unique public context"):
        encode(observation, [action("finish_relic_deck_choice")])


def test_reroll_eligibility_is_bound_to_actual_public_group():
    observation = state()
    observation["reward"] = {
        "pending_card_groups": [
            {"group_index": 0, "card_options": ["proto.silent.strike"]},
            {"group_index": 1, "card_options": ["proto.silent.defend"]},
        ],
        "available_card_rerolls": [{"group_index": 1}],
    }
    allowed = encode(observation, [action("reroll_reward_card", GroupIndex=1)])
    assert allowed.actions[0].target >= 0
    with pytest.raises(ValueError, match="not a public available group"):
        encode(observation, [action("reroll_reward_card", GroupIndex=0)])


def test_public_event_potion_choice_id_changes_offered_potion_features():
    observation = state()
    observation["event"] = {
        "event_id": "event.potion",
        "pending_potion_replacement": {
            "choice_id": "red", "potion_id": "proto.potion.healing",
            "candidate_slots": [0],
        },
    }
    action0 = [action("take_event_potion", Slot=0)]
    first = encode(observation, action0)
    second_state = copy.deepcopy(observation)
    second_state["event"]["pending_potion_replacement"]["choice_id"] = "blue"
    assert encode(second_state, action0) != first



def test_public_reroll_eligibility_changes_other_reward_actions():
    observation = state()
    observation["reward"] = {
        "pending_card_groups": [
            {"group_index": 0, "card_options": ["proto.silent.strike"]},
        ],
        "available_card_rerolls": [],
    }
    actions = [action("take_reward_card_group", GroupIndex=0, Index=0)]
    no_reroll = encode(observation, actions)
    changed = copy.deepcopy(observation)
    changed["reward"]["available_card_rerolls"] = [{"group_index": 0}]
    available = encode(changed, actions)
    assert no_reroll != available
    assert no_reroll.actions == available.actions


def test_public_combat_run_context_is_retained_on_v3_tokens():
    initial = state()
    initial["character_id"] = "silent"
    initial["act_one_region"] = "Overgrowth"
    initial["ruleset_id"] = "v111"
    initial["combat"] = {
        "turn": 1, "energy": 3, "player_block": 0,
        "draw_pile_count": 0,
        "hand": [], "discard_pile": [], "exhaust_pile": [],
        "play_pile": [], "enemies": [], "player_powers": [],
        "history": {"cards_played_this_turn": 0},
    }
    base = encode(initial, [action("end_turn")])
    for field, value in (
        ("character_id", "other_character"),
        ("act_one_region", "Underdocks"),
        ("ruleset_id", "v112"),
        ("act", 2),
    ):
        changed = copy.deepcopy(initial)
        changed[field] = value
        assert encode(changed, [action("end_turn")]) != base, field


def test_event_queue_and_deck_choice_future_affordances_are_distinct():
    observation = state()
    observation["event"] = {
        "event_id": "native.quest",
        "queued_deck_choices": [
            {"kind": 2, "requested_selections": 2, "source_relic_id": "relic.a"},
        ],
        "queued_potion_ids": ["proto.potion.healing"],
    }
    menu = [action("continue_event")]
    encoded = encode(observation, menu)
    changed = copy.deepcopy(observation)
    changed["event"]["queued_deck_choices"][0]["requested_selections"] = 1
    assert encode(changed, menu) != encoded



def test_top_level_public_event_id_is_an_encoded_cross_phase_attribute():
    initial = state()
    initial["event_id"] = "native.chapter_event_a"
    a = encode(initial, [action("start_run")])
    later = copy.deepcopy(initial)
    later["event_id"] = "native.chapter_event_b"
    b = encode(later, [action("start_run")])
    assert a != b
