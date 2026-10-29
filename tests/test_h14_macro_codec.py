"""H14 public semantic macro features and audited persistent counters."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from sts2_ai.models.shared_game_objects import (
    NUMERIC_FIELDS,
    SHARED_OBJECT_SCHEMA,
    _bucket,
    encode_public_decision,
)


def legal(*entries):
    return [
        SimpleNamespace(kind=kind, payload_json=json.dumps(payload))
        for kind, payload in entries
    ]


def _counter_tokens(encoded):
    return [
        (i, token) for i, token in enumerate(encoded.objects)
        if token.identity in (
            _bucket("persistent:proto.relic.pen_nib:AttacksPlayed", 4096),
            _bucket("persistent:proto.relic.pendulum:TurnsSeen", 4096),
        )
    ]


def _state(counter=9):
    return {
        "hp": 50, "max_hp": 70, "gold": 60,
        "relics": [
            {"relic_id": "proto.relic.pen_nib",
             "state": {"AttacksPlayed": counter, "rng_cursor": "SECRET"}},
            {"relic_id": "proto.relic.pen_nib",
             "state": {"AttacksPlayed": 1}},
            {"relic_id": "proto.relic.pendulum",
             "state": {"TurnsSeen": 2}},
        ],
        "event": {"event_id": "ev", "visible_choice_values": [
            {"choice_id": "left"}, {"choice_id": "right"},
        ]},
    }


def test_persistent_counters_are_distinct_owned_visible_numeric_tokens():
    menu = legal(("event_choice", {"ChoiceId": "left"}),
                 ("event_choice", {"ChoiceId": "right"}))
    encoded = encode_public_decision(_state(), menu)
    assert encoded.schema == SHARED_OBJECT_SCHEMA
    assert encoded.phase == "macro"
    tokens = _counter_tokens(encoded)
    assert len(tokens) == 3
    assert len({encoded.objects[i].owner for i, _ in tokens}) == 3
    idx = NUMERIC_FIELDS.index("count")
    assert all(token.observed[idx] == 1 for _, token in tokens)
    wrapped = encode_public_decision(_state(0), menu)
    assert encoded != wrapped
    # The only difference is a genuinely observed residue, never the secret
    # relic serializer's other arbitrary properties.
    swapped = _state()
    swapped["relics"][0]["state"]["rng_cursor"] = "DIFFERENT"
    assert encoded == encode_public_decision(swapped, menu)
    no_counter = _state()
    no_counter["relics"][0]["state"].pop("AttacksPlayed")
    assert encode_public_decision(no_counter, menu) == wrapped


def test_counter_feature_also_changes_public_combat_encoding():
    frame = _state()
    frame["combat"] = {
        "energy": 3, "turn": 1, "draw_pile_count": 4,
        "hand": [{"instance_id": 99, "card_id": "strike"}],
        "enemies": [{"instance_id": 100, "enemy_id": "beetle", "hp": 20}],
    }
    actions = legal(("play_card", {"CardInstanceId": 99, "TargetEnemyId": 100}))
    first = encode_public_decision(frame, actions)
    changed = copy.deepcopy(frame)
    changed["relics"][0]["state"]["AttacksPlayed"] = 0
    second = encode_public_decision(changed, actions)
    assert first.phase == second.phase == "combat"
    assert first != second
    assert len(_counter_tokens(first)) == 3


def test_macro_value_retains_map_information_even_during_reward_selection():
    state = {"hp": 30, "reward": {"card_options": ["poison"]},
             "map": [
                 {"node_id": "hidden-identity-a", "floor": 6,
                  "room_type": "elite", "next_node_ids": []},
                 {"node_id": "hidden-identity-b", "floor": 6,
                  "room_type": "rest", "next_node_ids": []},
             ]}
    actions = legal(("take_reward_card", {"Index": 0}))
    encoded = encode_public_decision(state, actions)
    altered = copy.deepcopy(state)
    altered["map"][0]["room_type"] = "shop"
    assert encoded != encode_public_decision(altered, actions)
    renamed = copy.deepcopy(state)
    renamed["map"][0]["node_id"] = "unrelated"
    renamed["map"][1]["node_id"] = "random"
    assert encoded == encode_public_decision(renamed, actions)


def test_legal_shop_and_reward_actions_keep_semantic_targets():
    state = {"shop": {"card_offers": [
        {"offer_id": 7, "item_id": "slice", "price": 50},
        {"offer_id": 8, "item_id": "poison", "price": 75},
    ]}}
    encoded = encode_public_decision(state, legal(
        ("buy_card", {"OfferId": 7}), ("buy_card", {"OfferId": 8}),
    ))
    assert encoded.actions[0] != encoded.actions[1]
    assert encoded.actions[0].target != encoded.actions[1].target
    with pytest.raises(ValueError, match="visible offer"):
        encode_public_decision(state, legal(("buy_card", {"OfferId": 999})))


def test_invalid_counter_is_explicitly_rejected():
    state = _state("nine")
    with pytest.raises(ValueError, match="AttacksPlayed"):
        encode_public_decision(state, legal(("event_choice", {"ChoiceId": "left"})))
