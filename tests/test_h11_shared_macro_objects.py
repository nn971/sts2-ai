"""H11 macro codec: legal options must carry public content semantics."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from sts2_ai.models.shared_game_objects import (
    SHARED_OBJECT_SCHEMA, encode_public_decision,
)
from sts2_ai.models.shared_transformer import SharedObjectTransformer, H7Config, collate_h7


def legal(*items):
    return [SimpleNamespace(kind=k, payload_json=json.dumps(p)) for k, p in items]


def test_shop_and_event_actions_distinguish_visible_options():
    shop = {"hp": 40, "gold": 180, "shop": {"card_offers": [
        {"offer_id": 41, "item_id": "proto.card.heel_hook", "price": 75},
        {"offer_id": 42, "item_id": "proto.card.haze", "price": 40},
    ]}}
    options = legal(("buy_card", {"OfferId": 41}), ("buy_card", {"OfferId": 42}))
    encoded = encode_public_decision(shop, options)
    assert encoded.schema == SHARED_OBJECT_SCHEMA and encoded.phase == "macro"
    assert encoded.actions[0] != encoded.actions[1]
    assert encoded.actions[0].target != encoded.actions[1].target
    # Ephemeral offer IDs only join within a frame.
    swapped = copy.deepcopy(shop)
    swapped["shop"]["card_offers"][0]["offer_id"] = 101
    swapped["shop"]["card_offers"][1]["offer_id"] = 102
    other = encode_public_decision(swapped, legal(
        ("buy_card", {"OfferId": 101}), ("buy_card", {"OfferId": 102}),
    ))
    assert encoded == other

    event = {"hp": 40, "event_id": "luminous_choir",
             "event": {"event_id": "luminous_choir", "visible_choice_values": [
                 {"choice_id": "remove", "gold_delta": -25},
                 {"choice_id": "upgrade", "gold_delta": -10},
             ]}}
    choices = encode_public_decision(event, legal(
        ("event_choice", {"ChoiceId": "remove"}),
        ("event_choice", {"ChoiceId": "upgrade"}),
    ))
    assert choices.actions[0].semantic != choices.actions[1].semantic
    assert choices.actions[0].target != choices.actions[1].target


def test_map_options_carry_public_topology_and_invariant_node_ids():
    state = {"floor": 6, "map": [
        {"node_id": "A", "room_type": "normal", "floor": 7,
         "next_node_ids": ["B"]},
        {"node_id": "C", "room_type": "normal", "floor": 7,
         "next_node_ids": ["D"]},
        {"node_id": "B", "room_type": "elite", "floor": 8,
         "next_node_ids": []},
        {"node_id": "D", "room_type": "rest", "floor": 8,
         "next_node_ids": []},
    ]}
    menu = legal(("choose_map_node", {"NodeId": "A"}),
                 ("choose_map_node", {"NodeId": "C"}))
    first = encode_public_decision(state, menu)
    assert first.actions[0].target != first.actions[1].target
    assert first.objects[first.actions[0].target].identity == first.objects[
        first.actions[1].target].identity
    assert {x.owner for x in first.objects if x.owner >= 0} == {
        first.actions[0].target, first.actions[1].target
    }
    renamed = copy.deepcopy(state)
    for node in renamed["map"]:
        node["node_id"] = {"A": "z9", "C": "z8", "B": "z7", "D": "z6"}[node["node_id"]]
        node["next_node_ids"] = [
            {"A": "z9", "C": "z8", "B": "z7", "D": "z6"}[x]
            for x in node["next_node_ids"]
        ]
    second = encode_public_decision(renamed, legal(
        ("choose_map_node", {"NodeId": "z9"}),
        ("choose_map_node", {"NodeId": "z8"}),
    ))
    assert first == second


def test_reward_and_deck_operation_are_encoded_and_unresolved_choices_fail():
    state = {"hp": 29, "deck": [
        {"instance_id": 21, "card_id": "strike", "upgrade_level": 0},
        {"instance_id": 22, "card_id": "haze", "upgrade_level": 0},
    ], "event": {"pending_deck_choice": {"kind": "remove"}}}
    selected = encode_public_decision(state, legal(
        ("choose_event_deck_card", {"CardInstanceId": 21}),
        ("choose_event_deck_card", {"CardInstanceId": 22}),
    ))
    assert selected.actions[0].source != selected.actions[1].source
    assert selected.actions[0].semantic == selected.actions[1].semantic
    upgrade = copy.deepcopy(state)
    upgrade["event"]["pending_deck_choice"]["kind"] = "upgrade"
    upgraded = encode_public_decision(upgrade, legal(
        ("choose_event_deck_card", {"CardInstanceId": 21}),
        ("choose_event_deck_card", {"CardInstanceId": 22}),
    ))
    assert selected.actions[0].semantic != upgraded.actions[0].semantic

    reward = {"reward": {"card_options": ["haze", "heel_hook"]}}
    options = encode_public_decision(reward, legal(
        ("take_reward_card", {"Index": 0}),
        ("take_reward_card", {"Index": 1}),
    ))
    assert options.actions[0].target != options.actions[1].target
    with pytest.raises(ValueError, match="identical public tokens"):
        encode_public_decision({}, legal(
            ("mystery_choice", {"SlotId": 13}),
            ("mystery_choice", {"SlotId": 14}),
        ))
    with pytest.raises(ValueError, match="visible"):
        encode_public_decision(reward, legal(("take_reward_card", {"Index": 99})))


def test_unmodified_h7_combat_checkpoint_architecture_accepts_macro_batch():
    torch = pytest.importorskip("torch")
    from sts2_ai.models.shared_game_objects import SHARED_OBJECT_SCHEMA as schema
    action = encode_public_decision({"event": {"event_id": "ev"}}, legal(
        ("event_choice", {"ChoiceId": "left"}),
        ("event_choice", {"ChoiceId": "right"}),
    ))
    assert action.schema == schema
    network = SharedObjectTransformer(H7Config(
        width=64, layers=2, heads=4, feedforward=128,
    )).eval()
    with torch.no_grad():
        logits, values = network(collate_h7([action]))
    assert logits.shape == (1, 2)
    assert values.shape == (1,)
    assert not torch.equal(logits[0, 0], logits[0, 1])
