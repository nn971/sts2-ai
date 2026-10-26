"""H6-D versioned public semantic macro actions: shop/event separability."""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.macro_objects import (
    MACRO_ATTENTION_FORMAT,
    MACRO_SEMANTIC_FORMAT,
    macro_attention_action_features,
    macro_semantic_action_features,
)
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.selfplay import PublicDecision


def choices():
    frame = {
        "act": 1, "floor": 5, "hp": 51, "max_hp": 70, "gold": 280, "deck": [],
        "shop": {
            "card_offers": [
                {"offer_id": 1, "item_id": "silent.backflip", "price": 85},
                {"offer_id": 2, "item_id": "silent.noxious_fumes", "price": 110},
            ],
            "potion_offers": [
                {"offer_id": 3, "item_id": "proto.potion.strength", "price": 50},
                {"offer_id": 4, "item_id": "proto.potion.fire", "price": 80},
            ],
        },
        "event_id": "proto.event.test",
        "event": {
            "event_id": "proto.event.test",
            "visible_choice_values": [
                {"choice_id": "donate", "gold_cost": 60, "heal": 22},
                {"choice_id": "steal", "gold_gain": 55, "hp_loss": 9},
            ],
        },
        "reward": {"relic_option": "proto.relic.test"},
    }
    actions = (
        LegalAction("buy1", "buy_card", '{"OfferId":1}'),
        LegalAction("buy2", "buy_card", '{"OfferId":2}'),
        LegalAction("potion1", "buy_potion", '{"OfferId":3}'),
        LegalAction("potion2", "buy_potion", '{"OfferId":4}'),
        LegalAction("event1", "event_choice", '{"ChoiceId":"donate"}'),
        LegalAction("event2", "event_choice", '{"ChoiceId":"steal"}'),
        LegalAction("take", "take_reward_relic", "{}"),
        LegalAction("skip", "skip_reward_relic", "{}"),
    )
    return frame, actions


def test_v17_regression_and_v18_public_identity():
    frame, actions = choices()
    legacy = [macro_attention_action_features(
        frame, a.kind, a.payload_json, 512,
    ) for a in actions]
    assert legacy[0] == legacy[1]
    assert legacy[2] == legacy[3]
    assert legacy[4] == legacy[5]
    rich = [macro_semantic_action_features(
        frame, a.kind, a.payload_json, 512,
    ) for a in actions]
    assert rich[0] != rich[1]
    assert rich[2] != rich[3]
    assert rich[4] != rich[5]
    assert rich[6] != rich[7]
    # Offer IDs are ephemeral, and should not alter semantic features if the
    # same publicly visible goods are reassigned different local offer IDs.
    switched = json.loads(json.dumps(frame))
    for offer in switched["shop"]["card_offers"]:
        offer["offer_id"] += 40
    card = LegalAction("changed", "buy_card", '{"OfferId":41}')
    assert macro_semantic_action_features(
        switched, card.kind, card.payload_json, 512,
    ) == rich[0]


def test_v18_batched_single_and_portable_logits():
    torch = pytest.importorskip("torch")
    from sts2_ai.training.enemy_attention import add_attention_parameters
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.macro_attention import macro_attention_forward
    from sts2_ai.training.neural import _export, _new_params
    from sts2_ai.training.phase_split_ppo_batch import collate, encode_decision, forward_batch

    frame, actions = choices()
    torch.manual_seed(38)
    params = _new_params(128, 16, torch)
    add_instance_parameters(params, 128, 16, torch)
    add_attention_parameters(params, 16, torch)
    with torch.no_grad():
        params["attn_gate"].fill_(0.03)
    value, logits = macro_attention_forward(
        frame, actions, params, 128, torch, semantic_actions=True,
    )
    pub = PublicDecision(
        Observation("prototype-fair-v0", json.dumps(frame), "public"),
        actions, 0, "strategy",
    )
    encoded = encode_decision(pub, 128, "enemy_instances", torch,
                              strategy_encoding="macro_semantic")
    values, batch_logits = forward_batch(collate([encoded], torch), params, torch)
    torch.testing.assert_close(values[0], value, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batch_logits[0, :len(actions)], logits,
                               rtol=1e-5, atol=1e-5)
    portable = _export(
        params, 128, 16, model_id="v18-semantic", value_head_trained=True,
        format_id=MACRO_SEMANTIC_FORMAT,
    )
    restored = NeuralPolicyValueModel.from_dict(portable.to_dict())
    estimate = restored.evaluate(pub.observation, actions)
    assert estimate.value == pytest.approx(value.detach().item(), abs=3e-5)
    assert estimate.action_logits == pytest.approx(logits.detach().tolist(), abs=3e-5)
    assert estimate.action_logits[0] != estimate.action_logits[1]
    assert restored.format_id == MACRO_SEMANTIC_FORMAT
    legacy = _export(params, 128, 16, model_id="v17-backward",
                     value_head_trained=True, format_id=MACRO_ATTENTION_FORMAT)
    assert NeuralPolicyValueModel.from_dict(legacy.to_dict()).format_id == MACRO_ATTENTION_FORMAT


def test_v18_native_shop_offer_schema_and_sparse_potion_slots():
    # Fields follow pinned PrototypeAiObservation/PrototypeAiShop/ShopOffer.
    state = {
        "gold": 300,
        "potions": [
            {"slot": 0, "potion_id": "proto.potion.poison", "state": {}},
            {"slot": 2, "potion_id": "proto.potion.fire", "state": {}},
        ],
        "shop": {
            "card_offers": [
                {"offer_id": 10, "item_id": "silent.backflip", "price": 80,
                 "base_price": 100, "on_sale": True, "sold": False},
                {"offer_id": 11, "item_id": "silent.footwork", "price": 95,
                 "base_price": 95, "on_sale": False, "sold": False},
            ],
            "potion_offer": {
                "offer_id": 12, "item_id": "proto.potion.strength",
                "price": 40, "sold": False,
            },
            "potion_offers": [
                {"offer_id": 12, "item_id": "proto.potion.strength",
                 "price": 40, "sold": False},
            ],
            "relic_offers": [
                {"offer_id": 13, "item_id": "proto.relic.nunchaku",
                 "price": 150, "sold": False},
                {"offer_id": 14, "item_id": "proto.relic.bag_of_preparation",
                 "price": 170, "sold": False},
            ],
            "card_offer_previews": [
                {"offer_id": 10, "card": {
                    "card_id": "silent.backflip", "upgrade_level": 1,
                    "enchantment": None,
                }},
            ],
        },
    }

    def encoded(kind, payload):
        return macro_semantic_action_features(state, kind, json.dumps(payload), 512)

    assert encoded("buy_card", {"OfferId": 10}) != encoded(
        "buy_card", {"OfferId": 11},
    )
    assert encoded("buy_relic", {"OfferId": 13}) != encoded(
        "buy_relic", {"OfferId": 14},
    )
    assert encoded("replace_shop_potion", {"OfferId": 12, "Slot": 0}) != encoded(
        "replace_shop_potion", {"OfferId": 12, "Slot": 2},
    )
    before = encoded("replace_shop_potion", {"OfferId": 12, "Slot": 2})
    # The public potions collection is packed, but native Slot is a slot number.
    state["potions"].reverse()
    assert encoded("replace_shop_potion", {"OfferId": 12, "Slot": 2}) == before
    # Different transient offer IDs leave semantic encoding unchanged.
    before = encoded("buy_card", {"OfferId": 10})
    state["shop"]["card_offers"][0]["offer_id"] = 88
    state["shop"]["card_offer_previews"][0]["offer_id"] = 88
    assert encoded("buy_card", {"OfferId": 88}) == before


def test_v18_native_visible_event_choices_and_potion_replacement():
    state = {
        "event_id": "proto.native.event.test",
        "event": {
            "event_id": "proto.native.event.test",
            "offered_choice_ids": ["trudge", "rest"],
            "visible_choice_values": [
                {"choice_id": "trudge", "gold_gain": 83, "hp_loss": 8},
                {"choice_id": "rest", "heal": 21},
            ],
            "pending_potion_replacement": {
                "choice_id": "take", "potion_id": "proto.potion.energy",
                "candidate_slots": [0, 2],
            },
        },
        "potions": [
            {"slot": 0, "potion_id": "proto.potion.poison", "state": {}},
            {"slot": 2, "potion_id": "proto.potion.fire", "state": {}},
        ],
    }

    def encoded(kind, payload):
        return macro_semantic_action_features(state, kind, json.dumps(payload), 512)

    assert encoded("event_choice", {"ChoiceId": "trudge"}) != encoded(
        "event_choice", {"ChoiceId": "rest"},
    )
    assert encoded("replace_event_potion", {"Slot": 0}) != encoded(
        "replace_event_potion", {"Slot": 2},
    )
    # IDs link visible choices; the network receives the chosen effect data.
    old = encoded("event_choice", {"ChoiceId": "rest"})
    state["event"]["visible_choice_values"][1]["heal"] = 45
    assert encoded("event_choice", {"ChoiceId": "rest"}) != old
