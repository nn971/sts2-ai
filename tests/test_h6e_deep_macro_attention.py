"""H6-E: deep semantic macro attention, pointer resolution and portable parity."""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.macro_objects import (
    MACRO_DEEP_FORMAT, macro_deep_action_target, macro_deep_object_vectors,
)
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.selfplay import PublicDecision


def _frame():
    frame = {
        "act": 1, "floor": 5, "hp": 42, "max_hp": 70, "gold": 200,
        "deck": [{"instance_id": 10, "card_id": "silent.backflip",
                  "upgrade_level": 0}],
        "relics": [{"relic_id": "proto.relic.anchor"}], "potions": [],
        "shop": {"card_offers": [
            {"offer_id": 11, "item_id": "silent.backflip", "price": 80},
            {"offer_id": 12, "item_id": "silent.footwork", "price": 95},
        ]},
        "event": {
            "event_id": "proto.event.demo",
            "visible_choice_values": [
                {"choice_id": "pay", "gold_cost": 40, "heal": 20},
                {"choice_id": "fight", "hp_loss": 9, "gold_gain": 60},
            ],
        },
        "reward": {"card_options": ["silent.catalyst", "silent.accuracy"]},
    }
    actions = (
        LegalAction("s1", "buy_card", '{"OfferId":11}'),
        LegalAction("s2", "buy_card", '{"OfferId":12}'),
        LegalAction("e1", "event_choice", '{"ChoiceId":"pay"}'),
        LegalAction("e2", "event_choice", '{"ChoiceId":"fight"}'),
        LegalAction("r1", "take_reward_card", '{"Index":0}'),
        LegalAction("r2", "take_reward_card", '{"Index":1}'),
        LegalAction("skip", "skip_reward_card", '{}'),
    )
    return frame, actions


def test_deep_shop_event_pointers_and_stable_offer_renaming():
    frame, actions = _frame()
    objects = macro_deep_object_vectors(frame, actions, 128)
    keys = [macro_deep_action_target(frame, a, objects) for a in actions]
    assert keys[0] != keys[1] and keys[0] in objects and keys[1] in objects
    assert keys[2] != keys[3] and keys[2] in objects and keys[3] in objects
    assert keys[4] != keys[5] and keys[4] in objects and keys[5] in objects
    original_features = objects[keys[0]]
    frame["shop"]["card_offers"][0]["offer_id"] = 91
    replaced = LegalAction("rename", "buy_card", '{"OfferId":91}')
    renamed = macro_deep_object_vectors(frame, (replaced,), 128)
    assert renamed[macro_deep_action_target(frame, replaced, renamed)] == original_features


def test_deep_torch_batch_portable_parity_and_gradients():
    torch = pytest.importorskip("torch")
    from sts2_ai.training.enemy_attention import (
        add_attention_parameters, add_deep_macro_parameters,
    )
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.macro_attention import macro_attention_forward
    from sts2_ai.training.neural import _export, _new_params
    from sts2_ai.training.phase_split_ppo_batch import (
        collate, encode_decision, forward_batch,
    )

    frame, actions = _frame()
    torch.manual_seed(2026)
    params = _new_params(128, 32, torch)
    add_instance_parameters(params, 128, 32, torch)
    add_attention_parameters(params, 32, torch)
    add_deep_macro_parameters(params, 32, torch)
    with torch.no_grad():
        params["attn_gate"].fill_(0.02)
        params["attn2_gate"].fill_(0.04)
        params["macro_ffn_gate"].fill_(0.03)
    value, logits = macro_attention_forward(
        frame, actions, params, 128, torch, deep_actions=True,
    )
    pub = PublicDecision(
        Observation("prototype-fair-v0", json.dumps(frame), "public"),
        actions, 0, "strategy",
    )
    encoded = encode_decision(
        pub, 128, "enemy_instances", torch, strategy_encoding="macro_deep",
    )
    values, batch_logits = forward_batch(collate([encoded], torch), params, torch)
    torch.testing.assert_close(values[0], value, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        batch_logits[0, :len(actions)], logits, atol=2e-5, rtol=2e-5,
    )
    portable = _export(
        params, 128, 32, model_id="h6e-deep", value_head_trained=True,
        format_id=MACRO_DEEP_FORMAT,
    )
    restored = NeuralPolicyValueModel.from_dict(portable.to_dict())
    estimate = restored.evaluate(pub.observation, actions)
    assert estimate.value == pytest.approx(value.detach().item(), abs=4e-5)
    assert estimate.action_logits == pytest.approx(logits.detach().tolist(), abs=4e-5)
    assert restored.format_id == MACRO_DEEP_FORMAT
    logits.sum().backward()
    assert params["attn2_gate"].grad is not None
    assert params["macro_ffn_gate"].grad is not None
    assert params["attn2_gate"].grad.abs().sum().item() > 0
    assert params["macro_ffn_gate"].grad.abs().sum().item() > 0


def test_deep_has_more_parameters_than_v18():
    torch = pytest.importorskip("torch")
    from sts2_ai.training.enemy_attention import (
        add_attention_parameters, add_deep_macro_parameters,
    )
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.neural import _new_params

    params = _new_params(128, 32, torch)
    add_instance_parameters(params, 128, 32, torch)
    add_attention_parameters(params, 32, torch)
    baseline_count = sum(x.numel() for x in params.values())
    add_deep_macro_parameters(params, 32, torch)
    deep_count = sum(x.numel() for x in params.values())
    assert deep_count > baseline_count * 1.5
