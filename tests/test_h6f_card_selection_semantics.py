"""V20 visible card selection operation semantics (H6-F correctness gate)."""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.macro_objects import (
    MACRO_CARD_OPERATIONS_FORMAT,
    MACRO_DEEP_FORMAT,
    macro_card_operation_action_features,
    macro_card_operation_action_target,
    macro_card_operation_object_vectors,
    macro_card_selection_operation,
    macro_semantic_action_features,
)
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.selfplay import PublicDecision


def _state(operation="Remove", *, source="event"):
    deck = [
        {"instance_id": 101, "card_id": "proto.silent.strike", "upgrade_level": 0},
        {"instance_id": 102, "card_id": "proto.silent.backflip", "upgrade_level": 1},
    ]
    container = {
        "pending_deck_choice": {
            "kind": operation,
            "remaining_selections": 2,
            "candidate_card_instance_ids": [101, 102],
        },
    }
    if source == "event":
        container["event_id"] = "proto.native.event.luminous_choir"
    else:
        container["pending_deck_choice"]["source_relic_id"] = "proto.relic.Pomander"
    return {
        "act": 1, "floor": 6, "hp": 55, "max_hp": 70, "gold": 200,
        "deck": deck, "relics": [], "potions": [], source: container,
    }


def _actions(kind):
    return (
        LegalAction("a", kind, '{"CardInstanceId":101}'),
        LegalAction("b", kind, '{"CardInstanceId":102}'),
    )


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        ("event", "choose_event_deck_card"),
        ("shop", "choose_shop_relic_deck_card"),
        ("reward", "choose_relic_deck_card"),
    ],
)
def test_operation_affects_features_and_selected_card_has_pointer(source, kind):
    actions = _actions(kind)
    remove = _state("Remove", source=source)
    upgrade = _state("Upgrade", source=source)
    transform = _state("Transform", source=source)
    enchant = _state("Enchant", source=source)
    assert [macro_card_selection_operation(state, kind) for state in
            (remove, upgrade, transform, enchant)] == [
                "remove", "upgrade", "transform", "enchant",
            ]
    old_remove = macro_semantic_action_features(
        remove, kind, actions[0].payload_json, 512,
    )
    old_upgrade = macro_semantic_action_features(
        upgrade, kind, actions[0].payload_json, 512,
    )
    assert old_remove == old_upgrade  # historical V18/V19 action representation
    representations = [
        macro_card_operation_action_features(
            state, kind, actions[0].payload_json, 512,
        ) for state in (remove, upgrade, transform, enchant)
    ]
    assert len({tuple(sorted(vec.items())) for vec in representations}) == 4

    objects = macro_card_operation_object_vectors(remove, actions, 512)
    pointer_a = macro_card_operation_action_target(remove, actions[0], objects)
    pointer_b = macro_card_operation_action_target(remove, actions[1], objects)
    assert pointer_a in objects and pointer_b in objects
    assert pointer_a != pointer_b
    renamed = json.loads(json.dumps(remove))
    renamed["deck"][0]["instance_id"] = 900
    renamed["event" if source == "event" else source][
        "pending_deck_choice"]["candidate_card_instance_ids"][0] = 900
    selected = LegalAction("renamed", kind, '{"CardInstanceId":900}')
    renamed_objects = macro_card_operation_object_vectors(renamed, (selected,), 512)
    renamed_target = macro_card_operation_action_target(
        renamed, selected, renamed_objects,
    )
    assert renamed_objects[renamed_target] == objects[pointer_a]


@pytest.mark.parametrize("value,expected", [
    ("remove", "remove"), ("UPGRADE", "upgrade"),
    (0, "remove"), (1, "upgrade"), (2, "transform"), (3, "enchant"),
    (True, None), (4, None), ("unsupported", None),
])
def test_operation_public_enum_decode(value, expected):
    assert macro_card_selection_operation(_state(value), "choose_event_deck_card") == expected


def test_selected_card_stays_targetable_after_card_cap():
    state = _state()
    state["deck"] = [
        {"instance_id": i + 1, "card_id": f"proto.silent.card_{i}", "upgrade_level": 0}
        for i in range(54)
    ]
    action = LegalAction("last", "choose_event_deck_card", '{"CardInstanceId":54}')
    objects = macro_card_operation_object_vectors(state, (action,), 128)
    assert macro_card_operation_action_target(state, action, objects) in objects


def test_torch_batched_and_portable_operation_aware_parity():
    torch = pytest.importorskip("torch")
    from sts2_ai.training.enemy_attention import (
        add_attention_parameters,
        add_deep_macro_parameters,
    )
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.macro_attention import macro_attention_forward
    from sts2_ai.training.neural import _export, _new_params
    from sts2_ai.training.phase_split_ppo_batch import (
        collate,
        encode_decision,
        forward_batch,
    )

    torch.manual_seed(121)
    params = _new_params(128, 32, torch)
    add_instance_parameters(params, 128, 32, torch)
    add_attention_parameters(params, 32, torch)
    add_deep_macro_parameters(params, 32, torch)
    with torch.no_grad():
        params["attn_gate"].fill_(0.09)
        params["attn2_gate"].fill_(0.08)
        params["macro_ffn_gate"].fill_(0.06)
    portable = _export(
        params, 128, 32, model_id="v20-semantic-test",
        value_head_trained=True, format_id=MACRO_CARD_OPERATIONS_FORMAT,
    )
    restored = NeuralPolicyValueModel.from_dict(portable.to_dict())
    assert restored.to_dict() == portable.to_dict()
    assert restored.format_id == MACRO_CARD_OPERATIONS_FORMAT

    actions = _actions("choose_event_deck_card")
    for operation in ("Remove", "Upgrade", "Transform", "Enchant"):
        state = _state(operation)
        value, logits = macro_attention_forward(
            state, actions, params, 128, torch, card_operations=True,
        )
        observation = Observation("prototype-fair-v0", json.dumps(state), "public")
        episode_decision = PublicDecision(observation, actions, 0, "strategy")
        encoded = encode_decision(
            episode_decision, 128, "enemy_instances", torch,
            strategy_encoding="macro_card_ops",
        )
        values, batch_logits = forward_batch(collate([encoded], torch), params, torch)
        torch.testing.assert_close(values[0], value, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            batch_logits[0, :len(actions)], logits, atol=2e-5, rtol=2e-5,
        )
        prediction = restored.evaluate(observation, actions)
        assert prediction.value == pytest.approx(value.detach().item(), abs=4e-5)
        assert prediction.action_logits == pytest.approx(
            logits.detach().tolist(), abs=4e-5,
        )
    assert len({
        tuple(sorted(macro_card_operation_action_features(
            _state(op), actions[0].kind, actions[0].payload_json, 128,
        ).items()))
        for op in ("Remove", "Upgrade", "Transform", "Enchant")
    }) == 4


def test_v19_inference_stays_unchanged():
    torch = pytest.importorskip("torch")
    from sts2_ai.training.enemy_attention import (
        add_attention_parameters,
        add_deep_macro_parameters,
    )
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.neural import _export, _new_params

    weights = _new_params(128, 32, torch)
    add_instance_parameters(weights, 128, 32, torch)
    add_attention_parameters(weights, 32, torch)
    add_deep_macro_parameters(weights, 32, torch)
    model = _export(
        weights, 128, 32, model_id="v19-unchanged",
        value_head_trained=True, format_id=MACRO_DEEP_FORMAT,
    )
    serialized = model.to_dict()
    assert NeuralPolicyValueModel.from_dict(serialized).to_dict() == serialized
    assert "attn2_gate" in serialized and "macro_ffn_gate" in serialized
