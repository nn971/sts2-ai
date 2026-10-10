"""v13 attention: initial v8 parity, public ID equivariance, batched PPO agreement."""
from __future__ import annotations

import copy
import json
import random

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import TACTICAL_ATTENTION_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.enemy_attention import add_attention_parameters
from sts2_ai.training.instance_combat import instance_forward
from sts2_ai.training.phase_split_ppo_batch import (
    collate,
    encode_decision,
    forward_batch,
)
from sts2_ai.training.selfplay import PublicDecision
from test_enemy_instances_v8 import build_model, combat_frame, targeted


def attention_model(gate: float = 0.0) -> NeuralPolicyValueModel:
    original = build_model().to_dict()
    original["format"] = TACTICAL_ATTENTION_FORMAT
    original["model_id"] = "v13-fixture"
    hidden = original["hidden"]
    for name in ("attn_query", "attn_key", "attn_value", "attn_output"):
        original[name] = [
            [float(i == j) for j in range(hidden)] for i in range(hidden)
        ]
    original["attn_gate"] = [gate] * hidden
    return NeuralPolicyValueModel.from_dict(original)


def frame_observation(state: dict) -> Observation:
    payload = json.dumps(state, sort_keys=True)
    return Observation("prototype-fair-v0", payload, payload)


def torch_params(model: NeuralPolicyValueModel, torch) -> dict:
    raw = model.to_dict()
    return {
        name: torch.tensor(value, dtype=torch.float32, requires_grad=True)
        for name, value in raw.items()
        if name in (
            "state_weight", "state_bias", "action_weight", "action_bias",
            "policy_weight", "policy_bias", "value_weight", "value_bias",
            "enemy_weight", "enemy_bias", "enemy_context_weight",
            "enemy_target_weight",
            "attn_query", "attn_key", "attn_value", "attn_output", "attn_gate",
        )
    }


def test_initial_attention_exactly_preserves_v8_logits_and_values() -> None:
    state = combat_frame()
    actions = [targeted(1), targeted(2), targeted(3), LegalAction("end", "end_turn", "{}")]
    original = build_model().evaluate(frame_observation(state), actions)
    widened = attention_model().evaluate(frame_observation(state), actions)
    assert widened.action_logits == original.action_logits
    assert widened.value == original.value
    assert attention_model().to_dict() == NeuralPolicyValueModel.from_dict(
        attention_model().to_dict()
    ).to_dict()
    with pytest.raises(ValueError, match="Attention hidden size"):
        altered = attention_model().to_dict()
        altered["hidden"] = 5
        NeuralPolicyValueModel.from_dict(altered)
    with pytest.raises((ValueError, KeyError)):
        altered = attention_model().to_dict()
        del altered["attn_query"]
        NeuralPolicyValueModel.from_dict(altered)


def test_attention_is_equivariant_under_enemy_id_renumbering_and_order() -> None:
    state = combat_frame()
    actions = (targeted(1), targeted(2), targeted(3))
    model = attention_model(gate=0.6)
    expected = model.evaluate(frame_observation(state), actions)
    changed = copy.deepcopy(state)
    for enemy in changed["combat"]["enemies"]:
        enemy["instance_id"] += 100
    random.Random(91).shuffle(changed["combat"]["enemies"])
    reordered = model.evaluate(
        frame_observation(changed), tuple(targeted(x + 100) for x in (1, 2, 3)),
    )
    assert reordered.action_logits == pytest.approx(expected.action_logits, abs=1e-6)
    assert reordered.value == pytest.approx(expected.value, abs=1e-6)


def test_no_relational_update_for_zero_or_one_enemy() -> None:
    state = combat_frame()
    model = attention_model(gate=0.9)
    old = build_model()
    for n in (0, 1):
        copy_state = copy.deepcopy(state)
        copy_state["combat"]["enemies"] = copy_state["combat"]["enemies"][:n]
        actions = [LegalAction("end", "end_turn", "{}")]
        if n:
            actions.append(targeted(copy_state["combat"]["enemies"][0]["instance_id"]))
        a = model.evaluate(frame_observation(copy_state), actions)
        b = old.evaluate(frame_observation(copy_state), actions)
        assert a.action_logits == b.action_logits
        assert a.value == b.value


def test_batched_matches_reference_and_portable_for_mixed_enemy_counts() -> None:
    torch = pytest.importorskip("torch")
    model = attention_model(gate=0.4)
    params = torch_params(model, torch)
    states = []
    for n in (0, 1, 3):
        state = combat_frame()
        state["combat"]["enemies"] = state["combat"]["enemies"][:n]
        actions = [LegalAction("end", "end_turn", "{}")]
        actions.extend(targeted(e["instance_id"]) for e in state["combat"]["enemies"])
        states.append((state, actions))
    decisions = [
        PublicDecision(frame_observation(state), tuple(actions), 0, "combat")
        for state, actions in states
    ]
    batch = collate(
        [encode_decision(d, model.dimension, "enemy_attention", torch) for d in decisions],
        torch,
    )
    batched_value, batched_logits = forward_batch(batch, params, torch)
    for i, (state, actions) in enumerate(states):
        value, logits = instance_forward(state, actions, params, model.dimension, torch)
        portable = model.evaluate(frame_observation(state), actions)
        assert float(batched_value[i]) == pytest.approx(float(value), abs=1e-5)
        assert float(value) == pytest.approx(portable.value, abs=1e-5)
        for j, number in enumerate(portable.action_logits):
            assert float(batched_logits[i, j]) == pytest.approx(float(logits[j]), abs=1e-5)
            assert float(batched_logits[i, j]) == pytest.approx(number, abs=1e-5)
        if len(actions) < batch.mask.shape[1]:
            assert (batched_logits[i, len(actions):] < -1e8).all().item()


def test_zero_gate_receives_gradient_and_attention_can_train() -> None:
    torch = pytest.importorskip("torch")
    model = attention_model(gate=0.0)
    params = torch_params(model, torch)
    state = combat_frame()
    actions = [targeted(1), targeted(2), targeted(3)]
    value, logits = instance_forward(state, actions, params, model.dimension, torch)
    loss = value + logits[0] + logits[1]
    loss.backward()
    assert params["attn_gate"].grad is not None
    assert float(params["attn_gate"].grad.abs().sum()) > 0


def test_default_initialized_gate_is_zero_and_export_format_works() -> None:
    torch = pytest.importorskip("torch")
    params = {}
    add_attention_parameters(params, 32, torch)
    assert torch.count_nonzero(params["attn_gate"]).item() == 0
    assert params["attn_query"].shape == (32, 32)
    with pytest.raises(ValueError, match="multiple of four"):
        add_attention_parameters({}, 33, torch)
    model = attention_model()
    strategy_raw = build_model().to_dict()
    strategy_raw.pop("enemy_weight")
    strategy_raw.pop("enemy_bias")
    strategy_raw.pop("enemy_context_weight")
    strategy_raw.pop("enemy_target_weight")
    strategy_raw["format"] = "sts2-neural-policy-value-v2-semantic-action"
    strategy = NeuralPolicyValueModel.from_dict(strategy_raw)
    phase_split = PhaseSplitNeuralModel(strategy, model, "v13-test")
    assert PhaseSplitNeuralModel.from_dict(phase_split.to_dict()).to_dict() == (
        phase_split.to_dict()
    )
