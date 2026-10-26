"""H6-C public macro-object attention: parity, invariance and frozen combat."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.macro_objects import (
    MACRO_ATTENTION_FORMAT, macro_action_target, macro_object_vectors,
)
from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.phase_split_selfplay import _forward_decision, train_phase_split
from sts2_ai.training.selfplay import PublicDecision
from test_phase_split_selfplay import TwoPhaseToy, small_model


def sample():
    state = {
        "act": 1, "floor": 7, "hp": 36, "max_hp": 70, "gold": 93,
        "deck": [
            {"card_id": "silent.strike", "instance_id": 17, "upgrades": 0},
            {"card_id": "silent.strike", "instance_id": 19, "upgrades": 0},
            {"card_id": "silent.poison", "instance_id": 23, "upgrades": 1},
        ],
        "relics": ["kunai", "bag_of_preparation"],
        "potions": [{"potion_id": "strength_potion"}],
        "map": [
            {"node_id": "left", "room_type": "combat", "floor": 8,
             "next_node_ids": ["rest"]},
            {"node_id": "right", "room_type": "combat", "floor": 8,
             "next_node_ids": ["elite"]},
            {"node_id": "rest", "room_type": "rest", "floor": 9,
             "next_node_ids": []},
            {"node_id": "elite", "room_type": "elite", "floor": 9,
             "next_node_ids": []},
        ],
        "reward": {"card_options": ["silent.poison", "silent.backflip"]},
    }
    actions = (
        LegalAction("left", "choose_map_node", json.dumps({"NodeId": "left"})),
        LegalAction("right", "choose_map_node", json.dumps({"NodeId": "right"})),
        LegalAction("poison", "take_reward_card", json.dumps({"Index": 0})),
        LegalAction("backflip", "take_reward_card", json.dumps({"Index": 1})),
        LegalAction("upgrade", "rest_upgrade", json.dumps({"CardInstanceId": 23})),
    )
    return state, actions


def public(state, actions):
    return PublicDecision(
        Observation("prototype-fair-v0", json.dumps(state), "public-test"),
        actions, 0, "strategy",
    )


def test_map_targets_use_future_topology_without_raw_ids():
    state, actions = sample()
    tokens = macro_object_vectors(state, actions, 512)
    assert tokens["map:left"] != tokens["map:right"]
    assert macro_action_target(state, actions[0], tokens) == "map:left"
    assert macro_action_target(state, actions[1], tokens) == "map:right"
    assert macro_action_target(state, actions[2], tokens) == "reward:0"
    assert macro_action_target(state, actions[4], tokens) == "deck:silent.poison:u1"
    # Reorder the visible lists: the selected candidate's representation
    # must not depend on the traversal order of JSON arrays.
    altered = copy.deepcopy(state)
    altered["map"].reverse()
    altered["deck"].reverse()
    assert macro_object_vectors(altered, actions, 512) == tokens


def test_unknown_map_id_fails_closed():
    state, actions = sample()
    missing = LegalAction("absent", "choose_map_node", '{"NodeId":"not-visible"}')
    objects = macro_object_vectors(state, (*actions, missing), 128)
    assert macro_action_target(state, missing, objects) is None


def test_macro_attention_single_batched_and_portable_parity():
    torch = pytest.importorskip("torch")
    from sts2_ai.models.neural import NeuralPolicyValueModel
    from sts2_ai.training.enemy_attention import add_attention_parameters
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.neural import _export, _new_params
    from sts2_ai.training.phase_split_ppo_batch import (
        collate, encode_decision, forward_batch,
    )

    torch.manual_seed(31)
    dimension, hidden = 128, 16
    params = _new_params(dimension, hidden, torch)
    add_instance_parameters(params, dimension, hidden, torch)
    add_attention_parameters(params, hidden, torch)
    # Nonzero residual ensures QKV/O and the relational attention are tested.
    with torch.no_grad():
        params["attn_gate"].fill_(0.07)
    state, actions = sample()
    decisions = [
        public(state, actions),
        public({**state, "hp": 20}, actions[:2]),
    ]
    old = [
        _forward_decision(
            dec, params, dimension, torch, strategy_encoding="macro_attention",
        )
        for dec in decisions
    ]
    encoded = [
        encode_decision(dec, dimension, "enemy_instances", torch,
                        strategy_encoding="macro_attention")
        for dec in decisions
    ]
    batch = collate(encoded, torch)
    values, logits = forward_batch(batch, params, torch)
    assert batch.enemies.shape[:2] == (2, len(encoded[0].enemies))
    assert (logits[1, 2:] < -1e8).all()
    for i, (old_v, old_logits) in enumerate(old):
        torch.testing.assert_close(values[i], old_v, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(
            logits[i, :len(old_logits)], old_logits, atol=3e-5, rtol=3e-5,
        )

    old_loss = sum(v + logit.sum() for v, logit in old)
    new_loss = values.sum() + sum(logits[i, :len(dec.legal_actions)].sum()
                                  for i, dec in enumerate(decisions))
    old_grads = torch.autograd.grad(old_loss, tuple(params.values()), retain_graph=True)
    new_grads = torch.autograd.grad(new_loss, tuple(params.values()))
    for one, batched in zip(old_grads, new_grads, strict=True):
        torch.testing.assert_close(one, batched, atol=4e-5, rtol=4e-4)

    portable = _export(
        params, dimension, hidden, model_id="h6c-parity",
        value_head_trained=True, format_id=MACRO_ATTENTION_FORMAT,
    )
    assert isinstance(portable, NeuralPolicyValueModel)
    restored = NeuralPolicyValueModel.from_dict(portable.to_dict())
    evaluation = restored.evaluate(decisions[0].observation, actions)
    assert evaluation.value == pytest.approx(old[0][0].detach().item(), abs=4e-5)
    assert evaluation.action_logits == pytest.approx(
        old[0][1].detach().tolist(), abs=4e-5,
    )


def test_node_renaming_and_private_fields_do_not_change_action_logits():
    torch = pytest.importorskip("torch")
    from sts2_ai.training.neural import _export, _new_params
    from sts2_ai.training.instance_combat import add_instance_parameters
    from sts2_ai.training.enemy_attention import add_attention_parameters

    state, actions = sample()
    torch.manual_seed(17)
    params = _new_params(128, 16, torch)
    add_instance_parameters(params, 128, 16, torch)
    add_attention_parameters(params, 16, torch)
    with torch.no_grad():
        params["attn_gate"].fill_(0.1)
    model = _export(params, 128, 16, model_id="h6c-identity",
                    value_head_trained=True, format_id=MACRO_ATTENTION_FORMAT)
    original = model.evaluate(public(state, actions).observation, actions)
    rename = {name: f"changed-{name}" for name in ("left", "right", "rest", "elite")}
    alternative = copy.deepcopy(state)
    for node in alternative["map"]:
        node["node_id"] = rename[node["node_id"]]
        node["next_node_ids"] = [rename[v] for v in node["next_node_ids"]]
    # The old identifiers have no remaining role; the observable graph is
    # isomorphic and actions select the corresponding public nodes.
    changed = tuple(
        LegalAction(a.action_id, a.kind, json.dumps({
            "NodeId": rename[json.loads(a.payload_json)["NodeId"]],
        })) if a.kind == "choose_map_node" else a
        for a in actions
    )
    after = model.evaluate(public(alternative, changed).observation, changed)
    assert after.value == pytest.approx(original.value, abs=1e-6)
    assert after.action_logits == pytest.approx(original.action_logits, abs=1e-6)


def test_h6c_frozen_combat_resume_and_versioned_export(tmp_path: Path):
    pytest.importorskip("torch")
    backbone = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT),
        small_model(TACTICAL_FORMAT, value_bias=0.11),
        "h6c-toy",
    )
    warm = tmp_path / "backbone.json"
    backbone.save(warm)
    opts = dict(
        backend=TwoPhaseToy(), episodes_per_round=6,
        dimension=32, hidden=4, max_decisions=8, workers=1,
        optimizer_method="ppo", ppo_backend="batched",
        hp_monotonic_weight=0, tactical_state_encoding="legacy",
        combat_objective="continuation", warm_start=warm,
        freeze_combat=True, strategy_encoding="macro_attention",
        temperature_start=1.0, temperature_end=1.0,
        checkpoint=tmp_path / "h6c.pt",
    )
    trained, rows = train_phase_split(**opts, rounds=1)
    assert rows[0].strategic_decisions > 0
    assert trained.combat.to_dict() == backbone.combat.to_dict()
    assert trained.strategy.format_id == MACRO_ATTENTION_FORMAT
    saved = tmp_path / "h6c.json"
    trained.save(saved)
    assert PhaseSplitNeuralModel.load(saved).to_dict() == trained.to_dict()
    continued, history = train_phase_split(**opts, rounds=2, resume=True)
    assert len(history) == 2
    assert continued.combat.to_dict() == backbone.combat.to_dict()
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(**{**opts, "strategy_encoding": "legacy"},
                          rounds=2, resume=True)
