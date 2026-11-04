"""H22 H21 joint-inference/training schema and checkpoint isolation tests."""
from __future__ import annotations

import json

import pytest
import torch

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
from sts2_ai.models.shared_transformer import H7Config, H20Batch, SharedObjectTransformer
from sts2_ai.training.h16_joint_act_pilot import (
    H22_SCHEMA, JointActPilotNet, PilotPolicy, encode_joint_decision,
)
from sts2_ai.training.h21_macro_offer_codec import H21_OBJECT_SCHEMA, H21_CODEC


def _state() -> dict:
    return {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [
            {"instance_id": 201, "card_id": "silent.strike",
             "upgrade_level": 0, "state": {}},
        ],
        "relics": [], "potions": [],
        "potion_slot_capacity": 3, "potion_slots_occupied": 0,
        "potion_slots_empty": 3, "map": [],
        "current_map_node_id": None, "combat": None,
        "event": {
            "event_id": "native.ancient_temple",
            "offered_choice_ids": ["gold", "heal"],
            "visible_choice_values": [
                {"choice_id": "gold", "gold_gain": 110, "hp_loss": 6},
                {"choice_id": "heal", "heal": 15},
            ],
        },
    }


def _actions():
    return tuple(
        LegalAction(f"choice-{choice}", "event_choice",
                    json.dumps({"ChoiceId": choice}))
        for choice in ("gold", "heal")
    )


def test_h22_policy_collation_relations_and_gradient(tmp_path):
    torch.manual_seed(1)
    base = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    model = JointActPilotNet.warmstart_h21(base)
    assert model.representation == "h21"
    assert isinstance(model.backbone, H21GroundedMacroTransformer)
    actions = _actions()
    frame = _state()
    encoded = model.encode_decision(frame, actions)
    assert encoded.schema == H21_OBJECT_SCHEMA
    assert encoded.actions[0].target != encoded.actions[1].target
    assert len(encoded.relations) > 0
    batch = model.collate([encoded])
    assert isinstance(batch, H20Batch)
    assert bool(batch.relation_mask.any())

    model.train()
    logits, act, hp = model(batch, value_logits=True, horizon_lambda=0.42)
    loss = logits[0, 0] - logits[0, 1] + 0.1 * (act + hp).sum()
    loss.backward()
    assert model.backbone.relation_target.weight.grad is not None
    assert bool(torch.isfinite(model.backbone.relation_target.weight.grad).all())

    path = tmp_path / "h22.pt"
    model.save(path)
    payload = torch.load(path, weights_only=True)
    assert payload["schema"] == H22_SCHEMA
    assert payload["codec"] == H21_CODEC
    assert payload["object_schema"] == H21_OBJECT_SCHEMA
    assert payload["public_schema"] == "prototype-ai-v1"
    restored = JointActPilotNet.load(path)
    assert restored.representation == "h21"
    assert restored.encode_decision(frame, actions) == encoded
    with pytest.raises(ValueError, match="representation"):
        PilotPolicy(restored, representation="h16", temperature=0.85,
                    horizon_lambda=0.42)


def test_h22_schema_rejects_legacy_and_tampered_checkpoint(tmp_path):
    base = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    h16_file = tmp_path / "h16.pt"
    base.save(h16_file)
    assert JointActPilotNet.load(h16_file).representation == "h16"
    h21 = JointActPilotNet.warmstart_h21(base)
    h21_file = tmp_path / "h21.pt"
    h21.save(h21_file)
    contents = torch.load(h21_file, weights_only=True)
    contents["public_schema"] = "prototype-ai-v0"
    torch.save(contents, h21_file)
    with pytest.raises(ValueError, match="schema"):
        JointActPilotNet.load(h21_file)


def test_h22_alias_audit_and_policy_forward():
    actions = _actions()
    state = _state()
    decision = encode_joint_decision(state, actions, representation="h21")
    assert len(decision.actions) == len(actions)
    model = JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    policy = PilotPolicy(model, representation="h21", temperature=0.85,
                         horizon_lambda=0.42, macro_prior_weight=0)
    scores, value, offsets = policy.infer(
        Observation("prototype-fair-v0", json.dumps(state), "unused"), actions)
    assert len(scores) == 2
    assert 0.0 <= value <= 1.0
    assert offsets == (0.0, 0.0)
