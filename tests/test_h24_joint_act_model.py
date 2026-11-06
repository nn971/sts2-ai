"""H24 isolated checkpoint, exact value order and H22 policy preservation."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.h16_batch_actor import RemoteH16Actor, SharedH16Service
from sts2_ai.models.shared_macro_transformer_v2 import H21GroundedMacroTransformer
from sts2_ai.models.shared_transformer import H7Config
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet, PilotDecision, PilotEpisode, PilotPolicy,
    train_joint_cohort,
)
from sts2_ai.training.h17_replay import fit_replay
from sts2_ai.training.h24_joint_act_model import H24Batch, H24JointActPilotNet


def _state():
    return {
        "hp": 55, "max_hp": 70, "gold": 90, "floor": 6, "act": 1,
        "deck": [{
            "instance_id": 201, "card_id": "silent.strike",
            "upgrade_level": 0, "state": {},
        }],
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
        LegalAction(f"choice-{name}", "event_choice",
                    '{"ChoiceId":"' + name + '"}')
        for name in ("gold", "heal")
    )


def _source():
    torch.manual_seed(24)
    return JointActPilotNet(H21GroundedMacroTransformer(
        H7Config(width=32, heads=4, layers=1, feedforward=64)
    )).eval()


def test_h24_warmstart_preserves_h22_policy_and_shaped_hp_logits():
    h22 = _source()
    state = _state()
    batch = h22.collate([h22.encode_decision(state, _actions())])
    with torch.no_grad():
        original = h22(batch, value_logits=True, horizon_lambda=0.3)
    h24 = H24JointActPilotNet.warmstart_h22(h22).eval()
    frame = h24.encode_decision(state, _actions())
    assert frame.neutral_value is not None
    assert h24.monotone_act.relic_weights is None
    b = h24.collate([frame])
    assert isinstance(b, H24Batch)
    with torch.no_grad():
        transformed = h24(b, value_logits=True, horizon_lambda=0.3)
    torch.testing.assert_close(transformed[0], original[0], rtol=0, atol=0)
    torch.testing.assert_close(transformed[2], original[2], rtol=0, atol=0)
    assert transformed[1].shape == original[1].shape


def test_h24_exact_order_with_fixed_neutral_context():
    h24 = H24JointActPilotNet.warmstart_h22(_source()).eval()
    lower, upper = _state(), _state()
    lower["hp"], lower["gold"] = 24, 20
    upper["hp"], upper["gold"] = 65, 400
    a = h24.encode_decision(lower, _actions())
    b = h24.encode_decision(upper, _actions())
    assert a.neutral_value.encoded == b.neutral_value.encoded
    batch = h24.collate([a, b])
    with torch.no_grad():
        logits, act, shaped_hp = h24(batch, value_logits=True)
    assert act[1] >= act[0] - 1e-7
    assert torch.isfinite(logits).all()
    assert torch.isfinite(shaped_hp).all()
    assert h24.monotone_act.relic_weights is None


def test_h24_checkpoint_roundtrip_isolated_from_h22(tmp_path):
    h22 = _source()
    oldpath = tmp_path / "baseline-h22.pt"
    h22.save(oldpath)
    assert not getattr(JointActPilotNet.load(oldpath), "h24_enabled", False)
    h24 = H24JointActPilotNet.warmstart_h22(h22).eval()
    path = tmp_path / "candidate-h24.pt"
    h24.save(path)
    loaded = JointActPilotNet.load(path)
    assert isinstance(loaded, H24JointActPilotNet)
    assert loaded.selection.relic is False
    assert loaded.vocabulary.relic_ids == ()
    state = _state()
    frames = [h24.encode_decision(state, _actions())]
    with torch.no_grad():
        expected = h24(h24.collate(frames), value_logits=True)
        observed = loaded(loaded.collate(
            [loaded.encode_decision(state, _actions())]), value_logits=True)
    for x, y in zip(expected, observed, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_h24_packed_collator_and_gradients():
    h24 = H24JointActPilotNet.warmstart_h22(_source())
    states = [_state(), deepcopy(_state())]
    states[1]["hp"] = 45
    frames = [h24.encode_decision(x, _actions()) for x in states]
    dense = h24.collate(frames)
    packed = h24.collate(
        frames, packed=True, pad_objects=128, pad_actions=16,
        pad_members=4, pad_relations=128,
    )
    assert packed.action_mask.sum() == dense.action_mask.sum()
    h24.train()
    logits, act, hp, auxiliary = h24(
        packed, value_logits=True, horizon_lambda=0.42,
        auxiliary_lambda=torch.tensor([0.1, 0.9]))
    loss = act.sum() + hp.sum() + auxiliary.sum() + logits[:, :2].sum()
    loss.backward()
    assert h24.monotone_act.base.weight.grad is not None
    assert torch.isfinite(h24.monotone_act.base.weight.grad).all()


def test_h24_refuses_unsupported_checkpoints(tmp_path):
    h24 = H24JointActPilotNet.warmstart_h22(_source())
    path = tmp_path / "h24.pt"
    h24.save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["neutral_encoder"] = "unknown"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="schema mismatch"):
        JointActPilotNet.load(path)


def test_h24_act_head_updates_in_existing_joint_optimizer_and_replay():
    model = H24JointActPilotNet.warmstart_h22(_source())
    state = _state()
    actions = _actions()
    observation = Observation("prototype-fair-v0", json.dumps(state), "h24-macro")
    chosen = PilotDecision(observation, actions, 0, (0.0, 0.0))
    terminal = {**state, "hp": 0}
    episode = PilotEpisode(
        "h24-synthetic-macro", (chosen,), (state, terminal),
        "defeat", searches=0, search_transitions=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    report = train_joint_cohort(
        model, optimizer, [episode], horizon_lambda=0.42,
        epochs=1, minibatch=1, hp_reference=100,
    )
    assert report["optimizer_steps"] == 1
    assert torch.isfinite(torch.tensor(report["loss"])).item()
    replay_rows = [{
        "observation": json.dumps(state),
        "actions": [[a.action_id, a.kind, a.payload_json] for a in actions],
        "phase": "macro", "act": 0.0, "room_hp": [0.8, 0.4],
        "combat_quality": None, "search_policy": [0.5, 0.5],
    }]
    replay = fit_replay(
        model, optimizer, replay_rows, horizon_lambda=0.42,
        replay_weight=0.2, minibatch=1,
    )
    assert replay["steps"] == 1
    assert torch.isfinite(torch.tensor(replay["loss"])).item()
    model.eval()
    lower, higher = _state(), _state()
    lower["hp"], lower["gold"] = 31, 40
    higher["hp"], higher["gold"] = 60, 250
    with torch.inference_mode():
        batch = model.collate([
            model.encode_decision(lower, actions),
            model.encode_decision(higher, actions),
        ])
        _, act, _ = model(batch, value_logits=True)
    assert act[1] >= act[0] - 1e-7


def test_h24_local_policy_infer_uses_dual_encoder():
    model = H24JointActPilotNet.warmstart_h22(_source()).eval()
    state = _state()
    observation = Observation("prototype-fair-v0", json.dumps(state), "h24-macro")
    policy = PilotPolicy(
        model, representation="h21", temperature=0.85, horizon_lambda=0.42,
        macro_prior_weight=0,
    )
    scores, value, offsets = policy.infer(observation, _actions())
    assert len(scores) == 2 and len(offsets) == 2
    assert 0 <= value <= 1


@pytest.mark.parametrize("collator", ["reference", "packed"])
def test_h24_shared_actor_matches_local_inference(tmp_path, collator):
    model = H24JointActPilotNet.warmstart_h22(_source()).eval()
    path = tmp_path / "h24-actor.pt"
    model.save(path)
    encoded = model.encode_decision(_state(), _actions())
    with torch.inference_mode():
        local_logits, _, local_hp = model(
            model.collate([encoded]), horizon_lambda=0.42,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with SharedH16Service(
        checkpoint=path, digest=digest, workers=1, device="cpu",
        horizon_lambda=0.42, max_batch=2,
        fixed_shape_batch_size=2, h21_collator=collator,
    ) as service:
        remote = RemoteH16Actor(service.transport)
        logits, hp = remote.infer(encoded)
    assert list(logits) == pytest.approx(
        local_logits[0, :len(encoded.actions)].detach().tolist(), abs=1e-5)
    assert hp == pytest.approx(float(local_hp[0]), abs=1e-5)
