"""H16 batched actor tests: public-token parity and independent clients."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.h16_batch_actor import (
    RemoteH16Actor,
    SharedH16Service,
    _stable_shape_bucket,
)
from sts2_ai.models.packed_actor_collation import prepare_h7_actor_packed
from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet, PilotPolicy
from sts2_ai.training.h16_public_codec import encode_h16_decision


def public_frame(*, combat: bool) -> dict:
    return {
        "act": 1, "floor": 2, "hp": 47, "max_hp": 70,
        "completed_rooms": [{"act": 1, "floor": 1, "room_type": 0}],
        "combat": {"turn": 1, "enemies": []} if combat else None,
        "deck": [], "relics": [], "potions": [], "map": [],
    }


@pytest.mark.parametrize("profile_actor", [False, True])
def test_shared_h16_actor_matches_local_logits_and_horizon(tmp_path, profile_actor):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    checkpoint = tmp_path / "actor.pt"
    model.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    menus = (
        (LegalAction("pass", "end_turn"), LegalAction("play", "other_legal")),
        (LegalAction("left", "event_choice"), LegalAction("right", "event_choice"),
         LegalAction("skip", "skip")),
    )
    states = (public_frame(combat=True), public_frame(combat=False))
    observations = tuple(
        Observation(FAIR_POLICY_ID, json.dumps(state), "test-hash")
        for state in states
    )
    local = PilotPolicy(model, temperature=0.85, horizon_lambda=0.3)
    reference = [
        local.infer(observation, menu)
        for observation, menu in zip(observations, menus, strict=True)
    ]
    with SharedH16Service(
        checkpoint=checkpoint, digest=digest,
        workers=2, device="cpu", horizon_lambda=0.3,
        max_batch=2, max_delay_ms=10.0,
        profile_actor=profile_actor,
    ) as service:
        first = RemoteH16Actor(service.transport)
        second = RemoteH16Actor(service.transport)
        remote_policies = [
            PilotPolicy(None, temperature=0.85, horizon_lambda=0.3,
                        remote_actor=first),
            PilotPolicy(None, temperature=0.85, horizon_lambda=0.3,
                        remote_actor=second),
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(policy.infer, observation, menu)
                for policy, observation, menu in zip(
                    remote_policies, observations, menus, strict=True,
                )
            ]
            received = [future.result(timeout=30) for future in futures]
        for (scores, hp, offsets), (expected, expected_hp, expected_offsets) in zip(
            received, reference, strict=True,
        ):
            assert scores == pytest.approx(expected, abs=1e-5)
            assert hp == pytest.approx(expected_hp, abs=1e-5)
            assert offsets == expected_offsets
        encoded = encode_h16_decision(states[0], menus[0])
        scores, hp = first.infer(encoded)
        assert scores == pytest.approx(reference[0][0], abs=1e-5)
        assert hp == pytest.approx(reference[0][1], abs=1e-5)
    assert service.metrics["requests"] == 3
    assert 1 <= service.metrics["batches"] <= 3
    assert service.metrics["max_batch_observed"] <= 2
    if profile_actor:
        metrics = service.metrics
        assert sum(metrics["p2_batch_size_histogram"].values()) == metrics["batches"]
        assert metrics["p2_actor_distributions"]["actor_queue_delay_ms"]["samples"] == 3
        for key in ("actor_queue_delay_ms", "host_collate_ms",
                    "host_to_device_ms", "forward_sync_ms"):
            distribution = metrics["p2_actor_distributions"][key]
            assert distribution is not None
            assert distribution["samples"] >= 1
            assert 0 <= distribution["p50_ms"] <= distribution["p95_ms"]
            assert distribution["p95_ms"] <= distribution["max_ms"]
    else:
        assert "p2_actor_distributions" not in service.metrics


def test_shared_h16_actor_rejects_invalid_arguments(tmp_path):
    with pytest.raises(ValueError, match="configuration"):
        SharedH16Service(
            checkpoint=tmp_path / "missing.pt", digest="unused",
            workers=0, device="cpu", horizon_lambda=0.0,
        )
    with pytest.raises(ValueError, match="configuration"):
        SharedH16Service(
            checkpoint=tmp_path / "missing.pt", digest="unused",
            workers=1, device="cpu", horizon_lambda=1.1,
        )


def test_mixed_h16_batch_numerical_parity_probe_records_exact_differences(tmp_path):
    torch.manual_seed(73)
    torch.set_num_threads(1)
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    checkpoint = tmp_path / "actor-batch-probe.pt"
    model.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    states = (
        public_frame(combat=True),
        {
            **public_frame(combat=False),
            "deck": [
                {"instance_id": i, "card_id": f"test.card.{i}"}
                for i in range(7)
            ],
        },
    )
    menus = (
        (LegalAction("end", "end_turn"), LegalAction("skip", "skip")),
        (LegalAction("left", "event_choice"), LegalAction("right", "event_choice"),
         LegalAction("skip", "skip")),
    )
    decisions = [
        encode_h16_decision(state, menu)
        for state, menu in zip(states, menus, strict=True)
    ]
    with SharedH16Service(
        checkpoint=checkpoint, digest=digest,
        workers=2, device="cpu", horizon_lambda=0.3,
        max_batch=2, max_delay_ms=150,
        profile_actor=True, verify_batch_parity=True,
    ) as service:
        remotes = [RemoteH16Actor(service.transport) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(remote.infer, decision)
                for remote, decision in zip(remotes, decisions, strict=True)
            ]
            for future in futures:
                scores, value = future.result(timeout=30)
                assert scores
                assert 0 <= value <= 1
    parity = service.metrics["p2_batch_numerical_parity"]
    assert parity["schema"] == "sts2-p2-actor-batch-numerical-parity-v1"
    assert parity["samples"] == 1
    assert 0 <= parity["changed_logits"] <= 1
    assert 0 <= parity["changed_value"] <= 1
    assert 0 <= parity["changed_argmax"] <= 1
    assert parity["max_abs_logit_difference"] >= 0
    assert parity["max_abs_value_difference"] >= 0
    assert service.metrics["max_batch_observed"] == 2


def test_h16_batch_probe_must_be_explicitly_profiled(tmp_path):
    with pytest.raises(ValueError, match="configuration"):
        SharedH16Service(
            checkpoint=tmp_path / "missing.pt", digest="unused",
            workers=2, device="cpu", horizon_lambda=0.25,
            verify_batch_parity=True, profile_actor=False,
        )


def test_h16_fixed_shape_bucket_preserves_real_token_masks():
    state = public_frame(combat=True)
    menu = (LegalAction("end", "end_turn"),
            LegalAction("skip", "skip"))
    token = encode_h16_decision(state, menu)
    phase, objects, actions, members = _stable_shape_bucket(token)
    assert phase == "combat"
    assert objects >= len(token.objects)
    assert actions >= len(token.actions)
    ordinary = prepare_h7_actor_packed([token]).to("cpu")
    stable = prepare_h7_actor_packed(
        [token, token], pad_objects=objects, pad_actions=actions,
        pad_members=members,
    ).to("cpu")
    assert stable.kinds.shape == (2, objects)
    assert stable.action_kinds.shape == (2, actions)
    assert stable.action_members.shape == (2, actions, members)
    assert torch.equal(
        stable.kinds[0, :len(token.objects)], ordinary.kinds[0])
    assert torch.equal(
        stable.action_kinds[0, :len(token.actions)],
        ordinary.action_kinds[0])
    assert stable.object_mask[0].sum().item() == len(token.objects)
    assert stable.action_mask[0].sum().item() == len(token.actions)
    with pytest.raises(ValueError, match="underflows"):
        prepare_h7_actor_packed([token], pad_objects=1)


def test_h16_fixed_shape_actor_exact_replay_under_changed_peer_batch(tmp_path):
    """Same frame is bitwise-stable alone and with a different active peer."""
    torch.manual_seed(97)
    torch.set_num_threads(1)
    model = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64),
    )).eval()
    ckpt = tmp_path / "fixed-shape.pt"
    model.save(ckpt)
    digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    target = encode_h16_decision(
        public_frame(combat=True),
        (LegalAction("end", "end_turn"), LegalAction("skip", "skip")),
    )
    peer = encode_h16_decision(
        {**public_frame(combat=True), "hp": 31},
        (LegalAction("pass", "end_turn"), LegalAction("play", "other_legal")),
    )
    assert _stable_shape_bucket(target) == _stable_shape_bucket(peer)
    with SharedH16Service(
        checkpoint=ckpt, digest=digest, workers=2, device="cpu",
        horizon_lambda=0.3, max_batch=2, max_delay_ms=100,
        fixed_shape_batch_size=2, profile_actor=True,
    ) as service:
        a, b = (RemoteH16Actor(service.transport) for _ in range(2))
        solo = a.infer(target)
        for _ in range(3):
            with ThreadPoolExecutor(max_workers=2) as pool:
                same = pool.submit(a.infer, target)
                other = pool.submit(b.infer, peer)
                assert same.result(timeout=30) == solo
                assert other.result(timeout=30)[0]
    assert service.metrics["requests"] == 7
    assert service.metrics["p2_fixed_shape_batch_size"] == 2
    assert service.metrics["p2_fixed_shape_physical_rows"] >= 7
    assert service.metrics["p2_fixed_shape_bucket_forwards"] >= 4


def test_fixed_shape_actor_rejects_conflicting_numerical_probe(tmp_path):
    with pytest.raises(ValueError, match="configuration"):
        SharedH16Service(
            checkpoint=tmp_path / "missing.pt", digest="unused",
            workers=2, device="cpu", horizon_lambda=0.25,
            verify_batch_parity=True, profile_actor=True,
            fixed_shape_batch_size=2,
        )
