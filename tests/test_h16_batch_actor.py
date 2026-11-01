"""H16 batched actor tests: public-token parity and independent clients."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.models.h16_batch_actor import RemoteH16Actor, SharedH16Service
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


def test_shared_h16_actor_matches_local_logits_and_horizon(tmp_path):
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
