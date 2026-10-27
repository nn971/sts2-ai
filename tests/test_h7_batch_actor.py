"""H7 shared inference: legal-menu identity, batched parity, seeded sampling."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_shared_transformer_h7 import frame, legal, model  # noqa: E402

from sts2_ai.emulator.protocol import LegalAction, Observation  # noqa: E402
from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player  # noqa: E402
from sts2_ai.models.h7_batch_actor import (  # noqa: E402
    RemoteH7Policy,
    SharedH7Service,
)
from sts2_ai.models.h7_public_policy import TorchH7PublicPolicy  # noqa: E402
from sts2_ai.nature.protocol import canonical_sha256  # noqa: E402


def _source():
    source = frame()
    candidates = [
        {"actionId": str(i), "kind": action.kind, "payloadJson": action.payload_json}
        for i, action in enumerate(legal())
    ]
    public = Observation(
        "prototype-fair-v0",
        json.dumps(source, sort_keys=True),
        canonical_sha256(source),
    )
    return source, candidates, public, tuple(legal())


def _checkpoint(tmp_path: Path):
    torch.set_num_threads(1)
    network = model().eval()
    path = tmp_path / "pinned-h7.pt"
    network.save_checkpoint(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), network


def test_batched_inference_matches_direct_per_legal_action(tmp_path: Path):
    path, digest, network = _checkpoint(tmp_path)
    _, _, public, legal_actions = _source()
    reference = TorchH7PublicPolicy(network, model_id=f"h7-{digest[:16]}")
    full = reference.evaluate(public, legal_actions)
    partial = reference.evaluate(public, legal_actions[:1])
    with SharedH7Service(
        checkpoint=path, digest=digest, workers=4,
        max_batch=4, max_delay_ms=40.0,
    ) as service:
        policies = [
            RemoteH7Policy(service.transport, slot=service.transport.allocate_slot())
            for _ in range(4)
        ]
        def query(index: int):
            menu = legal_actions if index % 2 == 0 else legal_actions[:1]
            return policies[index].evaluate(public, menu)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(query, range(4)))
    assert service.metrics["requests"] == 4
    assert service.metrics["batches"] >= 1
    assert service.metrics["max_batch_observed"] >= 2
    for i, output in enumerate(results):
        expected = full if i % 2 == 0 else partial
        assert output.action_logits == pytest.approx(expected.action_logits, abs=2e-5)
        assert output.value == pytest.approx(expected.value, abs=2e-5)


def test_shared_model_preserves_frozen_player_seed_and_ppo_source(tmp_path: Path):
    path, digest, network = _checkpoint(tmp_path)
    source, candidates, _, _ = _source()
    direct = FrozenP0Player(
        TorchH7PublicPolicy(network, model_id=f"h7-{digest[:16]}"),
        actor_seed=193, temperature=0.85, greedy=False,
        action_scope="all-legal-v1", capture_training=True,
    )
    expected = direct.choose(source, candidates)
    with SharedH7Service(
        checkpoint=path, digest=digest, workers=1,
        max_batch=4, max_delay_ms=2.0,
    ) as service:
        remote = RemoteH7Policy(service.transport, slot=service.transport.allocate_slot())
        actor = FrozenP0Player(
            remote, actor_seed=193, temperature=0.85, greedy=False,
            action_scope="all-legal-v1", capture_training=True,
        )
        chosen = actor.choose(source, candidates)
    assert expected == chosen
    a, b = direct.training_actions[0], actor.training_actions[0]
    assert a["source_observation"] == b["source_observation"]
    assert a["source_legal_actions"] == b["source_legal_actions"]
    assert a["behavior_model_id"] == b["behavior_model_id"]
    assert a["chosen_index"] == b["chosen_index"]
    assert a["behavior_logprob"] == pytest.approx(b["behavior_logprob"], abs=5e-4)
    assert a["behavior_value"] == pytest.approx(b["behavior_value"], abs=5e-4)
    assert remote.timings_snapshot()["calls"] == 1


def test_unavailable_public_pointer_is_rejected_before_queue(tmp_path: Path):
    path, digest, _ = _checkpoint(tmp_path)
    _, _, public, _ = _source()
    wrong = LegalAction("bad", "play_card", json.dumps({
        "CardInstanceId": 99999,
    }))
    with SharedH7Service(checkpoint=path, digest=digest, workers=1) as service:
        remote = RemoteH7Policy(service.transport, slot=service.transport.allocate_slot())
        with pytest.raises(ValueError, match="visible hand"):
            remote.evaluate(public, (wrong,))
    assert service.metrics["requests"] == 0
