"""Optional PyTorch H14 macro policy/value objective and checkpoint contracts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sts2_ai.models.h14_macro_actor_critic import (  # noqa: E402
    H14MacroActorCritic,
    H14_MACRO_MODEL_FORMAT,
)
from sts2_ai.models.shared_game_objects import encode_public_decision  # noqa: E402
from sts2_ai.models.shared_transformer import H7Config  # noqa: E402


def menu():
    return [SimpleNamespace(kind="event_choice", payload_json=json.dumps(
        {"ChoiceId": c})) for c in ("left", "right")]


def decision():
    return encode_public_decision(
        {"hp": 49, "gold": 90, "event": {"event_id": "ev"},
         "relics": [{"relic_id": "proto.relic.pen_nib",
                     "state": {"AttacksPlayed": 9}}]},
        menu(),
    )


def test_h14_macro_shared_policy_and_critic_both_receive_gradients():
    network = H14MacroActorCritic(H7Config(
        width=48, layers=1, heads=4, feedforward=96,
    ))
    encoded = decision()
    total, metrics = network.policy_value_loss(
        [encoded], selected_actions=torch.tensor([1]),
        returns=torch.tensor([0.8]),
    )
    assert set(metrics) == {"policy", "value"}
    total.backward()
    assert network.transformer.policy_heads[1][-1].weight.grad is not None
    assert network.transformer.value_heads[1][-1].weight.grad is not None
    assert network.transformer.kinds.weight.grad is not None
    result = network.evaluate({
        "hp": 49, "gold": 90, "event": {"event_id": "ev"},
    }, menu())
    assert len(result.action_logits) == 2
    assert -1 <= result.continuation_value <= 1


def test_h14_search_distillation_preserves_chance_vs_player_actions():
    net = H14MacroActorCritic(H7Config(
        width=32, layers=1, heads=4, feedforward=64,
    ))
    loss, _ = net.policy_value_loss(
        [decision()], search_probabilities=torch.tensor([[0.2, 0.8]]),
        returns=torch.tensor([-1.0]),
    )
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="normalized"):
        net.policy_value_loss(
            [decision()], search_probabilities=torch.tensor([[0.5, 0.1]]),
            returns=torch.tensor([1.0]),
        )


def test_h14_new_checkpoint_rejects_legacy_schema(tmp_path: Path):
    config = H7Config(width=32, layers=1, heads=4, feedforward=64)
    net = H14MacroActorCritic(config).eval()
    path = tmp_path / "h14-macro.pt"
    net.save_checkpoint(path, provenance={"emulator_revision": "pinned"})
    loaded = H14MacroActorCritic.load_checkpoint(path).eval()
    with torch.no_grad():
        a = net([decision()])
        b = loaded([decision()])
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])
    raw = torch.load(path, weights_only=True)
    assert raw["format"] == H14_MACRO_MODEL_FORMAT
    raw["object_schema"] = "sts2-h7-public-shared-objects-v1"
    torch.save(raw, path)
    with pytest.raises(ValueError, match="Unsupported"):
        H14MacroActorCritic.load_checkpoint(path)
