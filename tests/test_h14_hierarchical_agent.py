"""H14 frozen-H9 codec boundary and macro PPO policy construction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sts2_ai.emulator import FAIR_POLICY_ID, LegalAction, Observation  # noqa: E402
from sts2_ai.models.h9_frozen_codec import SHARED_OBJECT_SCHEMA as H9_SCHEMA  # noqa: E402
from sts2_ai.models.h14_hierarchical_agent import (  # noqa: E402
    H9_CHECKPOINT_FORMAT,
    FrozenH9CombatActor,
    H14HierarchicalAgent,
)
from sts2_ai.models.h14_macro_actor_critic import H14MacroActorCritic  # noqa: E402
from sts2_ai.models.shared_transformer import (  # noqa: E402
    H7Config,
    SharedObjectTransformer,
)


def _observation(public):
    return Observation(FAIR_POLICY_ID, json.dumps(public), "observation-only")


def _h9_fixture(tmp_path: Path):
    config = H7Config(width=32, layers=1, heads=4, feedforward=64)
    model = SharedObjectTransformer(config)
    checkpoint = tmp_path / "legacy-h9.pt"
    torch.save({
        "format": H9_CHECKPOINT_FORMAT,
        "object_schema": H9_SCHEMA,
        "config": asdict(config),
        "state_dict": model.state_dict(),
    }, checkpoint)
    return checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest(), config


def test_frozen_h9_uses_legacy_codec_and_h14_macro_collects_own_choices(tmp_path: Path):
    checkpoint, digest, config = _h9_fixture(tmp_path)
    combat = FrozenH9CombatActor.load_pinned(checkpoint, expected_sha256=digest)
    assert all(not p.requires_grad for p in combat.model.parameters())
    macro = H14MacroActorCritic(config).eval()
    agent = H14HierarchicalAgent(
        combat, macro, episode_seed="h14-test-seed",
        macro_checkpoint_sha256="b" * 64, greedy_macro=True,
    )
    macro_menu = [
        LegalAction("macro-left", "event_choice", json.dumps({"ChoiceId": "left"})),
        LegalAction("macro-right", "event_choice", json.dumps({"ChoiceId": "right"})),
    ]
    chosen = agent.choose(_observation({
        "hp": 60, "gold": 10, "event": {"event_id": "choice"},
        "relics": [{"relic_id": "proto.relic.pen_nib",
                    "state": {"AttacksPlayed": 9}}],
    }), macro_menu)
    assert chosen.action in macro_menu
    assert len(agent.samples) == 1
    assert agent.samples[0].decision.phase == "macro"
    assert len(agent.macro_history) == 1
    assert agent.macro_history[0]["kind"] == "event_choice"

    combat_menu = [
        LegalAction("end", "end_turn", "{}"),
    ]
    combat_obs = {
        "hp": 60,
        "deck": [],
        "relics": [{"relic_id": "proto.relic.pen_nib",
                    "state": {"AttacksPlayed": 9}}],
        "combat": {
            "energy": 3, "turn": 1,
            "hand": [], "enemies": [],
        },
    }
    result = agent.choose(_observation(combat_obs), combat_menu)
    assert result.action == combat_menu[0]
    assert result.policy_name.startswith("frozen-h9-")
    assert len(agent.samples) == 1
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        FrozenH9CombatActor.load_pinned(checkpoint, expected_sha256="0" * 64)


def test_ppo_loss_uses_behavior_probabilities_and_real_values():
    config = H7Config(width=32, layers=1, heads=4, feedforward=64)
    model = H14MacroActorCritic(config)
    combat_path = [
        LegalAction("left", "event_choice", '{"ChoiceId": "left"}'),
        LegalAction("right", "event_choice", '{"ChoiceId": "right"}'),
    ]
    from sts2_ai.models.shared_game_objects import encode_public_decision

    encoded = encode_public_decision({"event": {"event_id": "ev"}}, combat_path)
    loss, stats = model.ppo_loss(
        [encoded], selected_actions=torch.tensor([1]),
        behavior_log_probabilities=torch.tensor([-0.69]),
        behavior_values=torch.tensor([0.0]),
        returns=torch.tensor([1.0]),
    )
    assert set(stats) == {"policy", "value", "entropy"}
    assert torch.isfinite(loss)
    loss.backward()
    assert model.transformer.policy_heads[1][-1].weight.grad is not None
    assert model.transformer.value_heads[1][-1].weight.grad is not None
