"""H3 player PPO correctness: source projection, adversary cohort and updates."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_enemy_instances_v8 import build_model, combat_frame, targeted

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenP0Player
from sts2_ai.models.neural import NEURAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.joint_player_ppo import JointPlayerPPO

torch = pytest.importorskip("torch")


def specimen():
    combat = build_model()
    strategy_raw = combat.to_dict()
    strategy_raw["format"] = NEURAL_FORMAT
    strategy_raw["model_id"] = "h3-test-strategy"
    model = PhaseSplitNeuralModel(
        strategy=NeuralPolicyValueModel.from_dict(strategy_raw),
        combat=combat, model_id="h3-test-player",
        combat_value_objective="hp_preservation",
    )
    return model


def single_record(model, seed):
    actor = FrozenP0Player(
        model, actor_seed=seed, greedy=False,
        temperature=0.85, action_scope="all-legal-v1",
        capture_training=True,
    )
    actions = [
        {"actionId": action.action_id, "kind": action.kind,
         "payloadJson": action.payload_json}
        for action in (targeted(1), targeted(2), targeted(3))
    ]
    selected = actor.choose(combat_frame(), actions)
    episode = SimpleNamespace(
        decisions=(SimpleNamespace(phase="player", selected_id=selected),),
        source_seed=f"joint-test-{seed}", score=0.4 if seed % 2 else 0.9,
        outcome="victory", exit_hp=seed + 20,
    )
    return episode, tuple(actor.training_actions)


def test_joint_player_ppo_updates_exact_v8_weights():
    model = specimen()
    trainer = JointPlayerPPO(
        model, torch=torch, ppo_epochs=2, batch_size=2,
        sample_limit=8,
    )
    cohort = tuple(single_record(model, k) for k in range(4))
    for ep, trace in cohort:
        assert len(trainer.validate_capture(ep, trace, model)) == 1
    before = trainer.model.combat.to_dict()
    result = trainer.update(cohort, seed=8)
    assert result["player_ppo_steps"] >= 2
    assert result["player_ppo_samples"] == 4
    assert trainer.revision == 1
    assert trainer.model.model_id == "h3-player-joint-r1"
    assert trainer.model.combat.to_dict()["policy_weight"] != before["policy_weight"]
    payload = trainer.dump()
    assert payload["optimizer"]
    restored = JointPlayerPPO(
        trainer.model, torch=torch, ppo_epochs=2, batch_size=2,
    )
    restored.restore(payload)
    assert restored.model.to_dict() == trainer.model.to_dict()
    assert restored.revision == trainer.revision
    with pytest.raises(ValueError, match="behavior"):
        restored.update(cohort, seed=9)


def test_joint_player_rejects_tampered_record_and_reordered_support():
    model = specimen()
    example, capture = single_record(model, 11)
    modified = deepcopy(capture[0])
    modified["source_support_identity"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        JointPlayerPPO.validate_capture(example, (modified,), model)
    modified = deepcopy(capture[0])
    modified["behavior_logprob"] = 0.0
    with pytest.raises(ValueError, match="behavior"):
        JointPlayerPPO.validate_capture(example, (modified,), model)
    modified = deepcopy(capture[0])
    modified["selected_id"] = "other"
    with pytest.raises(ValueError, match="action"):
        JointPlayerPPO.validate_capture(example, (modified,), model)


def test_joint_player_frozen_strategy_head():
    model = specimen()
    trainer = JointPlayerPPO(model, torch=torch)
    assert trainer.export().strategy.to_dict() == model.strategy.to_dict()
