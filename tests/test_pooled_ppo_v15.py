"""v15c PPO: real public segment GAE, frozen strategy and deterministic resume."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from test_enemy_instances_v8 import build_model
from test_object_instances_v15 import frame
from test_phase_split_selfplay import TwoPhaseToy

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NEURAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.object_instances import extract_combat_objects
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.pooled_objects import PooledObjectTorch
from sts2_ai.training.combat_outcomes import CombatOutcome, RunResources
from sts2_ai.training.pooled_ppo_v15 import (
    collect_public_vocabulary,
    ppo_update_pooled,
    prepare_pooled_samples,
    train_pooled_combat_ppo,
)
from sts2_ai.training.selfplay import Episode, PublicDecision


def frozen_policy() -> PhaseSplitNeuralModel:
    combat = build_model()
    strategy_raw = combat.to_dict()
    strategy_raw["format"] = NEURAL_FORMAT
    for name in ("enemy_weight", "enemy_bias",
                 "enemy_context_weight", "enemy_target_weight"):
        strategy_raw.pop(name)
    strategy = NeuralPolicyValueModel.from_dict(strategy_raw)
    return PhaseSplitNeuralModel(strategy, combat, "frozen-v8-hp",
                                 combat_value_objective="hp_preservation")


def obs(state: dict) -> Observation:
    return Observation("prototype-fair-v0", json.dumps(state, sort_keys=True), "")


def combat_episode(*, completed: bool = True, resolved: bool = True) -> Episode:
    state = frame()
    actions = (
        LegalAction("attack-first", "play_card",
                    '{"CardInstanceId":11,"TargetEnemyId":1}'),
        LegalAction("attack-second", "play_card",
                    '{"CardInstanceId":12,"TargetEnemyId":2}'),
        LegalAction("end", "end_turn", "{}"),
    )
    decisions = (
        PublicDecision(obs(state), actions, 0, "combat"),
        PublicDecision(obs(state), actions, 1, "combat"),
    )
    resources = RunResources.from_public(state)
    out = replace(resources, hp=resources.hp - 5)
    record = CombatOutcome(
        act=1, floor=10, enemy_ids=("nibbit",), start_decision=0, end_decision=2,
        turn_count=1, result="victory", entry=resources, exit=out,
        hp_decreases=5, hp_increases=0, potions_used=(),
    )
    return Episode(
        "fake-public-only", decisions,
        "victory" if completed else "truncated",
        1, 10, 0.5, completed, combat_outcomes=(record,) if resolved else (),
    )


def test_onpolicy_segment_gae_discards_only_unresolved_combats() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    base = frozen_policy()
    vocabulary = ObjectVocabulary.from_object_sets([extract_combat_objects(frame())])
    head = PooledObjectTorch(vocabulary, width=16)
    episode = combat_episode(completed=False, resolved=True)
    samples, targets = prepare_pooled_samples(
        [episode], base.combat, head,
        temperature=0.85, gamma=0.995, gae_lambda=0.98,
    )
    assert len(samples) == 2
    assert len(targets) == 1 and 0 < targets[0] <= 1
    assert all(s.action_refs and len(s.action_refs) == 3 for s in samples)
    assert all(s.old_logprob <= 0 and s.target <= 1.01 for s in samples)
    none, rewards = prepare_pooled_samples(
        [combat_episode(completed=False, resolved=False)], base.combat, head,
        temperature=0.85, gamma=0.995, gae_lambda=0.98,
    )
    assert none == [] and rewards == []
    with pytest.raises(ValueError, match="unrecorded combat"):
        prepare_pooled_samples(
            [combat_episode(completed=True, resolved=False)], base.combat, head,
            temperature=0.85, gamma=0.995, gae_lambda=0.98,
        )


def test_ppo_updates_only_pooled_head_and_allows_variable_menus() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    base = frozen_policy()
    vocabulary = ObjectVocabulary.from_object_sets([extract_combat_objects(frame())])
    head = PooledObjectTorch(vocabulary, width=16)
    episode = combat_episode()
    samples, _ = prepare_pooled_samples(
        [episode], base.combat, head,
        temperature=0.85, gamma=0.995, gae_lambda=0.98,
    )
    # One-action samples can coexist with wider action menus.
    shorter = replace(samples[0], action_refs=samples[0].action_refs[:1],
                      baseline=replace(samples[0].baseline,
                                       action_logits=samples[0].baseline.action_logits[:1]),
                      chosen_index=0, old_logprob=0.0)
    optimizer = torch.optim.AdamW(head.module.parameters(), lr=0.02)
    old_base = copy.deepcopy(base.to_dict())
    metrics = ppo_update_pooled(
        head, [samples[0], samples[1], shorter], optimizer,
        temperature=0.85, epochs=2, batch_size=2,
        sample_limit=16, seed=11,
    )
    assert metrics.steps == 4 and metrics.samples == 3
    assert metrics.loss is not None
    assert torch.isfinite(head.module.value_gate).item()
    assert base.to_dict() == old_base
    assert head.module.value_gate.abs().item() > 0


class PublicObjectsToy(TwoPhaseToy):
    def observe(self, state, policy):
        original = super().observe(state, policy)
        if self.states[state] != "combat":
            return original
        public = frame()
        public["hp"] = 30
        public["max_hp"] = 40
        return obs(public)


def test_real_rollout_ppo_resume_and_vocabulary_binding(tmp_path) -> None:
    pytest.importorskip("torch")
    base = frozen_policy()
    vocab = collect_public_vocabulary(
        PublicObjectsToy(), base, runs=2, max_decisions=8, seed=7,
    )
    options = dict(
        rounds=1, episodes_per_round=3, workers=1,
        seed=37, max_decisions=8, width=16,
        epochs=2, batch_size=2, sample_limit=16,
        temperature_start=1.0, temperature_end=1.0,
        learning_rate=0.003,
    )
    checkpoint = tmp_path / "v15c.pt"
    initial, rows = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, checkpoint=checkpoint, **options,
    )
    assert len(rows) == 1
    assert rows[0].combat_decisions == 3
    assert rows[0].resolved_combats == 3
    assert rows[0].optimizer_steps > 0
    assert initial.combat.format_id.endswith("pooled-object-residual")
    assert initial.strategy.to_dict() == base.strategy.to_dict()

    longer = {**options, "rounds": 2}
    resumed, history = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab,
        checkpoint=checkpoint, resume=True, **longer,
    )
    assert len(history) == 2 and history[0] == rows[0]
    independent, comparison = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, **longer,
    )
    assert resumed.to_dict() == independent.to_dict()
    assert [r.optimizer_steps for r in history] == [
        r.optimizer_steps for r in comparison
    ]
    # A different vocabulary must not silently reuse the old optimizer.
    different = ObjectVocabulary.from_object_sets([
        extract_combat_objects({**frame(), "character_id": "other_character"})
    ])
    with pytest.raises(ValueError, match="mismatch"):
        train_pooled_combat_ppo(
            PublicObjectsToy(), base, different,
            checkpoint=checkpoint, resume=True, **longer,
        )
