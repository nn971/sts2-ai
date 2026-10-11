"""End-to-end v15d PPO entrypoint: native API contract via public toy run."""
from __future__ import annotations

import pytest
from test_pooled_ppo_v15 import PublicObjectsToy, frozen_policy

from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.relational_objects import (
    RELATIONAL_OBJECT_FORMAT,
    PortableRelationalObjectModel,
)
from sts2_ai.training.pooled_ppo_v15 import (
    collect_public_vocabulary,
    train_pooled_combat_ppo,
)


def test_v15d_public_rollout_training_export_and_resume(tmp_path) -> None:
    pytest.importorskip("torch")
    base = frozen_policy()
    vocab = collect_public_vocabulary(
        PublicObjectsToy(), base, runs=2, max_decisions=8, seed=7,
    )
    opts = {
        "rounds": 1, "episodes_per_round": 2, "workers": 1,
        "seed": 103, "max_decisions": 8, "width": 16,
        "epochs": 1, "batch_size": 2, "sample_limit": 8,
        "temperature_start": 1.0, "temperature_end": 1.0,
        "learning_rate": 0.003, "architecture": "relational",
    }
    checkpoint = tmp_path / "relational-ppo.pt"
    model, rounds = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, checkpoint=checkpoint, **opts,
    )
    assert len(rounds) == 1
    assert rounds[0].resolved_combats > 0
    assert rounds[0].optimizer_steps > 0
    assert model.combat.format_id == RELATIONAL_OBJECT_FORMAT
    assert isinstance(model.combat, PortableRelationalObjectModel)
    assert model.strategy.to_dict() == base.strategy.to_dict()
    model_path = tmp_path / "relational-model.json"
    model.save(model_path)
    assert PhaseSplitNeuralModel.load(model_path).to_dict() == model.to_dict()
    assert checkpoint.is_file()
    with pytest.raises(ValueError, match="format"):
        train_pooled_combat_ppo(
            PublicObjectsToy(), base, vocab, checkpoint=checkpoint,
            resume=True, **{**opts, "architecture": "pooled"},
        )
    resumed, history = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, checkpoint=checkpoint,
        resume=True, **{**opts, "rounds": 2},
    )
    assert len(history) == 2
    assert history[0] == rounds[0]
    assert resumed.combat.format_id == RELATIONAL_OBJECT_FORMAT
    independent, other = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, **{**opts, "rounds": 2},
    )
    assert [r.optimizer_steps for r in history] == [
        r.optimizer_steps for r in other
    ]
    assert resumed.to_dict() == independent.to_dict()


def test_v15d_rejects_bad_width_and_preserves_frozen_vocabulary() -> None:
    pytest.importorskip("torch")
    base = frozen_policy()
    vocabulary = ObjectVocabulary.from_dict(ObjectVocabulary.from_object_sets(
        [],
    ).to_dict())
    with pytest.raises(ValueError, match="divisible by 4"):
        train_pooled_combat_ppo(
            PublicObjectsToy(), base, vocabulary, rounds=1,
            architecture="relational", width=15,
        )
