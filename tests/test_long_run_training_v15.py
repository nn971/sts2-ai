"""Long-run v15l snapshots, split PPO diagnostics, and uninterrupted resume."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_pooled_ppo_v15 import PublicObjectsToy, frozen_policy

from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.pooled_ppo_v15 import (
    PooledTrainingRound,
    collect_public_vocabulary,
    train_pooled_combat_ppo,
)
from tools.train_pooled_objects_v15 import _atomic_json, _atomic_model


def test_long_run_public_milestones_resume_without_resetting_optimizer(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    base = frozen_policy()
    vocab = collect_public_vocabulary(
        PublicObjectsToy(), base, runs=2, max_decisions=8, seed=7,
    )
    params = {
        "episodes_per_round": 2, "workers": 1, "seed": 47,
        "max_decisions": 8, "width": 16,
        "epochs": 1, "batch_size": 2, "sample_limit": 8,
        "temperature_start": 1.0, "temperature_end": 1.0,
        "learning_rate": 0.003, "architecture": "relational",
    }
    checkpoint = tmp_path / "checkpoint.pt"
    snapshots: list[tuple[int, bool]] = []

    def publish(history, portable):
        assert checkpoint.exists()
        assert history[-1].round_index == len(history) - 1
        assert history[-1].ppo_samples_used is not None
        assert history[-1].ppo_samples_used <= history[-1].combat_decisions
        assert history[-1].mean_policy_loss is not None
        assert history[-1].mean_value_loss is not None
        assert history[-1].mean_entropy is not None
        snapshots.append((len(history), portable is not None))
        _atomic_json(tmp_path / "live.json", {
            "rows": [asdict(row) for row in history],
        })
        if portable is not None:
            stage = tmp_path / f"stage-{len(history):04d}.json"
            _atomic_model(stage, portable)
            assert PhaseSplitNeuralModel.load(stage).to_dict() == portable.to_dict()
            assert not stage.with_name(stage.name + ".pending").exists()

    _, first = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, rounds=2, checkpoint=checkpoint,
        snapshot_every=2, on_round=publish, **params,
    )
    assert snapshots == [(1, False), (2, True)]
    assert len(json.loads((tmp_path / "live.json").read_text())["rows"]) == 2
    snapshots.clear()

    resumed_model, history = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, rounds=4, checkpoint=checkpoint,
        resume=True, snapshot_every=2, on_round=publish, **params,
    )
    assert snapshots == [(3, False), (4, True)]
    assert history[:2] == first
    assert len(history) == 4
    assert len(json.loads((tmp_path / "live.json").read_text())["rows"]) == 4
    assert (tmp_path / "stage-0002.json").is_file()
    assert (tmp_path / "stage-0004.json").is_file()

    uninterrupted_model, uninterrupted_history = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, rounds=4, **params,
    )
    assert resumed_model.to_dict() == uninterrupted_model.to_dict()
    assert [row.optimizer_steps for row in history] == [
        row.optimizer_steps for row in uninterrupted_history
    ]

    _, finished = train_pooled_combat_ppo(
        PublicObjectsToy(), base, vocab, rounds=4, checkpoint=checkpoint,
        resume=True, snapshot_every=2, on_round=publish, **params,
    )
    assert finished == history and len(snapshots) == 2


def test_old_round_without_diagnostics_remains_loadable() -> None:
    old = {
        "round_index": 4, "played": 32, "completed": 32, "censored": 0,
        "wins": 6, "resolved_combats": 231, "combat_decisions": 5324,
        "optimizer_steps": 96, "mean_loss": 0.00596,
        "mean_combat_target": 0.816, "rollout_seconds": 71.7,
        "optimizer_seconds": 15.3,
    }
    restored = PooledTrainingRound(**old)
    assert restored.ppo_samples_used is None
    assert restored.mean_entropy is None


def test_negative_snapshot_interval_is_rejected() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Snapshot interval"):
        train_pooled_combat_ppo(
            PublicObjectsToy(), frozen_policy(),
            ObjectVocabulary.from_object_sets([]),
            rounds=1, snapshot_every=-1, architecture="relational",
        )
