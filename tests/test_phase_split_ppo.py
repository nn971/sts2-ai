"""PPO importance ratios, combat-terminal GAE and checkpoint safeguards."""
from __future__ import annotations

from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy
from test_tactical_intents_v5 import zeros

from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_DAMAGE_FORMAT
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.phase_split_ppo import gae_terminal
from sts2_ai.training.phase_split_selfplay import train_phase_split


def test_gae_terminal_reward_and_bootstrap() -> None:
    adv, returns = gae_terminal(
        [0.2, 0.4], 1.0, gamma=0.9, lam=1.0,
    )
    # Last TD error = 1-0.4; earlier TD error = 0.9*0.4-0.2.
    assert adv == pytest.approx([0.7, 0.6])
    assert returns == pytest.approx([0.9, 1.0])
    adv, returns = gae_terminal(
        [0.2, 0.4], 0.0, gamma=0.9, lam=0.0,
    )
    assert adv == pytest.approx([0.16, -0.4])
    assert returns == pytest.approx([0.36, 0.0])
    assert gae_terminal([], 1.0, gamma=1.0, lam=1.0) == ([], [])
    with pytest.raises(ValueError, match="GAE"):
        gae_terminal([0.1], 1.0, gamma=0, lam=0.98)


def test_ppo_performs_multiple_updates_and_resumes_with_exact_fingerprint(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    prior = PhaseSplitNeuralModel(
        zeros(NEURAL_FORMAT), zeros(TACTICAL_DAMAGE_FORMAT), "ppo-warm",
        combat_value_objective="hp_preservation",
    )
    warm = tmp_path / "warm.json"
    prior.save(warm)
    checkpoint = tmp_path / "ppo.pt"
    kwargs = dict(
        episodes_per_round=4, dimension=128, hidden=4,
        max_decisions=8, workers=1, seed=22,
        seed_prefix="ppo-test",
        tactical_state_encoding="relational_damage",
        optimizer_method="ppo",
        combat_objective="hp_preservation",
        combat_advantage_baseline="critic",
        loss_normalization="phase_mean",
        hp_monotonic_weight=0.0,
        temperature_start=0.75,
        temperature_end=0.75,
        ppo_epochs=2,
        ppo_batch_size=2,
        ppo_sample_limit=16,
        ppo_clip_epsilon=0.2,
        warm_start=warm, checkpoint=checkpoint,
    )
    initial, rows = train_phase_split(TwoPhaseToy(), rounds=1, **kwargs)
    assert initial.combat.format_id == TACTICAL_DAMAGE_FORMAT
    assert len(rows) == 1
    assert rows[0].optimization_steps > 2
    assert rows[0].phase_loss_diagnostics is not None
    assert rows[0].phase_loss_diagnostics["combat"]["minibatch_updates"] >= 2
    assert rows[0].phase_loss_diagnostics["combat"]["used_samples"] > 0
    assert 0 <= rows[0].phase_loss_diagnostics["combat"]["clip_fraction"] <= 1
    assert initial.to_dict() == PhaseSplitNeuralModel.from_dict(
        initial.to_dict()
    ).to_dict()

    resumed, more = train_phase_split(
        TwoPhaseToy(), rounds=2, resume=True, **kwargs
    )
    assert len(more) == 2
    assert more[0] == rows[0]
    assert resumed.model_id != initial.model_id
    assert resumed.to_dict() != initial.to_dict()

    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            TwoPhaseToy(), rounds=2, resume=True,
            **{**kwargs, "ppo_gamma": 0.9},
        )
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            TwoPhaseToy(), rounds=2, resume=True,
            **{**kwargs, "optimizer_method": "reinforce"},
        )


def test_ppo_rejects_unused_baselines_and_unimplemented_regularization() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="own GAE"):
        train_phase_split(
            TwoPhaseToy(), rounds=1, episodes_per_round=2,
            optimizer_method="ppo", hp_monotonic_weight=0,
            combat_objective="hp_preservation",
            combat_advantage_baseline="leave_one_run_out",
        )
    with pytest.raises(ValueError, match="hp_monotonic_weight"):
        train_phase_split(
            TwoPhaseToy(), rounds=1, episodes_per_round=2,
            optimizer_method="ppo", hp_monotonic_weight=0.2,
        )
    with pytest.raises(ValueError, match="clipping"):
        train_phase_split(
            TwoPhaseToy(), rounds=1, episodes_per_round=2,
            optimizer_method="ppo", hp_monotonic_weight=0,
            ppo_clip_epsilon=2,
        )
