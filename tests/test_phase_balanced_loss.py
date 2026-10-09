"""Loss normalization and v4 checkpoint schema regressions."""
from __future__ import annotations

from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy

from sts2_ai.models.neural import TACTICAL_STRUCTURED_FORMAT
from sts2_ai.training.phase_split_selfplay import (
    BALANCED_TRAINING_VERSION,
    SPLIT_TRAINING_VERSION,
    train_phase_split,
)


def test_phase_mean_updates_both_models_and_reports_mean_components(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    env = TwoPhaseToy()
    checkpoint = tmp_path / "phase-mean.pt"
    model, rows = train_phase_split(
        env, rounds=2, episodes_per_round=6, dimension=64, hidden=8,
        max_decisions=8, seed=113, temperature_start=0.7,
        temperature_end=0.7, tactical_state_encoding="structured",
        loss_normalization="phase_mean", checkpoint=checkpoint,
    )
    assert model.combat.format_id == TACTICAL_STRUCTURED_FORMAT
    assert len(rows) == 2 and all(row.optimization_steps == 2 for row in rows)
    for row in rows:
        assert row.phase_loss_diagnostics is not None
        assert set(row.phase_loss_diagnostics) == {"strategy", "combat"}
        for components in row.phase_loss_diagnostics.values():
            assert set(components) == {"policy", "value_mse", "entropy"}
            assert components["value_mse"] >= 0
            assert components["entropy"] >= 0
        assert row.strategic_decisions > 0
        assert row.tactical_decisions > 0
    assert SPLIT_TRAINING_VERSION != BALANCED_TRAINING_VERSION
    assert not env.states

    resumed, after = train_phase_split(
        env, rounds=3, episodes_per_round=6, dimension=64, hidden=8,
        max_decisions=8, seed=113, temperature_start=0.7,
        temperature_end=0.7, tactical_state_encoding="structured",
        loss_normalization="phase_mean", checkpoint=checkpoint, resume=True,
    )
    assert len(after) == 3
    assert rows == after[:2]
    assert resumed.to_dict() != model.to_dict()

    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            env, rounds=3, episodes_per_round=6, dimension=64, hidden=8,
            max_decisions=8, seed=113, temperature_start=0.7,
            temperature_end=0.7, tactical_state_encoding="structured",
            loss_normalization="legacy_episode_sum",
            checkpoint=checkpoint, resume=True,
        )


def test_loss_normalization_rejects_unsupported_modes() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Loss normalization"):
        train_phase_split(
            TwoPhaseToy(), rounds=1, episodes_per_round=2,
            loss_normalization="none",
        )


def test_phase_mean_vs_legacy_updates_are_not_silently_identical() -> None:
    pytest.importorskip("torch")
    arguments = {
        "rounds": 1, "episodes_per_round": 8,
        "dimension": 64, "hidden": 8, "seed": 27,
        "max_decisions": 8, "temperature_start": 0.7,
        "temperature_end": 0.7, "tactical_state_encoding": "structured",
    }
    a, a_rows = train_phase_split(
        TwoPhaseToy(), loss_normalization="phase_mean", **arguments,
    )
    b, b_rows = train_phase_split(
        TwoPhaseToy(), loss_normalization="legacy_episode_sum", **arguments,
    )
    assert a_rows[0].strategic_decisions == b_rows[0].strategic_decisions
    assert a_rows[0].tactical_decisions == b_rows[0].tactical_decisions
    # AdamW can yield nearly identical one-step parameter updates when
    # gradients only differ by a positive scalar. Check the objective itself.
    assert a_rows[0].mean_loss != pytest.approx(b_rows[0].mean_loss)
    assert a.combat.format_id == b.combat.format_id
