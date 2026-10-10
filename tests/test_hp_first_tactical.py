"""Temporary HP-first tactical curriculum: preserve HP, spend potions freely."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_phase_split_selfplay import TwoPhaseToy

from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    TACTICAL_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.combat_outcomes import CombatOutcome, RunResources
from sts2_ai.training.phase_split_selfplay import (
    _hp_first_combat_target,
    train_phase_split,
)


def resources(hp: int | None, potions: tuple[str, ...] = ()) -> RunResources:
    return RunResources(hp=hp, max_hp=70, potions=potions, gold=25, relics=())


def outcome(
    result: str = "victory", *,
    exit_hp: int | None = 35,
    entry_potions: tuple[str, ...] = ("fire", "swift"),
    exit_potions: tuple[str, ...] = (),
    potions_used: tuple[str, ...] = ("fire", "swift"),
) -> CombatOutcome:
    return CombatOutcome(
        act=1, floor=4, enemy_ids=("slime",),
        start_decision=0, end_decision=2,
        turn_count=2, result=result,
        entry=resources(55, entry_potions),
        exit=resources(exit_hp, exit_potions),
        hp_decreases=20, hp_increases=0,
        potions_used=potions_used,
    )


def small_model(fmt: str) -> NeuralPolicyValueModel:
    return NeuralPolicyValueModel.from_dict({
        "format": fmt, "dimension": 32, "hidden": 4, "model_id": fmt,
        "state_weight": [[0.0] * 32 for _ in range(4)],
        "state_bias": [0.0] * 4,
        "action_weight": [[0.0] * 32 for _ in range(4)],
        "action_bias": [0.0] * 4,
        "policy_weight": [0.0] * 4, "policy_bias": 0.0,
        "value_weight": [0.0] * 4, "value_bias": 0.9,
        "value_head_trained": True,
    })


def test_hp_first_orders_victory_and_healthy_exit_above_defeat() -> None:
    healthy = _hp_first_combat_target(outcome(exit_hp=50))
    damaged = _hp_first_combat_target(outcome(exit_hp=10))
    defeat = _hp_first_combat_target(outcome("defeat", exit_hp=0))
    assert defeat == 0.0
    assert healthy == pytest.approx(0.5 + 0.5 * 50 / 70)
    assert damaged == pytest.approx(0.5 + 0.5 * 10 / 70)
    assert healthy > damaged > defeat
    assert _hp_first_combat_target(outcome(exit_hp=70)) == 1.0


def test_no_potion_penalty_or_bonus_even_if_named_potions_change() -> None:
    used_everything = outcome(exit_hp=32)
    kept_everything = replace(
        used_everything,
        exit=resources(32, ("fire", "swift")),
        potions_used=(),
    )
    changed_names = replace(
        used_everything,
        entry=resources(55, ("unknown_potion",)),
        potions_used=("unknown_potion",),
    )
    assert _hp_first_combat_target(used_everything) == (
        _hp_first_combat_target(kept_everything)
    )
    assert _hp_first_combat_target(changed_names) == (
        _hp_first_combat_target(kept_everything)
    )
    # Terminal defeats can lack reliable HP and should still train as defeat.
    assert _hp_first_combat_target(outcome("defeat", exit_hp=None)) == 0
    with pytest.raises(ValueError, match="HP-first"):
        _hp_first_combat_target(outcome(exit_hp=None))


def test_objective_is_serialized_and_old_models_stay_backward_compatible() -> None:
    legacy = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT),
        small_model(TACTICAL_FORMAT), "legacy",
    )
    assert "combat_value_objective" not in legacy.to_dict()
    assert PhaseSplitNeuralModel.from_dict(legacy.to_dict()).combat_value_objective == (
        "continuation"
    )
    hp_first = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT), small_model(TACTICAL_FORMAT),
        "hp-first", "hp_first",
    )
    assert PhaseSplitNeuralModel.from_dict(hp_first.to_dict()).to_dict() == (
        hp_first.to_dict()
    )
    with pytest.raises(ValueError, match="objective"):
        PhaseSplitNeuralModel(
            small_model(NEURAL_FORMAT), small_model(TACTICAL_FORMAT),
            "bad", "potions_are_free_forever",
        )


def test_hp_first_tactical_training_and_checkpoint_mode_guard(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    backend = TwoPhaseToy()
    old = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT), small_model(TACTICAL_FORMAT), "old",
    )
    warm = tmp_path / "warm.json"
    old.save(warm)
    checkpoint = tmp_path / "hp-first.pt"
    model, rows = train_phase_split(
        backend, rounds=2, episodes_per_round=4,
        dimension=32, hidden=4, max_decisions=8,
        combat_objective="hp_first", warm_start=warm,
        checkpoint=checkpoint, temperature_start=1,
        temperature_end=1,
    )
    assert model.combat_value_objective == "hp_first"
    assert all(row.optimization_steps == 2 for row in rows)
    assert all(row.mean_victory_exit_hp_fraction == pytest.approx(23 / 40) for row in rows)
    assert all(row.mean_potions_used_per_combat == 0 for row in rows)
    assert not backend.states
    resumed, extended = train_phase_split(
        backend, rounds=3, episodes_per_round=4,
        dimension=32, hidden=4, max_decisions=8,
        combat_objective="hp_first", warm_start=warm,
        checkpoint=checkpoint, resume=True,
        temperature_start=1, temperature_end=1,
    )
    assert len(extended) == 3
    assert resumed.combat_value_objective == "hp_first"
    assert rows == extended[:2]
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            backend, rounds=3, episodes_per_round=4,
            dimension=32, hidden=4, max_decisions=8,
            combat_objective="continuation", warm_start=warm,
            checkpoint=checkpoint, resume=True,
            temperature_start=1, temperature_end=1,
        )
