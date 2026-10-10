"""HP-conservation targets and on-policy, other-run tactical baselines."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_hp_first_tactical import outcome, resources, small_model
from test_phase_split_selfplay import TwoPhaseToy

from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.training.combat_outcomes import CombatOutcome
from sts2_ai.training.phase_split_selfplay import (
    _hp_preservation_target,
    _other_run_combat_baselines,
    train_phase_split,
)


def test_hp_preservation_depends_on_health_lost_not_starting_health() -> None:
    high_entry = replace(
        outcome(exit_hp=50), entry=resources(70),
    )
    low_entry = replace(
        outcome(exit_hp=20), entry=resources(40),
    )
    assert _hp_preservation_target(high_entry) == pytest.approx(1 - 0.5 * 20 / 70)
    assert _hp_preservation_target(low_entry) == pytest.approx(
        _hp_preservation_target(high_entry)
    )
    unhurt = replace(outcome(exit_hp=30), entry=resources(30))
    healed = replace(outcome(exit_hp=40), entry=resources(30))
    assert _hp_preservation_target(unhurt) == 1
    assert _hp_preservation_target(healed) == 1
    assert _hp_preservation_target(outcome("defeat", exit_hp=0)) == 0
    assert _hp_preservation_target(outcome("defeat", exit_hp=None)) == 0


def test_potion_identity_does_not_affect_hp_conservation_target() -> None:
    spent = outcome(exit_hp=30)
    saved = replace(spent, exit=resources(30, ("fire", "swift")), potions_used=())
    assert _hp_preservation_target(spent) == _hp_preservation_target(saved)
    with pytest.raises(ValueError, match="HP preservation"):
        _hp_preservation_target(replace(spent, entry=resources(None)))


def test_other_run_baseline_is_encounter_matched_and_excludes_whole_run() -> None:
    slime = outcome(exit_hp=30)
    boss = replace(slime, enemy_ids=("boss",))
    cohorts: list[list[tuple[CombatOutcome, float]]] = [
        [(slime, 0.6), (slime, 0.7), (boss, 0.8)],
        [(slime, 0.9)],
        [(boss, 0.55)],
    ]
    values = _other_run_combat_baselines(cohorts)
    assert values[0] == pytest.approx([0.9, 0.9, 0.55])
    assert values[1] == pytest.approx([0.65])
    assert values[2] == pytest.approx([0.8])
    assert _other_run_combat_baselines([[(slime, 0.8)]]) == [[0.5]]
    assert _other_run_combat_baselines([]) == []


def test_hp_preservation_training_checkpoint_and_objective_isolation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    old_model = PhaseSplitNeuralModel(
        small_model(NEURAL_FORMAT),
        small_model(TACTICAL_FORMAT),
        "old-hp-first", "hp_first",
    )
    warm = tmp_path / "warm.json"
    old_model.save(warm)
    checkpoint = tmp_path / "save.pt"
    backend = TwoPhaseToy()
    model, rounds = train_phase_split(
        backend, rounds=2, episodes_per_round=4,
        dimension=32, hidden=4, max_decisions=8,
        combat_objective="hp_preservation",
        combat_advantage_baseline="leave_one_run_out",
        warm_start=warm, checkpoint=checkpoint, temperature_start=1,
        temperature_end=1,
    )
    assert model.combat_value_objective == "hp_preservation"
    assert PhaseSplitNeuralModel.from_dict(model.to_dict()).to_dict() == model.to_dict()
    assert all(row.mean_victory_hp_lost_fraction == pytest.approx(7 / 40)
               for row in rounds)
    assert all(row.optimization_steps == 2 for row in rounds)
    assert not backend.states
    resumed, continuation = train_phase_split(
        backend, rounds=3, episodes_per_round=4,
        dimension=32, hidden=4, max_decisions=8,
        combat_objective="hp_preservation",
        combat_advantage_baseline="leave_one_run_out",
        warm_start=warm, checkpoint=checkpoint, resume=True,
        temperature_start=1, temperature_end=1,
    )
    assert len(continuation) == 3
    assert continuation[:2] == rounds
    assert resumed.combat_value_objective == "hp_preservation"
    with pytest.raises(ValueError, match="mismatch"):
        train_phase_split(
            backend, rounds=3, episodes_per_round=4,
            dimension=32, hidden=4, max_decisions=8,
            combat_objective="hp_preservation",
            combat_advantage_baseline="critic",
            warm_start=warm, checkpoint=checkpoint, resume=True,
            temperature_start=1, temperature_end=1,
        )


def test_loo_rejected_for_long_horizon_objective() -> None:
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="HP-based"):
        train_phase_split(
            TwoPhaseToy(), rounds=1, episodes_per_round=2,
            combat_objective="continuation",
            combat_advantage_baseline="leave_one_run_out",
        )
