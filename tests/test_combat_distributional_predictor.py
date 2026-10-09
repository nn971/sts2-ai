"""Distributional combat auxiliary head: finite, calibrated interfaces and tests."""
from __future__ import annotations

import json
import math

import pytest

from sts2_ai.training.combat_distributional_fit import (
    constant_distributional_baseline,
    distributional_metrics,
    fit_distributional_predictor,
)
from sts2_ai.training.combat_distributional_predictor import (
    DISTRIBUTIONAL_FORMAT,
    HP_BINS,
    DistributionalCombatOutcomePredictor,
    hp_bin,
)
from sts2_ai.training.combat_predictor import OutcomeExample
from sts2_ai.training.combat_predictor_fit import split_by_run_seed


def example(seed: str, hp: float, win: bool) -> OutcomeExample:
    public = json.dumps({
        "act": 1, "floor": 2, "hp": 70,
        "max_hp": 70, "potions": [
            {"slot": 0, "potion_id": "proto.potion.vulnerable"}
        ], "combat": {
            "turn": 1,
            "enemies": [{"enemy_id": "proto.enemy.twig_slime_s", "hp": 10}],
        },
    })
    return OutcomeExample(seed, public, float(win), hp)


def zero_predictor() -> DistributionalCombatOutcomePredictor:
    return DistributionalCombatOutcomePredictor(
        dimension=16, hidden=3,
        state_weight=[[0.0] * 16 for _ in range(3)],
        state_bias=[0.0] * 3,
        survival_weight=[0.0] * 3,
        survival_bias=math.log(3.0),
        hp_bin_weight=[[0.0] * 3 for _ in range(HP_BINS)],
        hp_bin_bias=[0.0] * HP_BINS,
        model_id="test-distribution",
    )


def test_predictor_has_normalized_joint_probability_and_consistent_moments() -> None:
    model = zero_predictor()
    pred = model.predict(example("seed", 0.6, True).entry_public_json)
    assert pred.survival_probability == pytest.approx(0.75)
    assert pred.joint_hp_probabilities[0] == pytest.approx(0.25)
    assert len(pred.joint_hp_probabilities) == HP_BINS + 1
    assert len(pred.conditional_hp_probabilities) == HP_BINS
    assert sum(pred.joint_hp_probabilities) == pytest.approx(1.0)
    assert pred.expected_exit_hp_fraction == pytest.approx(0.375)
    assert pred.normalized_hp_variance > 0
    assert pred.quantile(0.1) == 0.0
    assert 0.0 <= pred.quantile(0.95) <= 1.0
    with pytest.raises(ValueError):
        pred.quantile(1.01)


def test_model_roundtrip_and_validation(tmp_path) -> None:
    model = zero_predictor()
    saved = tmp_path / "distribution.json"
    model.save(saved)
    loaded = DistributionalCombatOutcomePredictor.load(saved)
    assert loaded.to_dict() == model.to_dict()
    assert loaded.to_dict()["format"] == DISTRIBUTIONAL_FORMAT
    with pytest.raises(ValueError, match="histogram"):
        DistributionalCombatOutcomePredictor(
            dimension=16, hidden=3,
            state_weight=[[0.0] * 16 for _ in range(3)],
            state_bias=[0.0] * 3,
            survival_weight=[0.0] * 3, survival_bias=0.0,
            hp_bin_weight=[[0.0] * 3],
            hp_bin_bias=[0.0],
            model_id="bad",
        )
    with pytest.raises(ValueError, match="combat-entry"):
        loaded.predict('{"combat":null}')
    assert hp_bin(0.0) == 0
    assert hp_bin(1.0) == HP_BINS - 1


def test_distributional_fit_is_run_disjoint_and_uses_no_teacher() -> None:
    pytest.importorskip("torch")
    samples = [
        example(f"run-{i}", 0.7 if i % 2 else 0.0, bool(i % 2))
        for i in range(12)
        for _ in range(2)
    ]
    train, test = split_by_run_seed(samples)
    assert {e.seed for e in train}.isdisjoint({e.seed for e in test})
    progress: list[tuple[int, float]] = []
    model = fit_distributional_predictor(
        train, dimension=32, hidden=8, epochs=3, seed=33,
        progress=lambda i, loss: progress.append((i, loss)),
    )
    assert [x[0] for x in progress] == [1, 2, 3]
    assert all(math.isfinite(x[1]) for x in progress)
    metrics = distributional_metrics(model, test)
    baseline = constant_distributional_baseline(train, test)
    assert all(math.isfinite(x) for x in metrics.values())
    assert all(math.isfinite(x) for x in baseline.values())
    assert metrics["survival_brier"] >= 0.0
    assert metrics["joint_hp_negative_log_likelihood"] >= 0.0


def test_all_defeats_still_fit_survival_without_fictitious_hp_targets() -> None:
    pytest.importorskip("torch")
    samples = [example(f"defeat-{i}", 0.0, False) for i in range(5)]
    model = fit_distributional_predictor(
        samples, dimension=16, hidden=4, epochs=2,
    )
    prediction = model.predict(samples[0].entry_public_json)
    assert sum(prediction.joint_hp_probabilities) == pytest.approx(1.0)
    assert math.isfinite(prediction.normalized_hp_variance)
