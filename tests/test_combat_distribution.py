"""Combat-outcome distribution retains joint HP and potion outcomes."""
from __future__ import annotations

import json

import pytest

from sts2_ai.training.combat_distribution import (
    EmpiricalCombatDistribution,
    load_combat_samples,
)


def sample(hp: int, potions: list[str], *, victory: bool = True) -> dict:
    return {
        "result": "victory" if victory else "defeat",
        "enemy_ids": ["slime"],
        "entry": {"hp": 65, "max_hp": 70, "potions": ["power", "weak"]},
        "exit": {"hp": hp, "max_hp": 70, "potions": potions},
    }


def test_joint_potion_hp_distribution_and_sparse_abstention() -> None:
    outcomes = [
        sample(60, ["power", "weak"]),
        sample(55, ["weak"]),
        sample(0, ["power", "weak"], victory=False),
    ]
    distribution = EmpiricalCombatDistribution(outcomes, min_group_size=2)
    matched = distribution.estimate(outcomes[0])
    assert matched["matched_context"]
    assert matched["sample_count"] == 3
    assert matched["observed_win_probability"] == pytest.approx(2 / 3)
    assert len(matched["joint_exit_outcomes"]) == 3
    assert matched["exit_hp_normalized_variance"] > 0
    unmatched = sample(45, ["power"])
    unmatched["enemy_ids"] = ["other"]
    abstention = distribution.estimate(unmatched)
    assert abstention["status"] == "insufficient_data"
    assert not abstention["matched_context"]
    assert abstention["sample_count"] == 0
    assert abstention["observed_win_probability"] is None
    assert abstention["exit_hp_mean"] is None
    assert abstention["joint_exit_outcomes"] == []
    summary = distribution.dataset_summary()
    assert summary["scope"] == "unconditional_mixed_contexts"
    assert not summary["usable_as_conditional_forecast"]
    assert summary["sample_count"] == 3
    assert summary["observed_win_probability"] == pytest.approx(2 / 3)


def test_sparse_matching_context_abstains_without_relabeling_other_fights() -> None:
    outcomes = [sample(55, ["power"]), sample(0, [], victory=False)]
    model = EmpiricalCombatDistribution(outcomes, min_group_size=3)
    estimate = model.estimate(outcomes[0])
    assert estimate["matched_sample_count"] == 2
    assert estimate["required_sample_count"] == 3
    assert estimate["status"] == "insufficient_data"
    assert estimate["observed_win_probability"] is None


def test_normalized_variance_uses_each_record_maximum_hp() -> None:
    a = sample(35, [])
    b = sample(70, [])
    a["entry"]["hp"] = 35
    a["entry"]["max_hp"] = 35
    b["entry"]["hp"] = 70
    b["entry"]["max_hp"] = 70
    distribution = EmpiricalCombatDistribution([a, b], min_group_size=1)
    summary = distribution.dataset_summary()
    assert summary["exit_hp_mean"] == pytest.approx(52.5)
    assert summary["exit_hp_fraction_mean"] == pytest.approx(1.0)
    assert summary["exit_hp_normalized_variance"] == pytest.approx(0.0)


def test_loader_rejects_unknown_schemas(tmp_path) -> None:
    file = tmp_path / "outcomes.jsonl"
    file.write_text(json.dumps({
        "schema": "sts2-public-combat-sample-v1",
        "outcome": sample(60, ["weak"]),
    }) + "\n")
    assert len(load_combat_samples([file])) == 1
    file.write_text('{"schema":"other"}\n')
    with pytest.raises(ValueError, match="schema"):
        load_combat_samples([file])


def test_empty_samples_are_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="No completed"):
        EmpiricalCombatDistribution([])
