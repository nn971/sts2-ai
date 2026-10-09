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


def test_joint_potion_hp_distribution_and_context_fallback() -> None:
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
    fallback = distribution.estimate(unmatched)
    assert not fallback["matched_context"]
    assert fallback["sample_count"] == 3


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
