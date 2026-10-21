"""Deterministic, source-balanced H2 screens and eight-way diagnostics."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_nature_h1_frozen_model_reuse import Episode
from test_nature_draw_only_h1 import episode

from sts2_ai.nature.combat_scenarios import (
    DECK_PRESETS, deck_for_seed, scenario_for_seed,
)
from sts2_ai.nature.h2_benchmark import (
    balanced_h2_source_seeds, configuration_key,
    reference_by_configuration,
)
from tools.train_draw_only_nature_h1 import paired_holdout_metrics


POOL = ("shrinker-a10", "nibbits-a10")


def test_h2_balanced_source_generation_is_exact_and_reproducible():
    seeds = balanced_h2_source_seeds(
        "selection", 128, POOL, DECK_PRESETS,
    )
    assert seeds == balanced_h2_source_seeds(
        "selection", 128, tuple(reversed(POOL)),
        tuple(reversed(DECK_PRESETS)),
    )
    assert len(set(seeds)) == 128
    assert set(seeds).isdisjoint(balanced_h2_source_seeds(
        "confirmation", 128, POOL, DECK_PRESETS,
    ))
    assert set(seeds).isdisjoint({
        f"nature-h1-frozen-selection-{i:08d}" for i in range(128)
    })
    counts = {}
    for seed in seeds:
        key = configuration_key(
            scenario_for_seed(seed, POOL).scenario_id,
            deck_for_seed(seed, DECK_PRESETS),
        )
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 8
    assert set(counts.values()) == {16}


@pytest.mark.parametrize("split,count,scenarios", [
    ("selection", 127, POOL),
    ("selection", 128, ("shrinker-a0", "nibbits-a10")),
    ("unexpected", 128, POOL),
    ("selection", 128, ("shrinker-a10", "shrinker-a10")),
])
def test_h2_balanced_screen_rejects_invalid_design(split, count, scenarios):
    with pytest.raises(ValueError):
        balanced_h2_source_seeds(split, count, scenarios, DECK_PRESETS)


def _row(i, scenario, deck, score, draw):
    return Episode(
        source_seed=f"h2-source-{i}",
        reference_seed=44 + i,
        source_opening_fingerprint=f"h2-opening-{i}",
        first_public_fingerprint=f"h2-public-{i}",
        score=score,
        outcome="victory" if score > 0.5 else "defeat",
        exit_hp=int(score * 70),
        scenario_id=scenario,
        deck_preset_id=deck,
        nontrivial_draw_count=3,
        controlled_draw_count=4,
        decisions=(SimpleNamespace(
            phase="nature:draw_card", selected_id=draw,
        ),),
    )


def test_h2_joint_reference_metrics_resolve_deck_and_enemy_effects():
    first = (
        _row(0, "shrinker-a10", "silent-starter-v1", 0.8, "a"),
        _row(1, "shrinker-a10", "silent-starter-v1", 0.7, "a"),
        _row(2, "nibbits-a10", "silent-poison-kit-v1", 0.3, "a"),
        _row(3, "nibbits-a10", "silent-poison-kit-v1", 0.4, "a"),
    )
    second = tuple(
        replace(e, score=e.score - 0.2, exit_hp=e.exit_hp - 10,
                decisions=(SimpleNamespace(
                    phase="nature:draw_card", selected_id="b",
                ),))
        for e in first
    )
    report = reference_by_configuration([first, second])
    assert len(report) == 2
    key = "nibbits-a10::silent-poison-kit-v1"
    assert report[key]["source_count"] == 2
    assert report[key]["combat_count"] == 4
    assert report[key]["nontrivial_draws_per_combat"] == 3
    assert report[key]["reference_draw_changed_pair_fraction"] == 1
    assert report[key]["score_changed_pair_fraction"] == 1
    assert report[key]["mean_abs_draw_replicate_score_delta"] == pytest.approx(0.2)
    assert len(report[key]["mean_score_source_bootstrap_95pct"]) == 2
    invalid = (replace(second[0], deck_preset_id="silent-poison-kit-v1"), *second[1:])
    with pytest.raises(ValueError, match="do not match"):
        reference_by_configuration([first, invalid])


def test_h2_paired_holdout_reports_joint_cells_and_source_intervals():
    nature = [
        replace(episode("s0", mode="learned", score=0.4),
                scenario_id="shrinker-a10",
                deck_preset_id="silent-starter-v1", exit_hp=39),
        replace(episode("s1", mode="learned", score=0.2),
                scenario_id="nibbits-a10",
                deck_preset_id="silent-poison-kit-v1", exit_hp=25),
    ]
    reference = [
        replace(episode("s0", mode="reference", score=0.7),
                scenario_id="shrinker-a10",
                deck_preset_id="silent-starter-v1", exit_hp=50),
        replace(episode("s1", mode="reference", score=0.8),
                scenario_id="nibbits-a10",
                deck_preset_id="silent-poison-kit-v1", exit_hp=60),
    ]
    result = paired_holdout_metrics(
        nature, reference, bootstrap_seed=101, replicates=100,
    )
    cells = result["paired_by_configuration"]
    assert len(cells) == 2
    poison = cells["nibbits-a10::silent-poison-kit-v1"]
    assert poison["source_count"] == 1
    assert poison["mean_nature_minus_reference_score"] == pytest.approx(-0.6)
    assert poison["paired_mean_exit_hp_delta"] == -35
    assert poison["paired_source_bootstrap_95pct"] == pytest.approx([-0.6, -0.6])
    assert "draw_kl_q_to_p" in poison
