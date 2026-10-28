"""Deterministic, no-emulator coverage of H9 progress-driven curriculum."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from sts2_ai.nature.adaptive_rich_curriculum import (
    ADAPTIVE_SCHEMA, FAMILIES, AdaptiveRichProgress,
    encounter_mean_rewards, probe_sources, training_pools,
    weighted_train_sources,
)
from sts2_ai.nature.research_combat_fixtures import fixture_from_source


def test_adaptive_training_pools_use_only_original_1024_fixtures():
    pools = training_pools(20261028)
    assert set(pools) == set(FAMILIES)
    assert all(pools.values())
    indices = [i for part in pools.values() for i in part]
    assert len(indices) == len(set(indices)) == 1024
    assert min(indices) == 0 and max(indices) == 1023


def test_fixed_probes_cover_each_encounter_without_holdout_leak():
    sources = probe_sources(study_seed=73127, curriculum_seed=20261028)
    assert sources == probe_sources(study_seed=73127, curriculum_seed=20261028)
    assert len(sources) == len(FAMILIES) * 6
    assert len(set(sources)) == len(sources)
    encounters = Counter()
    for seed in sources:
        scenario, _, idx = fixture_from_source(
            seed, curriculum_seed=20261028)
        assert 0 <= idx < 1024
        encounters[scenario.encounter_id] += 1
    assert set(encounters) == set(FAMILIES)
    assert all(value == 6 for value in encounters.values())


def test_controller_tracks_absolute_reward_change_with_uniform_floor():
    adaptive = AdaptiveRichProgress()
    initial = {name: 1.0 for name in FAMILIES}
    before = adaptive.update(initial)
    assert all(value == 0 for value in before["absolute_change"].values())
    assert max(adaptive.weights.values()) == pytest.approx(1 / len(FAMILIES))
    active = FAMILIES[4]
    next_scores = {**initial, active: 1.5}
    feedback = adaptive.update(next_scores)
    assert feedback["absolute_change"][active] == pytest.approx(0.5)
    assert feedback["smoothed_progress"][active] == pytest.approx(0.25)
    assert adaptive.weights[active] > adaptive.weights[FAMILIES[0]]
    assert all(w >= (1 - adaptive.beta) / len(FAMILIES)
               for w in adaptive.weights.values())
    assert sum(adaptive.weights.values()) == pytest.approx(1.0)
    # Regression also yields a positive progress scalar.
    reversed_feedback = adaptive.update(initial)
    assert reversed_feedback["absolute_change"][active] == pytest.approx(0.5)
    assert reversed_feedback["smoothed_progress"][active] > 0
    # Static/very-hard families eventually lose their special allocation.
    for _ in range(30):
        adaptive.update(initial)
    assert adaptive.weights[active] < feedback["weights"][active]
    assert adaptive.weights[active] > 0


def test_adaptive_sampler_is_deterministic_and_responds_to_feedback():
    adaptive = AdaptiveRichProgress()
    adaptive.update({name: 1.0 for name in FAMILIES})
    sampled = weighted_train_sources(
        256, round_index=0, study_seed=73127,
        curriculum_seed=20261028, weights=adaptive.weights)
    assert sampled == weighted_train_sources(
        256, round_index=0, study_seed=73127,
        curriculum_seed=20261028, weights=adaptive.weights)
    assert len(set(sampled)) == len(sampled)
    assert all(fixture_from_source(s, curriculum_seed=20261028)[2] < 1024
               for s in sampled)
    new_mean = {name: 1.0 for name in FAMILIES}
    new_mean[FAMILIES[-1]] = 0.1
    adaptive.update(new_mean)
    sampled2 = weighted_train_sources(
        4096, round_index=1, study_seed=73127,
        curriculum_seed=20261028, weights=adaptive.weights)
    counts = Counter(
        fixture_from_source(seed, curriculum_seed=20261028)[0].encounter_id
        for seed in sampled2)
    assert counts[FAMILIES[-1]] > counts[FAMILIES[0]]
    assert len(set(sampled2)) == 4096
    assert all(counts[name] > 0 for name in FAMILIES)


def test_probe_rewards_and_resume_state_are_exact_and_validated():
    seeds = probe_sources(study_seed=3, curriculum_seed=20261028, per_family=2)
    episodes = tuple(SimpleNamespace(
        source_seed=seed, outcome="victory", score=1.0
    ) for seed in seeds)
    rewards = encounter_mean_rewards(
        episodes, curriculum_seed=20261028, expected_per_family=2)
    assert rewards == {name: 1.0 for name in FAMILIES}
    with pytest.raises(ValueError, match="cover"):
        encounter_mean_rewards(
            episodes[:-1], curriculum_seed=20261028, expected_per_family=2)
    progress = AdaptiveRichProgress()
    progress.update(rewards)
    progress.update({**rewards, FAMILIES[-1]: 1.5})
    persisted = progress.to_dict()
    assert persisted["schema"] == ADAPTIVE_SCHEMA
    assert AdaptiveRichProgress.from_dict(persisted).to_dict() == persisted
    with pytest.raises(ValueError, match="schema"):
        AdaptiveRichProgress.from_dict({**persisted, "schema": "wrong"})
    with pytest.raises(ValueError, match="weights"):
        weighted_train_sources(
            1, round_index=0, study_seed=3,
            curriculum_seed=20261028,
            weights={**persisted["weights"], FAMILIES[-1]: 0.0})
