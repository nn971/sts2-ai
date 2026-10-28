"""Focused contract tests; the native emulator replay remains a local acceptance gate."""
from __future__ import annotations

import random

import pytest
import torch

from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation.combat_snapshots import (
    SCHEMA, CombatSnapshotRecipe, progress_band,
)
from sts2_ai.training.h10_native_transformer import (
    NATIVE_REWARD, build_gae, choose_sampled, native_reward, select_native,
    verify_corpora,
)


def recipe(seed: str, identity: str, *, tier: str = "normal",
           floor: int = 4) -> CombatSnapshotRecipe:
    return CombatSnapshotRecipe(
        schema=SCHEMA, emulator_revision="revision-test",
        environment=NATIVE_OVERGROWTH,
        source_policy="frozen-hybrid-collector-test", seed=seed, ascension=10,
        action_ids=("a", "b", "c"), observation_hash="public-" + identity,
        exact_hash="exact-" + identity, act=1, floor=floor,
        progress=progress_band(floor), tier=tier, enemy_ids=("proto.enemy.nibbit",),
        hp=50, max_hp=70, deck_size=11, relic_ids=(), potion_ids=(),
    )


def test_total_nonminion_reward_is_shared_with_h9():
    assert NATIVE_REWARD == "encounter_nonminion_total_hp_v1"
    assert native_reward(
        "defeat", 0, 70, total_non_minion_hp=146,
        remaining_non_minion_hp=25,
    ) == pytest.approx(121 / 146)
    assert native_reward("victory", 42, 70) == pytest.approx(1.6)
    with pytest.raises(ValueError, match="authoritative"):
        native_reward("defeat", 0, 70)


def test_sampled_actor_uses_exact_temperature_and_frozen_logits():
    logits = (1.0, -3.0, 0.5)
    rng = random.Random(7)
    samples = [choose_sampled(logits, temperature=0.85, rng=rng) for _ in range(64)]
    logps = torch.log_softmax(torch.tensor(logits) / 0.85, dim=0)
    assert {s[0] for s in samples} <= {0, 1, 2}
    for chosen, logp in samples:
        assert logp == pytest.approx(float(logps[chosen]), abs=1e-6)
    rng2 = random.Random(7)
    assert samples == [choose_sampled(
        logits, temperature=0.85, rng=rng2,
    ) for _ in range(64)]


def test_native_returns_single_terminal_reward_and_gae():
    entries = [(object(), 0, -0.3, 0.15), (object(), 1, -0.5, 0.25)]
    result = build_gae(entries, reward=1.5, gamma=1.0, gae_lambda=1.0)
    assert len(result) == 2
    assert result[1].advantage == pytest.approx(0.75 - 0.25)
    assert result[0].advantage == pytest.approx(0.75 - 0.15)
    assert result[0].target == pytest.approx(0.75)
    assert result[1].target == pytest.approx(0.75)


def test_native_holdout_has_independent_run_seeds_and_revision():
    train = (recipe("train-1", "01"), recipe("train-1", "02"),
             recipe("train-2", "03"))
    holdout = (recipe("hold-1", "04"), recipe("hold-2", "05"))
    verify_corpora(train, holdout, emulator_revision="revision-test")
    with pytest.raises(ValueError, match="disjoint"):
        verify_corpora(train, (recipe("train-1", "09"),),
                       emulator_revision="revision-test")
    with pytest.raises(ValueError, match="revision"):
        verify_corpora(train, holdout, emulator_revision="different")
    with pytest.raises(ValueError, match="uniqueness"):
        verify_corpora(train + train[:1], holdout,
                       emulator_revision="revision-test")


def test_native_balanced_sampling_is_stable_and_reproducible():
    corpus = (
        recipe("normal-1", "01"),
        recipe("normal-2", "02"),
        recipe("elite-1", "03", tier="elite", floor=9),
        recipe("boss-1", "04", tier="boss", floor=16),
    )
    selected = select_native(corpus, count=80, study_seed=17, round_index=0)
    assert selected == select_native(
        corpus, count=80, study_seed=17, round_index=0)
    assert len(selected) == 80
    assert sum(x.tier == "boss" for x in selected) == 80 // 3
    assert selected != select_native(
        corpus, count=80, study_seed=17, round_index=1)
