"""Scientific H1 source partitions, paired evaluation and fail-closed guards."""
from __future__ import annotations

from dataclasses import replace

import pytest

from sts2_ai.nature.draw_only_episode import DrawOnlyEpisode
from tools.train_draw_only_nature_h1 import (
    paired_holdout_metrics,
    source_seed,
)


def episode(seed, *, mode, score, fingerprint="opening"):
    return DrawOnlyEpisode(
        schema="sts2-ai-nature-draw-only-law-router-v1",
        emulator_revision="pinned-emulator",
        source_seed=seed,
        reference_seed=100,
        draw_mode=mode,
        player_model_id="frozen-v15-pinned",
        nature_revision=0,
        source_opening_fingerprint=fingerprint,
        first_public_fingerprint="initial-public",
        decisions=(),
        nature_choices=(),
        reference_intent_count=2,
        controlled_draw_count=3,
        nontrivial_draw_count=2,
        outcome="victory" if score else "defeat",
        exit_hp=30 if score else 0,
        score=score,
        terminal_fingerprint="end",
        terminal_replay_verified=True,
    )


def test_disjoint_source_namespace_and_cross_replica_eval_immutability():
    assert source_seed("train", 0, training_seed=29) != source_seed(
        "train", 0, training_seed=101
    )
    assert source_seed("evaluation", 0, training_seed=29) == source_seed(
        "evaluation", 0, training_seed=101
    )
    assert source_seed("evaluation", 0, training_seed=29) != source_seed(
        "train", 0, training_seed=29
    )
    with pytest.raises(ValueError):
        source_seed("development", 0, training_seed=29)
    with pytest.raises(ValueError):
        source_seed("train", -1, training_seed=29)


def test_reference_vs_learned_holdout_source_bootstrap():
    sources = [f"eval-{x}" for x in range(8)]
    left = [episode(x, mode="learned", score=0.5) for x in sources]
    right = [episode(x, mode="reference", score=1.0) for x in sources]
    a = paired_holdout_metrics(left, right, bootstrap_seed=5, replicates=200)
    b = paired_holdout_metrics(left, right, bootstrap_seed=5, replicates=200)
    assert a == b
    assert a["paired_nature_minus_reference_score"] == pytest.approx(-0.5)
    assert a["paired_source_bootstrap_95pct"] == pytest.approx([-0.5, -0.5])
    assert a["reference_nontrivial_draws"] == 16
    assert a["nature_nontrivial_draws"] == 16
    assert a["training_seed_uncertainty"] == "NOT_ESTIMATED_WITHIN_SINGLE_RESTART"
    assert a["decision_grade"] is False
    assert a["chance_probability_status"] == "UNVERIFIED"


@pytest.mark.parametrize("field,bad", [
    ("source_seed", "wrong"),
    ("source_opening_fingerprint", "wrong"),
    ("reference_seed", 777),
    ("player_model_id", "other-model"),
    ("draw_mode", "learned"),
])
def test_paired_holdout_refuses_changed_baseline(field, bad):
    left = [episode("a", mode="learned", score=0.5)]
    right = [episode("a", mode="reference", score=1.0)]
    right = [replace(right[0], **{field: bad})]
    with pytest.raises(ValueError, match="differ"):
        paired_holdout_metrics(left, right, bootstrap_seed=5, replicates=100)
