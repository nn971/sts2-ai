"""Historical v8/v13 JSON compatibility and source-matched player comparisons."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from sts2_ai.emulator.semantic_p0_frozen_player import (
    FrozenPlayerSpec,
    FrozenPolicyInferenceError,
)
from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    TACTICAL_ATTENTION_FORMAT,
    TACTICAL_INSTANCES_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from tools.benchmark_nature_h1_frozen_players import (
    bootstrap_ci,
    candidate_argument,
    compare_source_matched_scores,
    source_seeds,
    summarize_reference_cohorts,
)


def _head(fmt: str, *, dimension: int = 32, hidden: int = 8):
    raw = {
        "format": fmt, "dimension": dimension, "hidden": hidden,
        "model_id": "synthetic-fixture",
        "state_weight": [[0.0] * dimension for _ in range(hidden)],
        "state_bias": [0.0] * hidden,
        "action_weight": [[0.0] * dimension for _ in range(hidden)],
        "action_bias": [0.0] * hidden,
        "policy_weight": [0.0] * hidden, "policy_bias": 0.0,
        "value_weight": [0.0] * hidden, "value_bias": 0.0,
    }
    if fmt in (TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT):
        raw.update(
            enemy_weight=[[0.0] * dimension for _ in range(hidden)],
            enemy_bias=[0.0] * hidden,
            enemy_context_weight=[0.0] * hidden,
            enemy_target_weight=[0.0] * hidden,
        )
    if fmt == TACTICAL_ATTENTION_FORMAT:
        raw.update(
            attn_query=[[0.0] * hidden for _ in range(hidden)],
            attn_key=[[0.0] * hidden for _ in range(hidden)],
            attn_value=[[0.0] * hidden for _ in range(hidden)],
            attn_output=[[0.0] * hidden for _ in range(hidden)],
            attn_gate=[0.0] * hidden,
        )
    return NeuralPolicyValueModel.from_dict(raw)


@pytest.mark.parametrize("fmt", [TACTICAL_INSTANCES_FORMAT, TACTICAL_ATTENTION_FORMAT])
@pytest.mark.parametrize("objective", ["continuation", "hp_preservation"])
def test_old_portable_phase_split_weights_load_frozen(
    fmt: str, objective: str, tmp_path: Path,
):
    model = PhaseSplitNeuralModel(
        strategy=_head(NEURAL_FORMAT), combat=_head(fmt),
        model_id="fixture-old-combat-policy",
        combat_value_objective=objective,
    )
    path = tmp_path / "old-model.json"
    model.save(path)
    frozen = FrozenPlayerSpec.from_file(path, seed=13)
    loaded = frozen.load_pinned()
    assert loaded.combat.format_id == fmt
    assert loaded.combat_value_objective == objective
    assert loaded.model_id == model.model_id
    assert frozen.policy_id.startswith("frozen-v15-greedy")
    # SHA pin and original value target survive serialization.
    path.write_text(path.read_text().replace(
        "fixture-old-combat-policy", "changed-but-valid-id"
    ))
    with pytest.raises(FrozenPolicyInferenceError, match="digest mismatch"):
        frozen.load_pinned()


@dataclass(frozen=True)
class Episode:
    source_seed: str
    reference_seed: int
    source_opening_fingerprint: str
    first_public_fingerprint: str
    score: float
    outcome: str
    exit_hp: int
    nontrivial_draw_count: int = 2
    controlled_draw_count: int = 3
    reference_intent_count: int = 0
    draw_mode: str = "reference"
    terminal_replay_verified: bool = True
    decisions: tuple = ()


def _cohort(scores, *, seed=500):
    return tuple(
        Episode(
            source_seed=f"source-{i}",
            reference_seed=seed + i * 1000003,
            source_opening_fingerprint=f"opening-{i}",
            first_public_fingerprint=f"public-{i}",
            score=score, outcome="victory" if score > 0 else "defeat",
            exit_hp=round(40 * score),
        )
        for i, score in enumerate(scores)
    )


def test_disjoint_source_splits_and_safe_candidate_labels():
    assert set(source_seeds("selection", 5)).isdisjoint(
        source_seeds("confirmation", 5)
    )
    assert not any(
        x.startswith("nature-draw-h1-eval-") or
        x.startswith("nature-draw-h1-train-")
        for x in source_seeds("selection", 5)
    )
    assert candidate_argument("v8=/tmp/one.json")[0] == "v8"
    for bad in ("../v8=/a.json", "v8", "v8=", "=something", "a/b=c"):
        with pytest.raises(argparse.ArgumentTypeError):
            candidate_argument(bad)


def test_reference_model_comparison_and_source_cluster_confidence():
    left = [_cohort((0.8, 0.9, 0.5)), _cohort((0.6, 0.7, 0.4), seed=1000)]
    right = [_cohort((0.4, 0.5, 0.2)), _cohort((0.3, 0.4, 0.1), seed=1000)]
    summary = summarize_reference_cohorts(left)
    assert summary["combat_count"] == 6
    assert summary["nontrivial_draws_per_combat"] == 2
    assert summary["mean_abs_draw_replicate_score_delta"] > 0
    assert summary["score_changed_pair_fraction"] == 1
    delta = compare_source_matched_scores(left, right)
    assert delta["mean_score_delta_first_minus_second"] > 0
    assert delta["source_count"] == 3
    assert len(delta["paired_source_bootstrap_95pct"]) == 2
    assert bootstrap_ci((0.5, 0.5, 0.5), 5) == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("override", [
    {"source_seed": "wrong"},
    {"reference_seed": 99},
    {"source_opening_fingerprint": "different"},
    {"first_public_fingerprint": "different"},
])
def test_paired_comparison_refuses_incompatible_sources(override):
    left = [_cohort((0.8, 0.9)), _cohort((0.7, 0.5), seed=1000)]
    right = [_cohort((0.5, 0.4)), _cohort((0.3, 0.3), seed=1000)]
    right[0] = (replace(right[0][0], **override), right[0][1])
    with pytest.raises(ValueError):
        compare_source_matched_scores(left, right)


def test_summary_refuses_wrong_replay_provenance():
    first = _cohort((0.8, 0.4))
    invalid = (replace(first[0], terminal_replay_verified=False), first[1])
    with pytest.raises(ValueError):
        summarize_reference_cohorts([invalid])
