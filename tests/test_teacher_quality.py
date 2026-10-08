"""Teacher signal diagnostics: duplicated actions are not distinct strategy."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sts2_ai.evaluation.experiment_report import paired_evaluation_report
from sts2_ai.training.export import load_training_jsonl, write_training_jsonl
from sts2_ai.training.targets import PolicyTarget, TrainingExample
from sts2_ai.training.teacher_quality import (
    TeacherFilter,
    curate_teacher_examples,
    grade_teacher_root,
    paired_teacher_quality_report,
    teacher_quality_report,
)


def _example(
    *,
    visits: tuple[int, int, int] = (7, 7, 6),
    probabilities: tuple[float, float, float] = (0.35, 0.35, 0.30),
    values: tuple[float, float, float] = (0.4, 0.4, 0.1),
    budget: int = 64,
) -> TrainingExample:
    state = {
        "phase": 3,
        "combat": {
            "hand": [
                {"instance_id": 1, "card_id": "STRIKE", "cost": 1},
                {"instance_id": 2, "card_id": "STRIKE", "cost": 1},
                {"instance_id": 3, "card_id": "DEFEND", "cost": 1},
            ],
        },
    }
    return TrainingExample(
        observation_hash="obs-map",
        information_policy="fair",
        policy_targets=tuple(
            PolicyTarget(
                action_id=f"card-{node_id}",
                action_kind="play_card",
                action_payload_json=json.dumps({"card_instance_id": node_id}),
                probability=probabilities[i],
                visits=visits[i],
                search_value=values[i],
            )
            for i, node_id in enumerate((1, 2, 3))
        ),
        value_target=0.4,
        source_search_id="search-map",
        emulator_revision="emulator",
        observation_json=json.dumps(state),
        source_state_hash="exact",
        search_budget=budget,
    )


def test_semantic_aggregation_finds_signal_hidden_by_duplicates() -> None:
    example = _example()
    quality = grade_teacher_root(example)
    assert quality.semantic_actions == 2
    assert quality.legal_actions == 3
    assert quality.literal_normalized_entropy > 0.99
    assert quality.semantic_normalized_entropy < 0.89
    assert quality.semantic_top_probability == pytest.approx(0.70)
    assert quality.semantic_top_margin == pytest.approx(0.20)
    assert quality.best_semantic_value_gap == pytest.approx(0.30)
    assert quality.semantic_min_visits == 6
    assert quality.value_visit_agreement is True
    assert quality.eligible
    report = teacher_quality_report((example,))
    assert report["eligible_examples"] == 1
    assert report["mean_semantic_actions"] == 2.0


def test_teacher_gate_rejects_flat_underexplored_and_conflicting_roots() -> None:
    sample = _example()
    flat = _example(probabilities=(0.25, 0.25, 0.50))
    underexplored = _example(visits=(0, 0, 20))
    conflict = _example(values=(0.1, 0.1, 0.8))
    low_budget = _example(budget=12)
    assert "flat-semantic-policy" in grade_teacher_root(flat).reasons
    assert "underexplored-semantic-action" in grade_teacher_root(underexplored).reasons
    assert "visits-disagree-with-best-value" in grade_teacher_root(conflict).reasons
    assert "low-budget" in grade_teacher_root(low_budget).reasons
    assert curate_teacher_examples((sample, flat, underexplored, conflict, low_budget)) == (
        sample,
    )


def test_teacher_curator_preserves_original_targets(tmp_path: Path) -> None:
    example = _example()
    weak = replace(_example(budget=8), observation_hash="weak", source_search_id="weak")
    source = tmp_path / "roots.jsonl"
    write_training_jsonl((example, weak), source)
    output = tmp_path / "strong.jsonl"
    finished = subprocess.run(
        [
            sys.executable, "-m", "sts2_ai.cli", "curate-teacher",
            str(source), "--output", str(output), "--min-budget", "48",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(finished.stdout)["eligible_examples"] == 1
    assert load_training_jsonl(output) == (example,)
    assert output.read_text()  # The curator does not renormalize or sharpen visits.


def test_bad_filters_and_missing_actions_are_rejected() -> None:
    with pytest.raises(ValueError, match="budget"):
        TeacherFilter(min_budget=-1)
    with pytest.raises(ValueError, match="margin"):
        TeacherFilter(min_semantic_top_margin=1.0)
    with pytest.raises(ValueError, match="missing legal action"):
        grade_teacher_root(replace(_example(), policy_targets=()))



def test_paired_report_uses_exact_seeds_and_frontier(tmp_path: Path) -> None:
    def write(path: Path, offset: float, *, seed_suffix: str = "1") -> None:
        path.write_text(json.dumps({
            "agent": "MCTS-32",
            "rollout_depth": 16,
            "runs": [
                {
                    "seed": f"heldout-{seed_suffix}",
                    "frontier_progress": 2.0 + offset,
                    "wall_seconds": 3.0 + offset,
                    "agent_compute_seconds": 2.5 + offset,
                    "emulator_transitions": 200,
                    "decisions": 20,
                    "outcome": "truncated",
                },
            ],
        }))
    left, right = tmp_path / "baseline.json", tmp_path / "other.json"
    write(left, 0.0)
    write(right, 0.5)
    report = paired_evaluation_report(left, right)
    assert report["paired_mean_frontier_delta"] == pytest.approx(0.5)
    assert report["paired_ahead"] == 1
    assert report["paired_behind"] == 0
    assert report["contender"]["mean_wall_seconds"] == pytest.approx(3.5)
    write(right, 0.5, seed_suffix="different")
    with pytest.raises(ValueError, match="identical"):
        paired_evaluation_report(left, right)



def test_unresolved_event_choices_are_not_assumed_equivalent() -> None:
    example = replace(
        _example(),
        observation_json=json.dumps({"phase": 6, "event": {}}),
        policy_targets=(
            PolicyTarget(
                "event-0", 0.7, "event_choice", '{"index":0}', visits=12, search_value=0.5
            ),
            PolicyTarget("event-1", 0.3, "event_choice", '{"index":1}', visits=8, search_value=0.2),
        ),
    )
    grade = grade_teacher_root(example)
    assert grade.semantic_actions == 2
    assert grade.eligible



def test_equal_room_type_does_not_make_paths_equivalent() -> None:
    example = replace(
        _example(),
        observation_json=json.dumps({
            "phase": 2,
            "map": [
                {"node_id": "a", "room_type": 0},
                {"node_id": "b", "room_type": 0},
                {"node_id": "c", "room_type": 4},
            ],
        }),
        policy_targets=tuple(
            PolicyTarget(f"path-{i}", p, "choose_map_node",
                         json.dumps({"node_id": node_id}), visits=visits,
                         search_value=value)
            for i, (node_id, p, visits, value) in enumerate([
                ("a", 0.35, 7, 0.4),
                ("b", 0.35, 7, 0.4),
                ("c", 0.30, 6, 0.1),
            ])
        ),
    )
    quality = grade_teacher_root(example)
    assert quality.semantic_actions == 3
    assert quality.literal_normalized_entropy == pytest.approx(
        quality.semantic_normalized_entropy
    )



def test_exact_root_teacher_pair_requires_matching_root_state() -> None:
    before = _example(budget=12, probabilities=(0.25, 0.25, 0.50))
    after = _example(budget=96)
    report = paired_teacher_quality_report((before,), (after,))
    assert report["matched_exact_roots"] == 1
    assert report["mean_semantic_entropy_change_high_minus_low"] < 0.0
    assert report["mean_top_margin_change_high_minus_low"] > 0.0
    with pytest.raises(ValueError, match="no identical"):
        paired_teacher_quality_report(
            (before,), (replace(after, source_state_hash="other-exact"),)
        )
    with pytest.raises(ValueError, match="different fair observation"):
        paired_teacher_quality_report(
            (before,), (replace(after, observation_hash="changed"),)
        )


def test_matched_teacher_q_rank_stability_detects_flips_and_ties() -> None:
    low = _example(budget=12, values=(0.6, 0.6, 0.1))
    stable = _example(budget=96, values=(0.7, 0.7, 0.2))
    unchanged = paired_teacher_quality_report((low,), (stable,))
    assert unchanged["q_winner_resolved_both"] == 1
    assert unchanged["q_winner_agreement_fraction"] == 1.0
    assert unchanged["q_winner_flip_count"] == 0
    assert unchanged["mean_high_semantic_q_gap"] == pytest.approx(0.5)

    flipped = _example(budget=96, values=(0.1, 0.1, 0.8))
    changed = paired_teacher_quality_report((low,), (flipped,))
    assert changed["q_winner_agreement_fraction"] == 0.0
    assert changed["q_winner_flip_count"] == 1

    tied = _example(budget=96, values=(0.4, 0.4, 0.4))
    uncertain = paired_teacher_quality_report((low,), (tied,))
    assert uncertain["q_winner_resolved_both"] == 0
    assert uncertain["q_winner_agreement_fraction"] is None
    assert uncertain["q_winner_unresolved_high_given_resolved_low"] == 1
    assert uncertain["mean_high_semantic_q_gap"] == pytest.approx(0.0)

    underexplored = _example(budget=12, visits=(0, 0, 20))
    unresolved = paired_teacher_quality_report((underexplored,), (stable,))
    assert unresolved["q_winner_resolved_both"] == 0
    assert unresolved["mean_low_semantic_q_gap"] is None
