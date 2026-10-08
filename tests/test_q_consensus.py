"""No Q-consensus target is invented from unmatched or unstable teacher roots."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sts2_ai.training.export import load_training_jsonl, write_training_jsonl
from sts2_ai.training.q_consensus import ConsensusConfig, distill_consensus_targets
from sts2_ai.training.q_targets import QTeacherConfig
from sts2_ai.training.targets import PolicyTarget, TrainingExample


def _root(
    budget: int,
    values: tuple[float, float] = (0.5, 0.1),
    *,
    visits: tuple[int, int] | None = None,
) -> TrainingExample:
    if visits is None:
        visits = (budget // 2, budget // 2)
    return TrainingExample(
        observation_hash="obs",
        source_state_hash="exact",
        information_policy="fair",
        search_regime="oracle-exact",
        search_budget=budget,
        search_version=f"uct-{budget}",
        game_build="test-build",
        emulator_revision="test-emu",
        source_search_id=f"search-{budget}",
        source_run_seeds=("run-7",),
        observation_json='{"phase":4,"hp":53,"max_hp":70}',
        value_target=max(values),
        policy_targets=(
            PolicyTarget(
                action_id="rest",
                action_kind="rest_heal",
                probability=visits[0] / max(1, sum(visits)),
                visits=visits[0],
                search_value=values[0],
            ),
            PolicyTarget(
                action_id="upgrade",
                action_kind="rest_upgrade",
                probability=visits[1] / max(1, sum(visits)),
                visits=visits[1],
                search_value=values[1],
            ),
        ),
    )


def _config() -> ConsensusConfig:
    return ConsensusConfig(
        high_q=QTeacherConfig(
            min_budget=64, min_semantic_visits=2,
            min_value_gap=0.02, uncertainty_scale=0.0, temperature=0.15
        ),
        min_low_semantic_visits=1,
        min_low_value_gap=0.01,
    )


def test_matching_winners_create_action_q_targets_without_changing_source() -> None:
    low, high = _root(12), _root(96)
    output, report = distill_consensus_targets((low,), (high,), _config())
    assert report["matched_exact_roots"] == 1
    assert report["comparable_resolved_roots"] == 1
    assert report["q_winner_agreement_fraction"] == 1.0
    assert report["accepted"] == 1
    assert report["source_run_seeds"] == ["run-7"]
    assert low.policy_target_mode == high.policy_target_mode == "uct-visits-v1"
    (target,) = output
    assert target.source_run_seeds == ("run-7",)
    assert target.policy_target_mode.startswith("cross-budget-q-consensus-v1:")
    assert target.policy_targets[0].probability > 0.9
    assert sum(t.probability for t in target.policy_targets) == pytest.approx(1.0)


def test_flip_low_tie_missing_exploration_and_small_high_gap_are_rejected() -> None:
    high = _root(96)
    reversed_low = _root(12, values=(0.1, 0.8))
    examples, report = distill_consensus_targets((reversed_low,), (high,), _config())
    assert not examples
    assert report["reason_counts"]["q-winner-disagreement"] == 1

    tied, info = distill_consensus_targets(
        (_root(12, values=(0.2, 0.2)),), (high,), _config()
    )
    assert not tied
    assert info["reason_counts"]["ambiguous-low-q"] == 1

    unseen, info = distill_consensus_targets(
        (_root(12, visits=(12, 0)),), (high,), _config()
    )
    assert not unseen
    assert info["reason_counts"]["underexplored-low-q"] == 1

    # The low budget agrees, but the high sampled means are not well separated.
    close_low, info = distill_consensus_targets(
        (_root(12, values=(0.3, 0.1)),),
        (_root(96, values=(0.3001, 0.299)),),
        _config(),
    )
    assert not close_low
    assert info["reason_counts"]["high-indistinguishable-action-values"] == 1


def test_exact_state_validation_and_budget_order_are_enforced() -> None:
    low, high = _root(12), _root(96)
    with pytest.raises(ValueError, match="No identical"):
        distill_consensus_targets(
            (low,), (replace(high, source_state_hash="other"),), _config()
        )
    with pytest.raises(ValueError, match="conflicting fair observation"):
        distill_consensus_targets(
            (low,), (replace(high, observation_hash="changed"),), _config()
        )
    with pytest.raises(ValueError, match="conflicting legal actions"):
        distill_consensus_targets(
            (low,),
            (replace(high, policy_targets=(
                replace(high.policy_targets[0], action_id="other"),
                high.policy_targets[1],
            )),),
            _config(),
        )
    with pytest.raises(ValueError, match="strictly below"):
        distill_consensus_targets((high,), (low,), _config())
    with pytest.raises(ValueError, match="Duplicate exact root"):
        distill_consensus_targets((low, low), (high,), _config())


def test_q_consensus_cli_roundtrip(tmp_path: Path) -> None:
    low, high = tmp_path / "low.jsonl", tmp_path / "high.jsonl"
    output, report_path = tmp_path / "consensus.jsonl", tmp_path / "report.json"
    write_training_jsonl((_root(12),), low)
    write_training_jsonl((_root(96),), high)
    result = subprocess.run([
        sys.executable, "-m", "sts2_ai.cli",
        "distill-consensus-q-teacher", str(low), str(high), str(output),
        "--min-value-gap", "0.02", "--uncertainty-scale", "0",
        "--json-output", str(report_path),
    ], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["accepted"] == 1
    assert json.loads(report_path.read_text())["matched_exact_roots"] == 1
    assert len(load_training_jsonl(output)) == 1


def test_reject_invalid_low_agreement_constraints() -> None:
    with pytest.raises(ValueError, match="positive"):
        ConsensusConfig(min_low_semantic_visits=0)
    with pytest.raises(ValueError, match="gap"):
        ConsensusConfig(min_low_value_gap=float("nan"))
