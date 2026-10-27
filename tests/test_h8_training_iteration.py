"""Source-valid H8 training/PPO acceptance gates (no emulator needed)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_h8_training_iteration import (  # noqa: E402
    validate_legacy_parity, validate_training_cohort,
)


def _row(seed: str) -> dict:
    actions = ({
        "source_support_identity": "support",
        "source_observation": {"hp": 66},
        "source_legal_actions": [{"actionId": "end_turn"}],
        "observation": {"policy_id": "prototype-fair-v0"},
        "legal_actions": [{"action_id": "end_turn"}],
        "chosen_index": 0, "selected_id": "end_turn",
        "behavior_model_id": "h7-test", "temperature": 0.85,
        "action_scope": "all-legal-v1",
        "behavior_logprob": -0.25, "behavior_value": 0.3,
    },)
    steps = [
        {"phase": "player", "selected_id": "end_turn"},
        {"phase": "nature:enemy_intent", "selected_id": "attack"},
    ]
    return {
        "seed": seed, "status": "terminal", "outcome": "victory",
        "exit_hp": 66, "terminal_enemy_hp": None, "score": 1.9,
        "replay_verified": True, "player_decisions": 1,
        "captured_player_decisions": 1,
        "player_steps": steps[:1], "decision_sequence": steps,
        "reference_event_digest": "abcdef",
        "reference_law_steps": [
            {"event_id": "event", "law_id": "law", "selected_id": "attack"},
        ],
        "training_actions": actions,
    }


def test_training_requires_all_episodes_terminal():
    seeds = ("a", "b")
    rows = {seed: _row(seed) for seed in seeds}
    good = {
        "episodes": 2, "statuses": {"terminal": 2}, "trajectories": rows,
    }
    episodes, captures = validate_training_cohort(
        good, seeds, replay_every=1)
    assert len(episodes) == len(captures) == 2
    assert episodes[0].draw_mode == "reference"
    rows["b"] = {**rows["b"], "status": "max_boundaries"}
    with pytest.raises(RuntimeError, match="Missing terminal"):
        validate_training_cohort(good, seeds, replay_every=1)
    with pytest.raises(RuntimeError, match="CENSORED"):
        validate_training_cohort(
            {**good, "statuses": {"terminal": 1, "max_boundaries": 1}},
            seeds, replay_every=1)


def test_training_audit_schedule_and_onpolicy_capture_are_mandatory():
    seeds = ("a", "b")
    rows = {seed: _row(seed) for seed in seeds}
    rows["b"]["replay_verified"] = False
    good = {
        "episodes": 2, "statuses": {"terminal": 2}, "trajectories": rows,
    }
    assert len(validate_training_cohort(good, seeds, replay_every=2)[0]) == 2
    with pytest.raises(RuntimeError, match="Missing terminal"):
        validate_training_cohort(good, seeds, replay_every=1)
    rows["a"]["training_actions"] = ()
    with pytest.raises(RuntimeError, match="Missing terminal"):
        validate_training_cohort(good, seeds, replay_every=2)


def test_legacy_comparator_rejects_law_and_action_drift():
    seed = "a"
    row = _row(seed)
    h8 = {"trajectories": {seed: row}}
    legacy = {
        "episodes": (SimpleNamespace(
            decisions=(
                SimpleNamespace(phase="player", selected_id="end_turn",
                                law_id=None),
                SimpleNamespace(phase="nature:enemy_intent",
                                selected_id="attack", law_id="law"),
            ),
            outcome="victory", exit_hp=66, score=1.9,
            terminal_enemy_hp=None,
        ),),
        "player_rollouts": (row["training_actions"],),
    }
    assert validate_legacy_parity((seed,), h8, legacy) == {
        "matching_source_trajectories": 1}
    row["reference_law_steps"][0]["law_id"] = "different"
    with pytest.raises(RuntimeError, match="Chance Law"):
        validate_legacy_parity((seed,), h8, legacy)
    row["reference_law_steps"][0]["law_id"] = "law"
    row["decision_sequence"][0]["selected_id"] = "different_action"
    with pytest.raises(RuntimeError, match="choice drift"):
        validate_legacy_parity((seed,), h8, legacy)
