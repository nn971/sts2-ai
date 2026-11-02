"""Focused H16 current-HP versus learned continuation-value diagnostics."""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.training.h16_joint_act_pilot import PilotDecision, PilotEpisode
from tools.diagnose_h16_hp_value import (
    SCHEMA,
    correlation,
    from_episode,
    read_journal,
    summarize,
    svg_chart,
    within_floor_phase_correlation,
)


def sample(seed: str, hp: int, value: float, *,
           floor: int = 7, phase: str = "combat") -> dict:
    return {
        "seed": seed, "decision_index": 0, "floor": floor,
        "phase": phase, "hp": hp, "max_hp": 70,
        "hp_fraction": hp / 70, "predicted_value": value,
        "realized_training_target": 0.4,
    }


def test_correlation_and_floor_phase_adjustment():
    rows = [
        sample("a", 14, 0.1),
        sample("b", 28, 0.2),
        sample("c", 42, 0.3),
        sample("d", 56, 0.4),
    ]
    assert correlation(rows, "hp_fraction", "predicted_value") == pytest.approx(1.0)
    assert within_floor_phase_correlation(rows) == pytest.approx(1.0)
    rows[0]["predicted_value"] = None
    assert correlation(rows[:1], "hp_fraction", "predicted_value") is None
    rows[0]["predicted_value"] = 0.1
    # Floor and phase adjustments should eliminate between-group effects.
    mixed = [
        sample("a", 14, 0.8, floor=1),
        sample("b", 28, 0.8, floor=1),
        sample("c", 42, 0.2, floor=13),
        sample("d", 56, 0.2, floor=13),
    ]
    assert correlation(mixed, "hp_fraction", "predicted_value") < 0
    assert within_floor_phase_correlation(mixed) is None


def test_macro_realized_target_uses_training_geometric_return():
    action = LegalAction("a", "end_turn")
    obs = Observation(FAIR_POLICY_ID, "{}", "sha")
    decision = PilotDecision(
        obs, (action,), 0, (0.0,), predicted_value=0.65,
    )
    initial = {
        "floor": 0, "hp": 70, "max_hp": 70, "combat": None,
        "completed_rooms": [],
    }
    after = {
        "floor": 1, "hp": 49, "max_hp": 70, "combat": None,
        "completed_rooms": [{"act": 1, "floor": 1}],
    }
    ep = PilotEpisode(
        "seed-a", (decision,), (initial, after), "defeat", 0, 0,
    )
    points = from_episode(
        ep, horizon_lambda=0.5, combat_credit_weight=0.25,
        loss_progress_cap=0.2,
    )
    assert len(points) == 1
    assert points[0]["hp_fraction"] == 1.0
    assert points[0]["predicted_value"] == 0.65
    assert points[0]["realized_training_target"] == pytest.approx(0.35)


def test_phase_specific_summaries_svg_and_same_hp_spread():
    data = [sample(f"s{i}", 35, i / 8) for i in range(8)]
    data += [
        sample("other", 28, 0.2, phase="macro"),
        sample("other2", 56, 0.8, phase="macro"),
    ]
    result = summarize(data)
    assert result["all_decisions"]["count"] == 10
    assert result["all_decisions"]["hp_value_linear_r2"] is not None
    assert result["all_decisions"]["current_hp_baseline_target_mae"] is not None
    assert result["by_phase"]["macro"]["count"] == 2
    spread = result["same_hp_floor_phase_value_spread"][0]
    assert spread["count"] == 8
    assert spread["min_value"] == 0
    assert spread["max_value"] == pytest.approx(7 / 8)
    graphic = svg_chart(data)
    assert "<svg" in graphic and "Macro" in graphic and "Combat" in graphic
    assert "Learned continuation value" in graphic


def test_value_journal_identity_and_seed_order(tmp_path):
    path = tmp_path / "progress.partial.jsonl"
    identity = {"frozen": "sha"}
    row = {"seed": "a", "decisions": [sample("a", 42, 0.5)]}
    path.write_text(
        json.dumps({"schema": SCHEMA, "identity": identity}) + "\n"
        + json.dumps(row) + "\n", encoding="utf-8",
    )
    assert read_journal(path, identity, ["a", "b"]) == [row]
    with pytest.raises(ValueError, match="identity mismatch"):
        read_journal(path, {"frozen": "changed"}, ["a", "b"])
    with pytest.raises(ValueError, match="missing/reordered"):
        read_journal(path, identity, ["b", "a"])
