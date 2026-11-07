"""Read-only H26 long-run preflight and monitoring contract tests."""
from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest
from test_h26_damage_risk import _catalog_v3, _state
from test_h26_searchfree import _models

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import KINDS
from sts2_ai.training.h21_macro_offer_codec import H21_OBJECT_LIMIT
from sts2_ai.training.h26_tactical import attach_visible_threat
from tools import train_h26_searchfree_a0 as trainer
from tools.monitor_h26 import render_h26


def test_require_forecast_features_prevents_incomplete_long_run(
    monkeypatch, tmp_path, capsys,
):
    h22 = tmp_path / "h22.pt"
    h22.write_bytes(b"checkpoint-only-for-fingerprint")
    emu = tmp_path / "emulator"
    emu.mkdir()
    root = tmp_path / "output"
    base = [
        "train_h26_searchfree_a0.py", "--initial-h22", str(h22),
        "--emulator-root", str(emu), "--emulator-revision", "test",
        "--output-root", str(root), "--require-forecast-features",
    ]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(ValueError, match="requires a v3"):
        trainer.main()
    catalog_path = tmp_path / "forecast.json"
    catalog_path.write_text(json.dumps(_catalog_v3()))
    monkeypatch.setattr(sys, "argv", base + [
        "--intent-catalog", str(catalog_path), "--disable-next-damage-risk",
    ])
    with pytest.raises(ValueError, match="requires a v3"):
        trainer.main()
    monkeypatch.setattr(sys, "argv", base + [
        "--intent-catalog", str(catalog_path),
        "--rounds", "200", "--max-wall-seconds", "21600",
    ])
    trainer.main()  # dry-run only: no --execute, no training
    output = json.loads(capsys.readouterr().out)
    assert output["execute"] is False
    assert output["experiment"]["next_damage_risk_enabled"] is True
    assert len(output["experiment"]["ai_source_sha256"]) == 64
    assert not (root / "progress.json").exists()


def test_readonly_monitor_handles_empty_and_completed_progress(tmp_path):
    root = tmp_path / "pilot"
    assert "No committed round yet" in render_h26(root, now=1000)
    root.mkdir()
    progress = {
        "experiment": {
            "next_damage_risk_enabled": True,
            "intent_catalog_sha256": "abc",
            "ascension": 0, "workers": 15,
            "runs_per_round": 64, "heldout_runs": 16,
        },
        "elapsed_wall_seconds": 3600,
        "wall_budget_seconds": 21600,
        "rounds": [{
            "round": 1,
            "train": {"cleared": 2, "completed": 64, "decisions": 16000},
            "heldout": {
                "cleared": 1, "completed": 16,
                "floor_progress": {"mean_max_floor_reached": 12.5},
                "mean_terminal_hp_fraction": 0.03,
            },
            "combat_search": {"roots": 0, "transitions": 0},
            "combat_update": {"examples": 4096, "steps": 512, "loss": 0.04},
            "macro_update": {"steps": 87},
            "timing": {"collection_seconds": 500},
            "intent_law_coverage": {
                "heldout": {
                    "living_availability_rate": 0.98,
                    "next_turn_support_audit": {"out_of_support": 0},
                },
            },
            "damage_risk_coverage": {
                "heldout": {
                    "availability_rate": 0.96,
                    "current_hud_preview_parity": {"mismatched": 0},
                },
            },
        }],
    }
    before = json.dumps(progress)
    (root / "progress.json").write_text(before)
    (root / "live.json").write_text(json.dumps({
        "stage": "collection", "round": 2, "timestamp_unix": 970,
        "completed_episodes": 5, "total_episodes": 64,
    }))
    view = render_h26(root, now=1000)
    assert "6.00h" in view
    assert "98%" in view and "96%" in view
    assert "12.5" in view and "32.0" in view
    assert "Current: round 2" in view
    assert (root / "progress.json").read_text() == before


def test_h26_tactical_never_truncates_an_enemy_intent_distribution():
    catalog = _catalog_v3()
    state = _state()
    actions = (LegalAction("wait", "end_turn", "{}"),)
    macro, _ = _models()
    original = macro.encode_decision(state, actions)
    filler = next(
        o for o in original.objects if o.kind != KINDS.index("enemy")
    )
    assert len(original.objects) < H21_OBJECT_LIMIT - 5
    nearly_full = replace(
        original, objects=original.objects + (
            filler,
        ) * (H21_OBJECT_LIMIT - len(original.objects) - 1),
    )
    enriched = attach_visible_threat(
        nearly_full, state, intent_catalog=catalog,
        damage_ascension=0, intent_histories={2: ()},
    )
    assert len(enriched.objects) == H21_OBJECT_LIMIT
    assert enriched.relations == original.relations
    full = replace(
        original, objects=original.objects + (
            filler,
        ) * (H21_OBJECT_LIMIT - len(original.objects)),
    )
    assert attach_visible_threat(
        full, state, intent_catalog=catalog, damage_ascension=0,
    ) == full
