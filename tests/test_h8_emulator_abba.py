"""Report-only contract tests for matched full-Transformer H8 A-B-B-A.

These run without CUDA, the .NET SDK, or trained checkpoints.
"""
from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_h8_emulator_abba import (  # noqa: E402
    AFTER_REV, BEFORE_REV, validate_and_compare,
)


def _report(arm: str, seconds: float) -> dict:
    return {
        "schema": "sts2-h8-h7-reference-ppo-benchmark-v1",
        "model_sha256": "a" * 64,
        "model_parameters": 14659972,
        "emulator_revision": BEFORE_REV if arm == "A" else AFTER_REV,
        "emulator_dll_sha256": ("b" if arm == "A" else "c") * 64,
        "source_digest": "d" * 64,
        "reference_law": "h7-event-scoped-declared-chance-law",
        "source_count": 16,
        "workers": 15,
        "max_boundaries": 1200,
        "replay_every": 8,
        "terminal_counts": {"terminal": 16},
        "score_mean": 0.7576398806873161,
        "victories": 6,
        "defeats": 10,
        "native_h8_rollout_seconds": seconds,
        "native_h8_player_decisions_per_second": 100,
        "native_h8_actor_metrics": {
            "requests": 320, "batches": 76,
            "forward_seconds": seconds * .25,
            "collate_seconds": seconds * .2,
        },
        "legacy_h7": {
            "matching_source_trajectories": 16,
            "player_decisions": 320,
            "wall_seconds": seconds * 1.6,
        },
        "ppo_seconds": seconds * .7,
        "total_h8_iteration_seconds": seconds * 1.7,
        "verified_terminal_replays": 2,
        "ppo": {"source_decisions": 320, "updates": 10},
    }


def reports() -> dict[str, dict]:
    return {
        "A1": _report("A", 32),
        "B1": _report("B", 24),
        "B2": _report("B", 28),
        "A2": _report("A", 36),
    }


def test_abba_uses_mirrored_pairs_and_confirms_full_parity():
    result = validate_and_compare(reports(), BEFORE_REV, AFTER_REV, 16, 8)
    comparison = result["rollout_comparison"]
    assert comparison["adjacent_pair_speedups"] == [
        32 / 24, 36 / 28]
    assert comparison["geometric_mean_speedup"] > 1.2
    assert result["legacy_h7_trajectory_matches_per_run"] == 16
    assert result["verified_terminal_replays_per_run"] == 2
    assert result["terminal_outcomes"] == {
        "victories": 6, "defeats": 10,
        "score_mean": 0.7576398806873161}
    assert result["target_reached_in_both_candidate_runs"] is False
    assert comparison["baseline_drift_A2_over_A1"] == 36 / 32
    assert comparison["candidate_drift_B2_over_B1"] == 28 / 24


@pytest.mark.parametrize("path,value,match", [
    (("B2", "source_digest"), "other", "source_digest"),
    (("B2", "model_sha256"), "other", "model_sha256"),
    (("A2", "emulator_revision"), AFTER_REV, "revision"),
    (("B1", "emulator_dll_sha256"), "b" * 64, "DLL"),
    (("B1", "terminal_counts"), {"terminal": 15, "capped": 1}, "censored"),
    (("A2", "verified_terminal_replays"), 1, "replay"),
    (("A1", "legacy_h7"), None, "legacy"),
    (("B1", "victories"), 7, "outcomes"),
    (("B1", "native_h8_actor_metrics", "requests"), 319, "decision"),
    (("B2", "ppo", "source_decisions"), 319, "source decision"),
    (("A2", "ppo", "updates"), 9, "update"),
])
def test_abba_rejects_mismatched_conditions(path, value, match):
    data = deepcopy(reports())
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        validate_and_compare(data, BEFORE_REV, AFTER_REV, 16, 8)


def test_abba_rejects_rebuilt_arm_between_paired_runs():
    data = reports()
    data["A2"]["emulator_dll_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="DLL"):
        validate_and_compare(data, BEFORE_REV, AFTER_REV, 16, 8)


def test_report_only_writes_summary_without_dotnet_or_cuda(tmp_path):
    for name, report in reports().items():
        (tmp_path / f"{name}.json").write_text(json.dumps(report))
    completed = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parents[1] /
             "tools/benchmark_h8_emulator_abba.py"),
         "--emulator-repo", str(tmp_path / "nonexistent"),
         "--output-root", str(tmp_path),
         "--episodes", "16", "--report-only"],
        text=True, capture_output=True, check=True)
    assert "rollout speedup" in completed.stdout
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["runs"]["B2"]["complete_combats_per_second"] == 16 / 28
