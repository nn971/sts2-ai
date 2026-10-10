"""Independent Act-1 selection/confirmation and partial-journal safety tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sts2_ai.evaluation.checkpoint_selection import (
    GOAL,
    JOURNAL_SCHEMA,
    analyze,
    exact_mcnemar,
    is_certified_clear,
    paired_comparison,
    summarize_model,
    wilson_interval,
)
from tools.evaluate_act1_checkpoints import (
    final_candidates,
    identity_for,
    load_journal,
)


def result(
    seed: str, won: bool, *, floor: int = 16,
    boss: dict | None = None,
) -> dict:
    return {
        "seed": seed, "outcome": "victory" if won else "defeat",
        "episode_goal_version": GOAL,
        "censored": False, "act1_cleared": won,
        "terminal_floor": floor, "frontier_progress": float(floor),
        "boss_progress": boss,
    }


def test_certified_goal_refuses_floor_guesses_and_censoring() -> None:
    assert not is_certified_clear(result("seed", False, floor=16))
    assert is_certified_clear(result("seed", True))
    for replacement in (
        {"act1_cleared": None}, {"act1_cleared": 1},
        {"episode_goal_version": "prototype-three-act-v0"},
        {"censored": True, "outcome": "truncated"},
    ):
        with pytest.raises(ValueError):
            is_certified_clear({**result("seed", False), **replacement})


def test_exact_paired_comparison_and_wilson_interval() -> None:
    # first-only wins=3, second-only=1, both=1, neither=1
    first = [result(f"s{i}", i in (0, 1, 2, 3)) for i in range(6)]
    second = [result(f"s{i}", i in (0, 4)) for i in range(6)]
    pair = paired_comparison(first, second)
    assert pair["both_win"] == 1
    assert pair["only_first_wins"] == 3
    assert pair["only_second_wins"] == 1
    assert pair["neither_win"] == 1
    assert pair["clear_rate_delta_first_minus_second"] == pytest.approx(2 / 6)
    assert pair["mcnemar_exact_two_sided_p"] == pytest.approx(0.625)
    assert exact_mcnemar(0, 0) == 1
    assert exact_mcnemar(0, 10) == pytest.approx(2 / 1024)
    assert wilson_interval(0, 10)[0] == 0
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="seed mismatch"):
        paired_comparison(first, second[::-1])


def test_predeclared_ranking_and_boss_summary() -> None:
    boss = {"initial_hp": 100, "remaining_hp": 10, "encounter_id": "kin"}
    baseline = [result("x", False, boss=boss), result("y", True)]
    better = [result("x", True), result("y", True)]
    stats = summarize_model(baseline)
    assert stats["boss_encounters"] == 1
    assert stats["near_misses_80pct_boss_hp_removed"] == 1
    selection = analyze(
        {"round40": baseline, "round60": better}, mode="selection",
        reference_label="round40",
    )
    assert selection["selected_label"] == "round60"
    assert selection["pairwise_vs_reference"]["round60"]["only_first_wins"] == 1
    # Select round40 on a win+progress exact tie.
    tie = analyze(
        {"round40": baseline, "round120": baseline}, mode="selection",
        reference_label="round40",
    )
    assert tie["selected_label"] == "round40"
    confirmation = analyze(
        {"round40": baseline, "round60": better}, mode="final",
        reference_label="round40",
    )
    assert confirmation["selected_label"] is None
    assert confirmation["ranking"] is None


def test_holdout_report_determines_exact_final_models() -> None:
    prior = {
        "schema": "sts2-certified-act1-checkpoint-evaluation-v1",
        "mode": "selection", "reference_label": "round40",
        "identity": {"model_specs": {
            "round40": {"path": "/a.json", "sha256": "one"},
            "round60": {"path": "/b.json", "sha256": "two"},
            "round100": {"path": "/c.json", "sha256": "three"},
        }},
        "analysis": {"selected_label": "round60"},
    }
    candidates = final_candidates(prior)
    assert list(candidates) == ["round40", "round60"]
    with pytest.raises(ValueError, match="selection report"):
        final_candidates({**prior, "mode": "final"})


def test_partial_journal_requires_matching_model_hashes_and_contiguous_seeds(
    tmp_path: Path,
) -> None:
    specs = {"round40": {"path": "/m.json", "sha256": "hash-a"}}
    manifest = identity_for(
        mode="final", specs=specs,
        prefix="fresh-unseen-final", n=2, max_decisions=4096,
        emulator_revision="revision-1", reference_label="round40",
        selection_sha256="selection-sha",
    )
    path = tmp_path / "evaluation.partial.jsonl"
    row0 = {"index": 0, "seed": "fresh-unseen-final-0",
            "models": {"round40": result("fresh-unseen-final-0", True)}}
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"schema": JOURNAL_SCHEMA, "identity": manifest}) + "\n")
        file.write(json.dumps(row0) + "\n")
    assert load_journal(path, manifest) == [row0]
    with pytest.raises(ValueError, match="identity mismatch"):
        load_journal(path, {**manifest, "model_specs": {
            "round40": {"path": "/m.json", "sha256": "changed"},
        }})
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({**row0, "index": 2, "seed": "fresh-unseen-final-2"}) + "\n")
    with pytest.raises(ValueError, match="missing/duplicated"):
        load_journal(path, manifest)
