"""Pure checks for the paired H16 fresh-seed diagnostic."""
from __future__ import annotations

import json

import pytest

from tools.diagnose_h16_paired_checkpoints import (
    SCHEMA,
    build_report,
    make_jobs,
    paired_comparison,
    read_journal,
    summarize_runs,
)


def sample(seed: str, *, floor: int, won7: bool, outcome: str = "defeat",
           death_floor: int = 7, damage: float = 0.7) -> dict:
    return {
        "seed": seed, "outcome": outcome, "max_floor_reached": floor,
        "death_floor": death_floor, "floor7_reached": floor >= 7,
        "floor7_won": won7,
        "post_floor7_hp": 0.2 if won7 else None,
        "lost_fights": (
            [] if outcome == "act_clear"
            else [{"floor": death_floor, "damage": damage}]
        ),
        "searches": 2, "search_transitions": 20, "search_overrides": 1,
        "decisions": 10,
    }


def test_seed_jobs_are_paired_and_distinct():
    jobs = make_jobs([39, 107], "new-seed", 2)
    assert jobs == [
        (39, "policy", "new-seed-0000"),
        (39, "policy", "new-seed-0001"),
        (39, "puct", "new-seed-0000"),
        (39, "puct", "new-seed-0001"),
        (107, "policy", "new-seed-0000"),
        (107, "policy", "new-seed-0001"),
        (107, "puct", "new-seed-0000"),
        (107, "puct", "new-seed-0001"),
    ]


def test_per_floor_defeat_damage_and_survival():
    rows = [sample("a", floor=7, won7=False, damage=0.9),
            sample("b", floor=10, won7=True, death_floor=10, damage=0.4),
            sample("c", floor=16, won7=True, outcome="act_clear")]
    result = summarize_runs(rows)
    assert result["floor7_won"] == 2
    assert result["floor7_win_rate"] == pytest.approx(2 / 3)
    assert result["post_floor7_hp_mean"] == pytest.approx(0.2)
    assert result["defeat_damage_by_floor"] == {
        "7": {"defeats": 1, "mean": 0.9},
        "10": {"defeats": 1, "mean": 0.4},
    }
    assert result["act_clears"] == 1


def test_paired_comparison_uses_matching_seeds_even_when_reordered():
    old = [sample("a", floor=7, won7=False),
           sample("b", floor=10, won7=True)]
    new = [sample("b", floor=7, won7=False),
           sample("a", floor=16, won7=True, outcome="act_clear")]
    result = paired_comparison(old, new)
    assert result["floor7_only_right"] == 1
    assert result["floor7_only_left"] == 1
    assert result["clears_only_right"] == 1
    assert result["mean_floor_delta_right_minus_left"] == 3.0
    with pytest.raises(ValueError, match="same unique seeds"):
        paired_comparison(old, [new[0], new[0]])


def test_full_report_has_puct_and_checkpoint_pairs():
    rows = []
    for number in (39, 107):
        for mode in ("policy", "puct"):
            for seed in ("a", "b"):
                rows.append({
                    "round": number, "mode": mode, "seed": seed,
                    "result": sample(
                        seed, floor=7 if mode == "policy" else 10,
                        won7=(mode == "puct"),
                    ),
                })
    report = build_report(rows, [39, 107])
    assert report["arms"]["r107-puct"]["floor7_win_rate"] == 1.0
    assert report["paired"]["puct_vs_policy"]["107"]["floor7_only_right"] == 2
    assert report["paired"]["checkpoint_progress"]["policy"]["r39_to_r107"][
        "floor7_only_right"] == 0


def test_resume_requires_exact_identity_and_order(tmp_path):
    jobs = make_jobs([39, 107], "fresh", 1)
    path = tmp_path / "partial.jsonl"
    identity = {"fingerprint": "a"}
    first = {"round": 39, "mode": "policy", "seed": "fresh-0000",
             "result": sample("fresh-0000", floor=7, won7=False)}
    path.write_text(
        json.dumps({"schema": SCHEMA, "identity": identity}) + "\n"
        + json.dumps(first) + "\n", encoding="utf-8",
    )
    assert read_journal(path, identity, jobs) == [first]
    with pytest.raises(ValueError, match="identity differs"):
        read_journal(path, {"fingerprint": "changed"}, jobs)
    bad_jobs = make_jobs([107, 39], "fresh", 1)
    with pytest.raises(ValueError, match="reordered"):
        read_journal(path, identity, bad_jobs)
