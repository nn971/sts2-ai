"""Focused tests for the read-only H16 16-vs-64 search budget ablation."""
from __future__ import annotations

import json

import pytest

from tools.diagnose_h16_puct_budget import (
    SCHEMA,
    analyze,
    exact_paired_binary_p,
    floor_delta_interval,
    load_journal,
    reference_arms,
)
from tools.diagnose_h16_paired_checkpoints import SCHEMA as BASE_SCHEMA


def outcome(seed: str, floor: int, win7: bool, *, searches: int = 2) -> dict:
    return {
        "seed": seed, "outcome": "defeat", "max_floor_reached": floor,
        "death_floor": floor, "floor7_reached": floor >= 7,
        "floor7_won": win7, "post_floor7_hp": 0.4 if win7 else None,
        "lost_fights": [{"floor": floor, "damage": 0.7}],
        "searches": searches, "search_transitions": 40,
        "search_overrides": 1, "decisions": 10,
    }


def test_selects_exact_r79_reference_seeds():
    rows = [
        {"round": 79, "mode": mode, "seed": f"fresh-{i:04d}",
         "result": outcome(f"fresh-{i:04d}", 7 + i, bool(i))}
        for mode in ("policy", "puct") for i in range(2)
    ]
    reference = {
        "schema": BASE_SCHEMA,
        "identity": {"seed_prefix": "fresh", "seeds": 2},
        "runs": rows,
    }
    selected = reference_arms(reference, 79)
    assert [r["seed"] for r in selected["puct"]] == [
        "fresh-0000", "fresh-0001",
    ]
    reference["runs"][2]["seed"] = "wrong"
    with pytest.raises(ValueError, match="incomplete or reordered"):
        reference_arms(reference, 79)


def test_budget_pairing_and_exact_sign_test():
    reference = {
        "policy": [outcome("a", 7, False), outcome("b", 7, False)],
        "puct": [outcome("a", 7, False), outcome("b", 10, True)],
    }
    budget64 = [outcome("a", 10, True), outcome("b", 7, False)]
    report = analyze(reference, budget64)
    pair = report["paired"]["puct64_vs_puct16"]
    assert pair["floor7_only_right"] == 1
    assert pair["floor7_only_left"] == 1
    assert pair["mean_floor_delta_right_minus_left"] == 0
    assert pair["floor7_exact_two_sided_p"] == 1.0
    assert report["summary"]["puct64"]["floor7_won"] == 1
    assert exact_paired_binary_p(0, 0) == 1.0
    assert exact_paired_binary_p(0, 5) == pytest.approx(0.0625)
    assert floor_delta_interval(reference["puct"], budget64) == (-3.0, 3.0)
    with pytest.raises(ValueError, match="matching paired seeds"):
        floor_delta_interval(reference["puct"], list(reversed(budget64)))


def test_journal_resume_validates_identity_and_order(tmp_path):
    path = tmp_path / "budget.partial.jsonl"
    identity = {"refsha": "x", "simulations": 64}
    record = {"seed": "fresh-0000", "result": outcome("fresh-0000", 7, False)}
    path.write_text(
        json.dumps({"schema": SCHEMA, "identity": identity}) + "\n"
        + json.dumps(record) + "\n", encoding="utf-8",
    )
    assert load_journal(path, identity, ["fresh-0000", "fresh-0001"]) == [record]
    with pytest.raises(ValueError, match="identity mismatch"):
        load_journal(path, {**identity, "simulations": 32}, ["fresh-0000"])
    with pytest.raises(ValueError, match="missing, duplicate or reordered"):
        load_journal(path, identity, ["other-0000", "fresh-0001"])
