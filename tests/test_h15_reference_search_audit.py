"""Offline H15 evaluation audit: search interventions and paired case identity."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.search.combat_puct import (
    H15_SEARCH_VERSION,
    CombatFrame,
    CombatSearchResult,
)
from sts2_ai.training.h15_search_supervision import target_record
from tools.audit_h15_reference_search import audit_root, compare_saved_evaluations


def test_audit_root_finds_search_override_without_hidden_state():
    state = {
        "act": 1, "floor": 2, "hp": 40, "max_hp": 70,
        "deck": [], "relics": [], "potions": [], "map": [],
        "combat": {
            "hand": [], "discard_pile": [], "exhaust_pile": [],
            "enemies": [{
                "instance_id": 1, "enemy_id": "crawler",
                "move_id": "attack", "hp": 20,
            }],
        },
    }
    actions = (
        LegalAction("a", "end_turn", "{}"),
        LegalAction("b", "end_turn", "{}"),
    )
    root = CombatFrame("public-root", Observation("fair", json.dumps(state), "hash"),
                       actions)
    result = CombatSearchResult(
        H15_SEARCH_VERSION, "public-root", ("a", "b"),
        (0, 16), (None, 0.6), 0.6,
        16, 0, 16, 16, "value", 2,
    )
    record = target_record(
        root, result, terminal_reward=1.5,
        search_model_sha256="a" * 64,
        source_seed="source-1", emulator_revision="emulator-sha",
    )
    audit = audit_root(record, (1.0, 0.0))
    assert audit["greedy_action_id"] == "a"
    assert audit["search_action_id"] == "b"
    assert audit["search_overrode_greedy"] is True
    assert audit["visit_policy_kl_from_prior"] == pytest.approx(
        -__import__("math").log(1 / (1 + __import__("math").exp(1.0))),
    )


def test_paired_evaluation_detects_equal_aggregates_with_different_cases():
    def record(case, hp):
        return {
            "case_id": case, "encounter": "mawler-a10", "deck": "silent-starter-v1",
            "source_nature_seed": 3 if case == "one" else 5,
            "outcome": "victory", "exit_hp": hp,
            "reward": 1 + hp / 70, "decisions": 15,
        }

    original = {"one": record("one", 30), "two": record("two", 40)}
    changed = {"one": record("one", 35), "two": record("two", 35)}
    report = compare_saved_evaluations(original, changed)
    assert report["paired_cases"] == 2
    assert report["different_recorded_outcomes"] == 2
    assert report["mean_paired_exit_hp_delta"] == 0
    with pytest.raises(ValueError, match="different source"):
        compare_saved_evaluations(original, {
            **changed, "one": {**changed["one"], "source_nature_seed": 4},
        })
