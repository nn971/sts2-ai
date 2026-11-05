"""P2 replay diagnostics must locate and preserve exact differences."""
from __future__ import annotations

from tools.p2_replay_diagnostics import (
    classify_trace_difference,
    first_trace_difference,
)


def _trace():
    return {
        "actions": [{
            "observation_hash": "abc",
            "legal_action_ids": ["a", "b"],
            "action_id": "a",
            "teacher_policy": [0.25, 0.75],
        }],
        "final_public_frame": {"floor": 1, "combat": None},
        "outcome": "censored",
    }


def test_equal_public_teacher_trace():
    assert first_trace_difference(_trace(), _trace()) is None


def test_floating_teacher_drift_is_reported_not_ignored():
    before, after = _trace(), _trace()
    after["actions"][0]["teacher_policy"][1] += 1e-7
    result = first_trace_difference(before, after)
    assert result is not None
    assert result["path"] == "$.actions[0].teacher_policy[1]"
    assert result["kind"] == "numeric_difference"
    assert result["absolute_difference"] > 0


def test_changed_player_action_is_first_mismatch():
    before, after = _trace(), _trace()
    after["actions"][0]["action_id"] = "b"
    after["final_public_frame"]["floor"] = 2
    result = first_trace_difference(before, after)
    assert result is not None
    assert result["path"] == "$.actions[0].action_id"


def test_different_legal_menu_and_trace_length():
    before, after = _trace(), _trace()
    after["actions"][0]["legal_action_ids"].append("c")
    assert first_trace_difference(before, after)["kind"] == "different_list_length"
    after = _trace()
    after["actions"].append(after["actions"][0])
    assert first_trace_difference(before, after)["path"] == "$.actions"


def test_different_public_keys():
    before, after = _trace(), _trace()
    after["final_public_frame"]["hidden"] = "extra"
    diff = first_trace_difference(before, after)
    assert diff is not None
    assert diff["path"] == "$.final_public_frame"
    assert diff["kind"] == "different_keys"
    assert diff["actual_only"] == ["hidden"]


def test_search_transition_only_drift_preserves_semantic_parity():
    before = {**_trace(), "searches": 2, "transitions": 4692}
    after = {**_trace(), "searches": 2, "transitions": 4693}
    diagnosis = classify_trace_difference(before, after)
    assert diagnosis["kind"] == "search_work_only"
    assert diagnosis["semantic_parity"] is True
    assert diagnosis["search_work_parity"] is False
    assert diagnosis["semantic_differences"] == {}
    assert diagnosis["first_difference"]["path"] == "$.transitions"
    assert diagnosis["search_work"]["absolute_drift"] == 1
    assert diagnosis["search_work"]["relative_drift"] == 1 / 4693


def test_search_work_difference_does_not_hide_teacher_drift():
    before = {**_trace(), "searches": 2, "transitions": 4692}
    after = {**_trace(), "searches": 2, "transitions": 4693}
    after["actions"][0] = {**after["actions"][0],
                          "teacher_policy": [0.24, 0.76]}
    diagnosis = classify_trace_difference(before, after)
    assert diagnosis["kind"] == "semantic_or_teacher_drift"
    assert diagnosis["semantic_parity"] is False
    assert diagnosis["search_work_parity"] is False
    assert diagnosis["first_difference"]["path"] == "$.actions[0].teacher_policy[0]"
    assert diagnosis["semantic_differences"]["actions"]["path"] == (
        "$.actions[0].teacher_policy[0]"
    )


def test_replay_classification_equal_is_exact():
    trace = {**_trace(), "transitions": 123, "searches": 1}
    assert classify_trace_difference(trace, trace) == {
        "kind": "exact_replay",
        "semantic_parity": True,
        "search_work_parity": True,
        "first_difference": None,
    }
