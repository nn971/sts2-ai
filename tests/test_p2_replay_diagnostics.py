"""P2 replay diagnostics must locate and preserve exact differences."""
from __future__ import annotations

from tools.p2_replay_diagnostics import first_trace_difference


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
