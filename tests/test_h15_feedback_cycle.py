"""H15 feedback-cycle integration checks, with no emulator or CUDA dependency."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

PATH = Path(__file__).resolve().parents[1] / "tools/run_h15_h8_feedback_cycle.py"
SPEC = importlib.util.spec_from_file_location("h15_feedback_cycle", PATH)
assert SPEC is not None and SPEC.loader is not None
cycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cycle)


def test_first_root_policy_comparison_requires_same_public_decision():
    state = {
        "public_state": {"combat": {"enemies": [{"hp": 9}]}},
        "legal_actions": [
            {"action_id": "card:1", "kind": "play_card", "payload_json": "{}"},
            {"action_id": "end", "kind": "end_turn", "payload_json": "{}"},
        ],
        "chance_prior_id": cycle.H8_DECLARED_LAW_MODEL,
        "source_seed": "fixture:reference:113",
        "search_simulations": 16,
        "visits": [4, 12],
    }
    changed = {**state, "visits": [12, 4]}
    report = cycle.compare_first_root(state, changed)
    assert report["visit_policy_l1_change"] == pytest.approx(1.0)
    assert report["root_action_ids"] == ["card:1", "end"]
    with pytest.raises(ValueError, match="different initial root"):
        cycle.compare_first_root(state, {**changed, "source_seed": "other"})


def test_feedback_requires_real_parameter_update(tmp_path):
    import torch

    initial = tmp_path / "before.pt"
    updated = tmp_path / "after.pt"
    torch.save({"state_dict": {"weight": torch.tensor([1.0, 2.0])}}, initial)
    torch.save({"state_dict": {"weight": torch.tensor([1.0, 2.0])}}, updated)
    with pytest.raises(RuntimeError, match="no finite model-parameter update"):
        cycle.verify_parameter_update(initial, updated)
    torch.save({"state_dict": {"weight": torch.tensor([1.0, 2.5])}}, updated)
    report = cycle.verify_parameter_update(initial, updated)
    assert report["changed_tensors"] == 1
    assert report["model_delta_l2"] == pytest.approx(0.5)
