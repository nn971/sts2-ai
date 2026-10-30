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


def test_student_receipt_is_bound_to_the_real_teacher_and_targets(tmp_path):
    import torch

    student = tmp_path / "student.pt"
    record = {
        "base_checkpoint_sha256": "a" * 64,
        "search_data_sha256": "b" * 64,
        "chance_prior": cycle.H8_DECLARED_LAW_MODEL,
        "optimizer_steps": 2,
        "in_sample_fit_before": {"policy_loss": 1.0, "value_loss": 0.2},
        "in_sample_fit_after": {"policy_loss": 0.8, "value_loss": 0.1},
    }
    torch.save({"provenance": record}, student)
    receipt = cycle.verify_distillation_receipt(
        student, initial_sha="a" * 64, source_sha="b" * 64,
        expected_steps=2,
    )
    assert receipt["optimizer_steps"] == 2
    assert receipt["in_sample_fit_after"]["policy_loss"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="inconsistent optimizer provenance"):
        cycle.verify_distillation_receipt(
            student, initial_sha="a" * 64, source_sha="c" * 64,
            expected_steps=2,
        )


def test_h9_codec_promotion_keeps_weights_and_is_idempotent(tmp_path):
    import torch
    from dataclasses import asdict

    from sts2_ai.models.h15_combat_codec import H15_COMBAT_SCHEMA
    from sts2_ai.models.h15_policy_value import (
        H15_FORMAT, H7_FORMAT, H15CombatEvaluator,
    )
    from sts2_ai.models.shared_game_objects import SHARED_OBJECT_SCHEMA
    from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer

    config = H7Config(width=32, layers=1, heads=4, feedforward=64)
    model = SharedObjectTransformer(config)
    old = tmp_path / "h9.pt"
    promoted = tmp_path / "h15-bootstrap.pt"
    promoted_twice = tmp_path / "h15-bootstrap-copy.pt"
    torch.save({
        "format": H7_FORMAT,
        "object_schema": SHARED_OBJECT_SCHEMA,
        "config": asdict(config),
        "state_dict": model.state_dict(),
    }, old)
    receipt = cycle.prepare_consistent_h15_teacher(old, promoted)
    assert receipt["source_codec"] == SHARED_OBJECT_SCHEMA
    assert receipt["bootstrap_codec"] == H15_COMBAT_SCHEMA
    h15 = torch.load(promoted, map_location="cpu", weights_only=True)
    assert h15["format"] == H15_FORMAT
    assert all(torch.equal(value, h15["state_dict"][name])
               for name, value in model.state_dict().items())
    # The actual search-teacher loader must use the move-aware H15 codec.
    evaluator = H15CombatEvaluator.load(promoted, device="cpu")
    assert evaluator.codec_schema == H15_COMBAT_SCHEMA
    followup = cycle.prepare_consistent_h15_teacher(promoted, promoted_twice)
    assert followup["source_codec"] == H15_COMBAT_SCHEMA
