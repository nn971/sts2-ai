"""H22 paired fixed-horizon pre/post training pilot contracts."""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.report_h22_learning_pilot import verify_pilot


def _actor():
    return {
        "device": "cuda",
        "p2_deterministic_cuda": True,
        "p2_cuda_cublas_workspace_config": ":4096:8",
        "p2_h21_collator": "packed",
        "p2_fixed_shape_batch_size": 4,
        "p2_fixed_shape_bucket_forwards": 3,
        "p2_fixed_shape_physical_rows": 12,
        "requests": 8,
    }


def _stage(floors, *, roots=0):
    records = [
        {
            "seed": seed, "max_floor_reached": floor,
            "outcome": "defeat", "death_floor": floor, "decisions": 20,
        }
        for seed, floor in floors.items()
    ]
    return {
        "played": len(floors), "completed": len(floors), "censored": 0,
        "searches": roots, "search_transitions": 16 * roots,
        "decisions": 20 * len(floors),
        "rollout_profile": {"actor_service": _actor()},
        "combat_progress": {"won_fights": sum(f > 1 for f in floors.values())},
        "floor_progress": {
            "mean_max_floor_reached": sum(floors.values()) / len(floors),
            "runs": records,
        },
    }


def _write_pilot(root: Path):
    opts = {
        "representation": "h21", "actor_inference": "shared",
        "actor_device": "cuda", "actor_deterministic_cuda": True,
        "actor_h21_collator": "packed", "actor_fixed_shape_batch_size": 4,
        "actor_max_batch": 4, "workers": 4, "runs_per_round": 4,
        "heldout_runs": 3, "rounds": 2,
        "map_profile": "generated", "ascension": 0,
        "rollout_transport": "fused",
        "evaluation_horizon_lambda": 0.4113950303216448,
    }
    baseline_floors = {"hold-0": 1, "hold-1": 4, "hold-2": 3}
    next_floors = (
        {"hold-0": 3, "hold-1": 2, "hold-2": 3},
        {"hold-0": 4, "hold-1": 5, "hold-2": 3},
    )
    rounds = []
    for index, heldout_floors in enumerate(next_floors, 1):
        saved = root / "models" / f"joint-r{index:04d}.pt"
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(f"checkpoint-{index}".encode())
        rounds.append({
            "round": index, "horizon_lambda": 0.411 + index / 1000,
            "checkpoint_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
            "train": _stage({f"train-{i}": 1 for i in range(4)}, roots=1),
            "heldout": _stage(heldout_floors),
            "update": {"optimizer_steps": 1},
            "timing": {"approx_emulator_transitions_per_second": 100.0},
        })
    progress = {
        "fingerprint": "synthetic-fixed-seeds",
        "experiment": {
            "schema": "sts2-h22-h21-grounded-joint-act-v1",
            "emulator_revision": "pinned-v1", "options": opts,
        },
        "rounds": rounds,
    }
    (root / "optimizer-state.pt").write_bytes(b"optimizer")
    (root / "baseline.json").write_text(json.dumps(_stage(baseline_floors)))
    (root / "progress.json").write_text(json.dumps(progress))
    return progress


def _check(root):
    return verify_pilot(
        root, expected_rounds=2, expected_train=4,
        expected_heldout=3, expected_workers=4,
    )


def test_h22_paired_learning_pilot_uses_identical_seed_outcomes(tmp_path):
    _write_pilot(tmp_path)
    result = _check(tmp_path)
    assert result["fixed_heldout_seed_count"] == 3
    assert [x["mean_max_floor"] for x in result["stages"]] == [
        8 / 3, 8 / 3, 4.0,
    ]
    assert result["paired_rounds"][0] == {
        "round": 1,
        "mean_delta_from_baseline": 0,
        "mean_delta_from_previous": 0,
        "improved_seeds": 1,
        "unchanged_seeds": 1,
        "regressed_seeds": 1,
    }
    assert result["paired_rounds"][1]["mean_delta_from_baseline"] == pytest.approx(
        4 / 3,
    )
    assert result["paired_rounds"][1]["improved_seeds"] == 2
    assert result["train_stages"][0]["train_floor_one_deaths"] == 4
    assert [row["seed"] for row in result["per_seed"]] == [
        "hold-0", "hold-1", "hold-2",
    ]
    assert (tmp_path / "learning-pilot-report.json").is_file()


@pytest.mark.parametrize("change,match", [
    ("horizon", "Fixed evaluation horizon"),
    ("seed_drift", "heldout seed drift"),
    ("censor", "completed, uncensored"),
    ("missing_checkpoint", "checkpoint missing"),
    ("cuda_receipt", "deterministic CUDA receipts"),
    ("training_censor", "incomplete search-assisted training cohort"),
    ("missing_search", "incomplete search-assisted training cohort"),
])
def test_h22_paired_pilot_rejects_unmatched_evidence(tmp_path, change, match):
    progress = _write_pilot(tmp_path)
    if change == "horizon":
        progress["experiment"]["options"]["evaluation_horizon_lambda"] = None
    elif change == "seed_drift":
        progress["rounds"][0]["heldout"]["floor_progress"]["runs"][0]["seed"] = "other"
    elif change == "censor":
        progress["rounds"][0]["heldout"]["censored"] = 1
    elif change == "missing_checkpoint":
        (tmp_path / "models" / "joint-r0001.pt").unlink()
    elif change == "cuda_receipt":
        progress["rounds"][0]["heldout"]["rollout_profile"][
            "actor_service"]["p2_deterministic_cuda"] = False
    elif change == "training_censor":
        progress["rounds"][0]["train"]["censored"] = 1
    elif change == "missing_search":
        progress["rounds"][0]["train"]["searches"] = 0
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match=match):
        _check(tmp_path)


def test_h22_paired_pilot_rejects_invalid_expected_settings(tmp_path):
    _write_pilot(tmp_path)
    with pytest.raises(ValueError, match="Invalid expected heldout"):
        verify_pilot(tmp_path, expected_heldout=0)
    with pytest.raises(ValueError, match="Pilot has incomplete training rounds"):
        verify_pilot(
            tmp_path, expected_rounds=3, expected_workers=4,
            expected_train=4, expected_heldout=3,
        )
