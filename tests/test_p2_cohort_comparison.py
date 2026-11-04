"""Offline P2 cohort report invariants: no live emulator needed."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tools.report_p2_cohort_comparison import MATCH_FIELDS, compare_cohorts


def fixture(workers: int, wall: float = 10.0):
    fields = {
        "emulator_sha": "e", "checkpoint_sha256": "ckpt", "macro_sha256": None,
        "macro_prior_weight": 0.0, "horizon_lambda": 0.25,
        "ascension": 0, "map_profile": "generated", "simulations": 16,
        "no_search": False, "depth": 8, "max_simulation_steps": 512,
        "max_searches": 2, "max_decisions": 80,
        "rollout_transport": "fused", "native_profile": False,
        "paired": True, "profile_overhead": False,
        "actor_inference": "shared", "actor_device": "cpu",
        "actor_max_batch": 16, "actor_max_delay_ms": 2.0,
        "actor_profile": True, "actor_batch_parity_probe": False,
        "actor_fixed_shape_batch_size": 0,
        "seeds": ["x", "y"],
    }
    assert set(fields) == set(MATCH_FIELDS)
    runs = [
        {
            "seed": seed, "trace_sha256": f"trace-{seed}",
            "public_teacher_parity": True,
            "runs": [{
                "seed": seed, "arm": arm, "outcome": "censored",
                "complete": False, "decisions": 8, "max_floor": 3,
                "search_roots": 2, "search_simulations": 32,
                "search_transitions": 100, "real_transitions": 8,
                "episode_wall_seconds": 1.0,
            } for arm in ("profile_b1", "profile_b2")],
        }
        for seed in ("x", "y")
    ]
    return {
        "schema": "sts2-p2-parallel-cohort-profile-v1",
        "identity": {
            **fields, "worker_count": workers, "ai_sha": "same",
            "emulator_cli": {"sha256": "dll", "path": "/fixed/dll"},
        },
        "seed_reports": runs, "cohort_wall_seconds": wall,
        "completed_episodes": 0, "censored_episodes": 4,
        "real_transitions": 32, "search_transitions": 400,
        "shared_actor_metrics": {"batches": 10},
    }


def test_valid_paired_cohorts_are_compared_by_wall_time():
    report = compare_cohorts(fixture(1, 10), fixture(15, 5))
    assert report["parity"]["passed"]
    assert report["wall_speedup_15_vs_1"] == 2.0
    assert report["parallel_efficiency_vs_15"] == pytest.approx(2 / 15)
    assert report["maximum_concurrently_active_seed_tasks"] == 2
    assert report["fifteen_workers"]["completed_episodes_per_second"] == 0.0
    assert report["one_worker"]["search_transitions_per_second"] == 40.0
    assert any("Zero completed" in w for w in report["warnings"])
    assert any("Fewer than 15" in w for w in report["warnings"])


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda x: x["identity"].update({"checkpoint_sha256": "other"}), "checkpoint"),
    (lambda x: x["identity"].update({"depth": 0}), "depth"),
    (lambda x: x["identity"].update({"emulator_cli": {"sha256": "other"}}), "binary"),
    (lambda x: x["seed_reports"][0].update({"trace_sha256": "changed"}), "replay drift"),
    (lambda x: x["seed_reports"][0]["runs"][0].update({"decisions": 9}), "Gameplay"),
    (lambda x: x["seed_reports"][0]["runs"][0].update({"arm": "control_a"}), "arms"),
    (lambda x: x.update({"search_transitions": 401}), "count drift"),
    (lambda x: x["seed_reports"][0].update({"public_teacher_parity": None}), "verification"),
    (lambda x: x.update({"cohort_wall_seconds": 0}), "cohort_wall_seconds"),
])
def test_mismatched_cohort_fails_closed(mutate, message):
    a, b = fixture(1), fixture(15)
    mutate(b)
    with pytest.raises(ValueError, match=message):
        compare_cohorts(a, b)


def test_duplicate_and_reordered_seeds_fail_closed():
    a, b = fixture(1), fixture(15)
    b["seed_reports"].reverse()
    with pytest.raises(ValueError, match="seed order"):
        compare_cohorts(a, b)
    b = deepcopy(a)
    b["identity"]["worker_count"] = 15
    b["seed_reports"][1]["seed"] = "x"
    with pytest.raises(ValueError, match="duplicate seed"):
        compare_cohorts(a, b)


def test_actor_numerical_probe_is_not_valid_throughput_evidence():
    a, b = fixture(1), fixture(15)
    a["identity"]["actor_batch_parity_probe"] = True
    b["identity"]["actor_batch_parity_probe"] = True
    with pytest.raises(ValueError, match="extra forwards"):
        compare_cohorts(a, b)


def test_shape_bucket_mode_must_match_in_comparison():
    a, b = fixture(1), fixture(15)
    b["identity"]["actor_fixed_shape_batch_size"] = 4
    with pytest.raises(ValueError, match="actor_fixed_shape_batch_size"):
        compare_cohorts(a, b)


def test_cohort_warning_tracks_real_completion_count():
    single, parallel = fixture(1), fixture(15)
    for profile in (single, parallel):
        for row in profile["seed_reports"]:
            for run in row["runs"]:
                run["complete"] = True
                run["outcome"] = "act_clear"
        profile["completed_episodes"] = 4
        profile["censored_episodes"] = 0
    report = compare_cohorts(single, parallel)
    assert not any("Zero completed" in w for w in report["warnings"])
    assert report["fifteen_workers"]["completed_episodes"] == 4
