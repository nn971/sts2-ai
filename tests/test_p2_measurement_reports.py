"""Offline P2.0 acceptance-report contracts; never depend on CUDA or emulator."""
from __future__ import annotations

from copy import deepcopy

import pytest

from tools.report_p2_actor_ab import compare_actor_candidate
from tools.report_p2_cohort_comparison import MATCH_FIELDS
from tools.report_p2_profile_overhead import report_profile_overhead


def cohort(*, workers: int, actor_max_batch: int = 1,
           fixed_shape: int = 0, wall: float = 20.0,
           overhead: bool = False) -> dict:
    fields = {
        "emulator_sha": "emu-v0", "checkpoint_sha256": "h16",
        "macro_sha256": "macro", "macro_prior_weight": 0.5,
        "horizon_lambda": 0.4, "ascension": 0, "map_profile": "generated",
        "simulations": 16, "no_search": False, "depth": 8,
        "max_simulation_steps": 512, "max_searches": 2,
        "max_decisions": 80, "rollout_transport": "fused",
        "native_profile": False, "paired": True,
        "profile_overhead": overhead, "actor_inference": "shared",
        "actor_device": "cuda", "actor_max_batch": actor_max_batch,
        "actor_max_delay_ms": 2.0, "actor_profile": True,
        "actor_batch_parity_probe": False,
        "actor_fixed_shape_batch_size": fixed_shape, "seeds": ["x", "y"],
    }
    assert set(fields) == set(MATCH_FIELDS)
    arms = ("control_a", "profile_b1", "profile_b2", "control_a2") if (
        overhead
    ) else ("profile_b1", "profile_b2")
    rows = []
    for seed in fields["seeds"]:
        runs = []
        for arm in arms:
            measured = arm.startswith("profile_")
            runs.append({
                "seed": seed, "arm": arm, "outcome": "censored",
                "complete": False, "decisions": 80, "max_floor": 5,
                "search_roots": 2, "search_simulations": 32,
                "search_transitions": 256, "real_transitions": 80,
                "episode_wall_seconds": 11.0 if measured else 10.0,
                "search_instrumentation": measured,
                "wire_instrumentation": measured,
                "native_instrumentation": False,
            })
        rows.append({
            "seed": seed, "runs": runs, "trace_sha256": f"trace-{seed}",
            "public_teacher_parity": True,
        })
    runs = [r for row in rows for r in row["runs"]]
    return {
        "schema": "sts2-p2-parallel-cohort-profile-v1",
        "identity": {
            **fields, "worker_count": workers, "ai_sha": "ai-sha",
            "emulator_cli": {"path": "/same/emulator.dll", "sha256": "dll-sha"},
        },
        "seed_reports": rows,
        "cohort_wall_seconds": wall,
        "completed_episodes": 0,
        "censored_episodes": len(runs),
        "search_transitions": 256 * len(runs),
        "real_transitions": 80 * len(runs),
        "shared_actor_metrics": {"requests": 100, "batches": 25},
    }


def test_profile_overhead_requires_exact_abba_and_reports_paired_totals():
    data = cohort(workers=1, overhead=True)
    result = report_profile_overhead(data)
    assert result["parity"] == {"passed": True, "seed_count": 2}
    assert result["ratio_of_paired_totals"] == pytest.approx(1.1)
    assert result["paired_mean_overhead_fraction"] == pytest.approx(0.1)
    assert result["per_seed_ratio_median"] == pytest.approx(1.1)
    assert result["source"]["actor_profile"] is True


@pytest.mark.parametrize(("change", "message"), [
    (lambda data: data["identity"].update({"profile_overhead": False}),
     "profile-overhead"),
    (lambda data: data["identity"].update({"paired": False}), "--paired"),
    (lambda data: data["seed_reports"][0].update({"public_teacher_parity": False}),
     "exact replay"),
    (lambda data: data["seed_reports"][0]["runs"][0].update(
        {"search_instrumentation": True}), "instrumentation"),
    (lambda data: data["seed_reports"][0]["runs"][0].update(
        {"episode_wall_seconds": 0}), "episode_wall_seconds"),
    (lambda data: data["seed_reports"][1]["runs"][1].update(
        {"search_transitions": 255}), "Workload mismatch"),
])
def test_bad_profile_overhead_report_fails_closed(change, message):
    data = cohort(workers=1, overhead=True)
    change(data)
    with pytest.raises(ValueError, match=message):
        report_profile_overhead(data)


def test_actor_ab_requires_exact_cross_config_replay():
    a1 = cohort(workers=1)
    a15 = cohort(workers=15, wall=14.0)
    b1 = cohort(workers=1, actor_max_batch=16, fixed_shape=4, wall=23.0)
    b15 = cohort(workers=15, actor_max_batch=16, fixed_shape=4, wall=12.0)
    result = compare_actor_candidate(a1, a15, b1, b15)
    assert result["parity"]["cross_configuration"] is True
    assert result["wall_speedup_baseline_over_candidate_15_workers"] == pytest.approx(14 / 12)
    assert result[
        "search_throughput_ratio_candidate_over_baseline_15_workers"
    ] == pytest.approx(14 / 12)
    assert result["candidate"]["fixed_shape_batch_size"] == 4


def test_actor_ab_rejects_cross_mode_teacher_drift():
    profiles = [cohort(workers=w, actor_max_batch=b, fixed_shape=f) for w, b, f
                in ((1, 1, 0), (15, 1, 0), (1, 16, 4), (15, 16, 4))]
    profiles[3]["seed_reports"][1]["trace_sha256"] = "different-teacher"
    with pytest.raises(ValueError, match="exact teacher/public replay drift"):
        compare_actor_candidate(*profiles)


@pytest.mark.parametrize(("field", "value"), [
    ("checkpoint_sha256", "changed-model"),
    ("simulations", 64),
    ("actor_profile", False),
    ("emulator_cli", {"sha256": "another-dll"}),
    ("ai_sha", "another-revision"),
])
def test_actor_ab_rejects_non_actor_identity_differences(field, value):
    profiles = [cohort(workers=w, actor_max_batch=b, fixed_shape=f) for w, b, f
                in ((1, 1, 0), (15, 1, 0), (1, 16, 4), (15, 16, 4))]
    profiles[2]["identity"][field] = value
    with pytest.raises(ValueError, match="(identity mismatch|cohort identity mismatch)"):
        compare_actor_candidate(*profiles)


def test_actor_ab_rejects_completed_outcome_drift_even_with_same_trace_digest():
    profiles = [cohort(workers=w, actor_max_batch=b, fixed_shape=f) for w, b, f
                in ((1, 1, 0), (15, 1, 0), (1, 16, 4), (15, 16, 4))]
    for p in profiles[2:]:
        p["seed_reports"][0]["runs"][0]["max_floor"] = 6
    with pytest.raises(ValueError, match="workload drift"):
        compare_actor_candidate(*profiles)


def test_actor_ab_does_not_accept_overhead_cohort():
    profiles = [cohort(workers=w, actor_max_batch=b, fixed_shape=f, overhead=True)
                for w, b, f in ((1, 1, 0), (15, 1, 0), (1, 16, 4), (15, 16, 4))]
    with pytest.raises(ValueError, match="ordinary B/B"):
        compare_actor_candidate(*profiles)


def test_report_inputs_are_not_mutated():
    profiles = [cohort(workers=w, actor_max_batch=b, fixed_shape=f) for w, b, f
                in ((1, 1, 0), (15, 1, 0), (1, 16, 4), (15, 16, 4))]
    original = deepcopy(profiles)
    compare_actor_candidate(*profiles)
    assert profiles == original
