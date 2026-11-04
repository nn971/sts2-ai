"""P2.5 offline uninstrumented A/B/B/A actor protocol contracts."""
from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

from tools.report_p2_actor_abba import report_actor_abba
from tools.report_p2_cohort_comparison import MATCH_FIELDS
from tools.run_p2_actor_abba import make_commands


def cohort(
    wall: float, *, batch: int = 1, fixed_shape: int = 0,
    complete: bool = False,
) -> dict:
    fields = {
        "emulator_sha": "emu-sha",
        "checkpoint_sha256": "model-sha",
        "macro_sha256": "macro-sha",
        "macro_prior_weight": 0.5,
        "horizon_lambda": 0.411395,
        "ascension": 0,
        "map_profile": "generated",
        "simulations": 16,
        "no_search": False,
        "depth": 8,
        "max_simulation_steps": 512,
        "max_searches": 2,
        "max_decisions": 80,
        "rollout_transport": "fused",
        "native_profile": False,
        "paired": True,
        "profile_overhead": False,
        "actor_inference": "shared",
        "actor_device": "cuda",
        "actor_max_batch": batch,
        "actor_max_delay_ms": 2.0,
        "actor_profile": False,
        "actor_batch_parity_probe": False,
        "actor_fixed_shape_batch_size": fixed_shape,
        "seeds": ["seed-a", "seed-b"],
    }
    assert set(fields) == set(MATCH_FIELDS)
    rows = []
    for seed in fields["seeds"]:
        done = complete and seed == "seed-a"
        runs = []
        for arm in ("control_a", "control_a2"):
            runs.append({
                "seed": seed, "arm": arm, "outcome": "act_clear" if done else "censored",
                "complete": done, "decisions": 8, "max_floor": 3,
                "search_roots": 2, "search_simulations": 32,
                "search_transitions": 100, "real_transitions": 8,
                "episode_wall_seconds": 1.0, "search_instrumentation": False,
                "wire_instrumentation": False, "native_instrumentation": False,
            })
        rows.append({
            "seed": seed, "runs": runs, "trace_sha256": f"sha-{seed}",
            "public_teacher_parity": True,
        })
    return {
        "schema": "sts2-p2-parallel-cohort-profile-v1",
        "identity": {
            **fields, "episode_profile": False,
            "worker_count": 15, "ai_sha": "ai-sha",
            "emulator_cli": {"path": "/tmp/cli.dll", "sha256": "dll-sha"},
            "torch": "2.9", "platform": "Linux-x", "cpu_count": 24,
        },
        "seed_reports": rows, "cohort_wall_seconds": wall,
        "completed_episodes": 2 if complete else 0,
        "censored_episodes": 2 if complete else 4,
        "search_transitions": 400, "real_transitions": 32,
    }


def inputs() -> list[dict]:
    return [
        cohort(100.0), cohort(70.0, batch=16, fixed_shape=4),
        cohort(60.0, batch=16, fixed_shape=4), cohort(100.0),
    ]


def test_order_balanced_comparison_requires_public_and_teacher_replay():
    data = inputs()
    frozen = deepcopy(data)
    report = report_actor_abba(*data)
    assert data == frozen
    assert report["parity"]["seed_count"] == 2
    assert report["parity"]["passed"] is True
    assert report["ratio_of_total_baseline_to_candidate_wall"] == pytest.approx(
        200 / 130
    )
    assert report["wall_reduction_fraction"] == pytest.approx(0.35)
    assert report["candidate_faster_in_both_pairs"] is True
    assert report["workload"]["search_transitions_per_run"] == 400
    assert report["workload"]["completed_episode_executions_per_run"] == 0
    assert any("Censored" in entry for entry in report["warnings"])


def test_true_completed_episodes_are_not_misreported_as_censored():
    data = [cohort(wall, batch=b, fixed_shape=f, complete=True)
            for wall, b, f in ((100, 1, 0), (70, 16, 4), (60, 16, 4),
                               (100, 1, 0))]
    report = report_actor_abba(*data)
    assert report["workload"]["completed_episode_executions_per_run"] == 2
    assert report["workload"]["censored_episode_executions_per_run"] == 2


@pytest.mark.parametrize(("modify", "reason"), [
    (lambda p: p[1]["seed_reports"][0].update({"trace_sha256": "changed"}),
     "teacher replay"),
    (lambda p: p[2]["identity"].update({"checkpoint_sha256": "changed"}),
     "provenance drift"),
    (lambda p: p[3]["identity"].update({"actor_max_batch": 16}),
     "inconsistent actor mode"),
    (lambda p: p[2]["seed_reports"][0]["runs"][0].update(
        {"search_transitions": 101}), "paired work drift"),
    (lambda p: p[1]["seed_reports"][0]["runs"][0].update(
        {"max_floor": 7}), "paired work drift"),
    (lambda p: p[3]["seed_reports"][0]["runs"][0].update(
        {"wire_instrumentation": True}), "instrumented"),
    (lambda p: p[1]["identity"].update({"actor_profile": True}),
     "measurement instrumentation"),
    (lambda p: p[1]["identity"].update({"episode_profile": True}),
     "measurement instrumentation"),
    (lambda p: p[1]["identity"].update({"worker_count": 1}),
     "provenance drift"),
    (lambda p: p[1].update({"censored_episodes": 0}),
     "inconsistent reported censored_episodes"),
    (lambda p: p[2]["identity"].update({"seeds": ["seed-b", "seed-a"]}),
     "seed-list/order mismatch"),
])
def test_report_fails_closed_on_invalid_evidence(modify, reason):
    data = inputs()
    modify(data)
    with pytest.raises(ValueError, match=reason):
        report_actor_abba(*data)


def test_closer_order_pair_result_is_preserved_instead_of_declaring_success():
    data = inputs()
    data[2]["cohort_wall_seconds"] = 110.0
    report = report_actor_abba(*data)
    assert report["candidate_faster_in_both_pairs"] is False
    assert report["order_pair_speedup_A2_over_B2"] == pytest.approx(100 / 110)


def test_orchestrator_stages_exact_15_worker_unprofiled_order(tmp_path):
    args = Namespace(
        seeds=None, seed_prefix="p2-independent", seed_count=32, workers=15,
        max_decisions=80, max_searches=2, simulations=16, depth=8,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0.5, macro_model=Path("macro.json"),
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-sha", checkpoint=Path("model.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    commands, files = make_commands(args)
    assert len(commands) == 5
    assert [path.parent.name for path in files[:4]] == ["A1", "B1", "B2", "A2"]
    for command in commands[:4]:
        assert "--no-episode-profile" in command
        assert "--paired" in command
        assert "--actor-profile" not in command
        assert command[command.index("--workers") + 1] == "15"
        assert command[command.index("--seeds") + 1].count(",") == 31
        assert "p2-independent-0" in command
    assert [int(c[c.index("--actor-max-batch") + 1]) for c in commands[:4]] == [
        1, 16, 16, 1,
    ]
    assert [
        int(c[c.index("--actor-fixed-shape-batch-size") + 1])
        for c in commands[:4]
    ] == [0, 4, 4, 0]
    assert "--a1" in commands[-1] and "--a2" in commands[-1]
