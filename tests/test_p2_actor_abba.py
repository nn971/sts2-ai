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
        assert command[command.index("--seeds") + 1].startswith("p2-independent-0,")
    assert [int(c[c.index("--actor-max-batch") + 1]) for c in commands[:4]] == [
        1, 16, 16, 1,
    ]
    assert [
        int(c[c.index("--actor-fixed-shape-batch-size") + 1])
        for c in commands[:4]
    ] == [0, 4, 4, 0]
    assert "--a1" in commands[-1] and "--a2" in commands[-1]


def test_actor_profile_abba_reports_overhead_with_identical_batching():
    data = [cohort(wall, batch=16, fixed_shape=4)
            for wall in (40.0, 44.0, 45.0, 41.0)]
    for profile in data[1:3]:
        profile["identity"]["actor_profile"] = True
    report = report_actor_abba(*data, experiment="actor-profile")
    assert report["parity"]["passed"]
    assert report["experiment"] == "actor-profile"
    assert report["actor_profiled_to_unprofiled_wall_ratio"] == pytest.approx(
        89 / 81
    )
    assert report["actor_profile_overhead_fraction"] == pytest.approx(
        89 / 81 - 1
    )
    assert report["workload"]["censored_episode_executions_per_run"] == 4


def test_actor_profile_abba_detects_wrong_toggle_and_changed_batching():
    data = [cohort(40, batch=16, fixed_shape=4) for _ in range(4)]
    for profile in data[1:3]:
        profile["identity"]["actor_profile"] = True
    report_actor_abba(*data, experiment="actor-profile")
    wrong = deepcopy(data)
    wrong[1]["identity"]["actor_profile"] = False
    with pytest.raises(ValueError, match="measurement instrumentation"):
        report_actor_abba(*wrong, experiment="actor-profile")
    wrong = deepcopy(data)
    wrong[2]["identity"]["actor_max_batch"] = 8
    with pytest.raises(ValueError, match="inconsistent actor mode"):
        report_actor_abba(*wrong, experiment="actor-profile")
    wrong = deepcopy(data)
    for profile in wrong[1:3]:
        profile["identity"]["actor_fixed_shape_batch_size"] = 3
    with pytest.raises(ValueError, match="changed batching"):
        report_actor_abba(*wrong, experiment="actor-profile")
    wrong = deepcopy(data)
    wrong[3]["seed_reports"][1]["trace_sha256"] = "changed"
    with pytest.raises(ValueError, match="teacher replay drift"):
        report_actor_abba(*wrong, experiment="actor-profile")
    with pytest.raises(ValueError, match="Unsupported"):
        report_actor_abba(*data, experiment="unknown")


def test_actor_profile_overhead_runner_preserves_mode_and_replay(tmp_path):
    args = Namespace(
        experiment="actor-profile",
        seeds=None, seed_prefix="p2-overhead", seed_count=32, workers=15,
        max_decisions=80, max_searches=2, simulations=16, depth=8,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0.5, macro_model=Path("macro.json"),
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-sha", checkpoint=Path("model.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    calls, files = make_commands(args)
    assert len(calls) == 5
    assert files[-1].name == "actor-profile-overhead-abba-report.json"
    assert [int(call[call.index("--actor-max-batch") + 1])
            for call in calls[:4]] == [16] * 4
    assert [int(call[call.index("--actor-fixed-shape-batch-size") + 1])
            for call in calls[:4]] == [4] * 4
    assert ["--actor-profile" in call for call in calls[:4]] == [
        False, True, True, False,
    ]
    assert all("--no-episode-profile" in call for call in calls[:4])
    assert all("--paired" in call for call in calls[:4])
    assert calls[-1][calls[-1].index("--experiment") + 1] == "actor-profile"


def test_prefix_cache_abba_requires_fixed4_and_exact_teacher_replay():
    data = [
        cohort(wall, batch=16, fixed_shape=4)
        for wall in (40.0, 34.0, 35.0, 41.0)
    ]
    for row, cap in zip(data, (0, 64, 64, 0), strict=True):
        row["identity"]["p2_prefix_cache_handles"] = cap
    report = report_actor_abba(*data, experiment="prefix-cache")
    assert report["parity"]["passed"]
    assert report["experiment"] == "prefix-cache"
    assert report["configurations"]["A_uncached_fixed4"][
        "p2_prefix_cache_handles"
    ] == 0
    assert report["configurations"]["B_cached_fixed4"][
        "p2_prefix_cache_handles"
    ] == 64
    assert report["ratio_of_total_baseline_to_candidate_wall"] == pytest.approx(
        81 / 69
    )
    wrong = deepcopy(data)
    wrong[2]["identity"]["p2_prefix_cache_handles"] = 32
    with pytest.raises(ValueError, match="Inconsistent P2 prefix cache"):
        report_actor_abba(*wrong, experiment="prefix-cache")
    wrong = deepcopy(data)
    wrong[1]["seed_reports"][0]["trace_sha256"] = "divergent"
    with pytest.raises(ValueError, match="teacher replay"):
        report_actor_abba(*wrong, experiment="prefix-cache")
    wrong = deepcopy(data)
    wrong[1]["identity"]["actor_profile"] = True
    with pytest.raises(ValueError, match="measurement instrumentation"):
        report_actor_abba(*wrong, experiment="prefix-cache")


def test_prefix_cache_abba_runner_only_toggles_retained_handle_cap(tmp_path):
    args = Namespace(
        experiment="prefix-cache",
        seeds=None, seed_prefix="p2-prefix", seed_count=8, workers=15,
        max_decisions=80, max_searches=2, simulations=16, depth=8,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0.5, macro_model=Path("macro.json"),
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-sha", checkpoint=Path("model.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    calls, files = make_commands(args)
    assert files[-1].name == "p2-prefix-cache-abba-report.json"
    assert [int(call[call.index("--p2-prefix-cache-handles") + 1])
            for call in calls[:4]] == [0, 64, 64, 0]
    assert all("--paired" in call for call in calls[:4])
    assert all("--no-episode-profile" in call for call in calls[:4])
    assert all("--actor-profile" not in call for call in calls[:4])
    assert all(int(call[call.index("--actor-max-batch") + 1]) == 16
               for call in calls[:4])
    assert all(int(call[call.index("--actor-fixed-shape-batch-size") + 1]) == 4
               for call in calls[:4])


def test_prefix_cache_report_surfaces_real_saved_steps_and_receipt_cost():
    data = [
        cohort(wall, batch=16, fixed_shape=4)
        for wall in (40.0, 44.0, 45.0, 41.0)
    ]
    for index, profile in enumerate(data):
        cached = index in (1, 2)
        profile["identity"]["p2_prefix_cache_handles"] = 64 if cached else 0
        profile["p2_prefix_cache_diagnostics"] = {
            "roots": 4 if cached else 0,
            "hits": 9 if cached else 0,
            "misses": 25 if cached else 0,
            "emulator_step_calls_avoided": 9 if cached else 0,
            "receipt_calls": 55 if cached else 0,
            "receipt_seconds": 1.25 if cached else 0.0,
            "skipped_capacity": 0, "skipped_provider": 0,
        }
    report = report_actor_abba(*data, experiment="prefix-cache")
    assert [entry["hits"] for entry in report["prefix_cache_diagnostics"]] == [
        0, 9, 9, 0,
    ]
    assert report["prefix_cache_diagnostics"][1]["receipt_calls"] == 55
    assert report["prefix_cache_diagnostics"][2]["receipt_seconds"] == 1.25

    broken = deepcopy(data)
    broken[2]["p2_prefix_cache_diagnostics"]["emulator_step_calls_avoided"] = 8
    with pytest.raises(ValueError, match="invalid cache step/receipt"):
        report_actor_abba(*broken, experiment="prefix-cache")

    broken = deepcopy(data)
    broken[0]["p2_prefix_cache_diagnostics"]["receipt_calls"] = 1
    with pytest.raises(ValueError, match="uncached arm"):
        report_actor_abba(*broken, experiment="prefix-cache")

    broken = deepcopy(data)
    del broken[1]["p2_prefix_cache_diagnostics"]
    with pytest.raises(ValueError, match="Partial P2 cache"):
        report_actor_abba(*broken, experiment="prefix-cache")


def test_p2_transport_abba_keeps_exact_teacher_and_reports_rpc_counts():
    data = [
        cohort(wall, batch=16, fixed_shape=4)
        for wall in (40.0, 27.0, 28.0, 41.0)
    ]
    for item, mode in zip(
        data, ("legacy", "fused", "fused", "legacy"), strict=True,
    ):
        item["identity"]["rollout_transport"] = mode
        for seed in item["seed_reports"]:
            for run in seed["runs"]:
                run["bridge"] = {
                    "batch_rollout_step_frame" if mode == "fused"
                    else "step": {
                        "calls": 5, "total_seconds": 0.75,
                    },
                    **({
                        "observe": {"calls": 5, "total_seconds": 0.33},
                        "legal_actions": {"calls": 5, "total_seconds": 0.22},
                    } if mode == "legacy" else {}),
                }
    report = report_actor_abba(*data, experiment="transport")
    assert report["parity"]["passed"] is True
    assert report["ratio_of_total_baseline_to_candidate_wall"] == pytest.approx(
        81 / 55
    )
    assert report["configurations"]["A_legacy_fixed4"][
        "rollout_transport"
    ] == "legacy"
    assert report["configurations"]["B_fused_fixed4"][
        "rollout_transport"
    ] == "fused"
    stats = report["transport_bridge_operation_totals"]
    assert stats[0]["operations"]["step"]["calls"] == 20
    assert stats[1]["operations"]["batch_rollout_step_frame"]["calls"] == 20
    assert "observe" not in stats[1]["operations"]
    assert stats[0]["worker_summed_nonadditive_bridge_seconds"] > (
        stats[1]["worker_summed_nonadditive_bridge_seconds"]
    )

    wrong = deepcopy(data)
    wrong[2]["identity"]["rollout_transport"] = "legacy"
    with pytest.raises(ValueError, match="legacy/fused/fused/legacy"):
        report_actor_abba(*wrong, experiment="transport")

    wrong = deepcopy(data)
    wrong[1]["seed_reports"][0]["runs"][0]["bridge"]["observe"] = {
        "calls": 1, "total_seconds": -1.0,
    }
    with pytest.raises(ValueError, match="invalid bridge calls/time"):
        report_actor_abba(*wrong, experiment="transport")

    wrong = deepcopy(data)
    wrong[1]["seed_reports"][0]["trace_sha256"] = "different"
    with pytest.raises(ValueError, match="teacher replay drift"):
        report_actor_abba(*wrong, experiment="transport")


def test_p2_transport_runner_stages_split_fused_fused_split(tmp_path):
    args = Namespace(
        experiment="transport",
        seeds=None, seed_prefix="p2-transport", seed_count=2, workers=1,
        max_decisions=80, max_searches=2, simulations=8, depth=8,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0.5, macro_model=Path("macro.json"),
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-sha", checkpoint=Path("model.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    calls, files = make_commands(args)
    assert files[-1].name == "p2-transport-abba-report.json"
    assert [call[call.index("--rollout-transport") + 1]
            for call in calls[:4]] == ["legacy", "fused", "fused", "legacy"]
    assert all("--no-episode-profile" in call for call in calls[:4])
    assert all(
        call[call.index("--p2-prefix-cache-handles") + 1] == "0"
        for call in calls[:4]
    )
    assert all(
        call[call.index("--actor-fixed-shape-batch-size") + 1] == "4"
        for call in calls[:4]
    )


def test_p2_wire_only_abba_decomposes_fused_payloads_without_puct_profiling():
    data = [
        cohort(wall, batch=16, fixed_shape=4)
        for wall in (40.0, 43.0, 42.0, 41.0)
    ]
    for index, profile in enumerate(data):
        enabled = index in (1, 2)
        profile["identity"]["wire_profile_only"] = enabled
        for row in profile["seed_reports"]:
            for run in row["runs"]:
                run["wire_instrumentation"] = enabled
                run["wire"] = ({
                    "batch_rollout_step_frame": {
                        "calls": 100,
                        "request_bytes": 10_000,
                        "response_bytes": 50_000,
                        "encode_seconds": 0.05,
                        "pipe_seconds": 0.2,
                        "decode_seconds": 0.1,
                    },
                } if enabled else {})
    report = report_actor_abba(*data, experiment="wire-profile")
    assert report["parity"]["passed"] is True
    assert report["wire_profile_overhead_fraction"] == pytest.approx(85/81 - 1)
    stats = report["wire_profile_diagnostics"]
    assert stats[0]["operations"] == {}
    assert stats[1]["operations"]["batch_rollout_step_frame"]["calls"] == 400
    assert stats[1]["total_response_bytes"] == 200_000
    assert stats[1]["worker_summed_pipe_seconds"] == pytest.approx(0.8)

    wrong = deepcopy(data)
    wrong[1]["seed_reports"][0]["runs"][0]["wire_instrumentation"] = False
    with pytest.raises(ValueError, match="instrumented unexpectedly"):
        report_actor_abba(*wrong, experiment="wire-profile")

    wrong = deepcopy(data)
    wrong[3]["identity"]["wire_profile_only"] = True
    with pytest.raises(ValueError, match="unexpected wire-only"):
        report_actor_abba(*wrong, experiment="wire-profile")

    wrong = deepcopy(data)
    wrong[1]["seed_reports"][0]["runs"][0]["wire"][
        "batch_rollout_step_frame"
    ]["decode_seconds"] = -0.2
    with pytest.raises(ValueError, match="invalid wire timer"):
        report_actor_abba(*wrong, experiment="wire-profile")

    wrong = deepcopy(data)
    wrong[0]["seed_reports"][0]["trace_sha256"] = "drift"
    with pytest.raises(ValueError, match="teacher replay drift"):
        report_actor_abba(*wrong, experiment="wire-profile")


def test_p2_wire_only_runner_toggles_just_jsonl_instrumentation(tmp_path):
    args = Namespace(
        experiment="wire-profile",
        seeds=None, seed_prefix="p2-wire", seed_count=4, workers=1,
        max_decisions=80, max_searches=2, simulations=16, depth=8,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0.5, macro_model=Path("macro.json"),
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-sha", checkpoint=Path("model.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    calls, files = make_commands(args)
    assert files[-1].name == "p2-wire-profile-abba-report.json"
    assert ["--wire-profile-only" in c for c in calls[:4]] == [
        False, True, True, False,
    ]
    assert all("--no-episode-profile" in c for c in calls[:4])
    assert all("--actor-profile" not in c for c in calls[:4])
    assert all(
        c[c.index("--rollout-transport") + 1] == "fused"
        for c in calls[:4]
    )
    assert all(
        c[c.index("--p2-prefix-cache-handles") + 1] == "0"
        for c in calls[:4]
    )


def test_p2_native_fused_abba_validates_versioned_timing_and_teacher_replay():
    data = [
        cohort(wall, batch=16, fixed_shape=4)
        for wall in (40.0, 42.0, 43.0, 41.0)
    ]
    for index, cohort_data in enumerate(data):
        enabled = index in (1, 2)
        cohort_data["identity"]["native_fused_profile_only"] = enabled
        for seed in cohort_data["seed_reports"]:
            for run in seed["runs"]:
                run["h16_native_fused_instrumentation"] = enabled
                run["h16_native_fused"] = ({
                    "schemaId": "sts2-p2-h16-fused-native-profile-v1",
                    "enabled": False,
                    "operations": {
                        "fused.environment_step_frame": {
                            "count": 100, "totalSeconds": 0.25,
                            "maxSeconds": 0.01, "allocatedBytes": 2048,
                        },
                    },
                } if enabled else None)
    report = report_actor_abba(*data, experiment="native-fused-profile")
    assert report["parity"]["passed"] is True
    assert report["native_fused_profile_overhead_fraction"] == pytest.approx(
        85 / 81 - 1,
    )
    diagnostics = report["native_fused_diagnostics"]
    assert diagnostics[0]["operations"] == {}
    assert diagnostics[1]["operations"]["fused.environment_step_frame"][
        "count"
    ] == 400
    assert diagnostics[1]["operations"]["fused.environment_step_frame"][
        "allocatedBytes"
    ] == 8192

    wrong = deepcopy(data)
    wrong[2]["identity"]["native_fused_profile_only"] = False
    with pytest.raises(ValueError, match="unexpected native-fused"):
        report_actor_abba(*wrong, experiment="native-fused-profile")

    wrong = deepcopy(data)
    wrong[1]["seed_reports"][0]["runs"][0]["h16_native_fused"][
        "schemaId"
    ] = "incompatible-v2"
    with pytest.raises(ValueError, match="missing versioned native profile"):
        report_actor_abba(*wrong, experiment="native-fused-profile")

    wrong = deepcopy(data)
    wrong[1]["seed_reports"][0]["runs"][0]["h16_native_fused"][
        "operations"
    ]["fused.environment_step_frame"]["totalSeconds"] = -0.1
    with pytest.raises(ValueError, match="invalid native timer"):
        report_actor_abba(*wrong, experiment="native-fused-profile")

    wrong = deepcopy(data)
    wrong[3]["seed_reports"][1]["trace_sha256"] = "drift"
    with pytest.raises(ValueError, match="teacher replay drift"):
        report_actor_abba(*wrong, experiment="native-fused-profile")


def test_p2_native_fused_runner_only_toggles_native_timing(tmp_path):
    args = Namespace(
        experiment="native-fused-profile",
        seeds=None, seed_prefix="p2-native", seed_count=4, workers=1,
        max_decisions=80, max_searches=2, simulations=16, depth=8,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0.5, macro_model=Path("macro.json"),
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-sha", checkpoint=Path("model.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    calls, files = make_commands(args)
    assert files[-1].name == "p2-native-fused-profile-abba-report.json"
    assert [
        "--native-fused-profile-only" in call for call in calls[:4]
    ] == [False, True, True, False]
    assert all("--no-episode-profile" in call for call in calls[:4])
    assert all("--wire-profile-only" not in call for call in calls[:4])
    assert all("--actor-profile" not in call for call in calls[:4])
    assert all(
        call[call.index("--rollout-transport") + 1] == "fused"
        for call in calls[:4]
    )
    assert all(
        call[call.index("--p2-prefix-cache-handles") + 1] == "0"
        for call in calls[:4]
    )
