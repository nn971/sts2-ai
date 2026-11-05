"""P2.5 offline uninstrumented A/B/B/A actor protocol contracts."""
from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import pytest

from tools.report_p2_actor_abba import report_actor_abba
from tools.report_p2_cohort_comparison import MATCH_FIELDS
from tools.run_p2_actor_abba import make_commands, preflight_checkpoint


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


def h22_collator_cohorts() -> list[dict]:
    arms = ("reference", "packed", "packed", "reference")
    data = [cohort(wall, batch=4, fixed_shape=0, complete=True)
            for wall in (18.0, 13.0, 12.0, 19.0)]
    for record, collator in zip(data, arms, strict=True):
        record["identity"].update({
            "representation": "h21",
            "public_schema_id": "prototype-ai-v1",
            "codec": "h21-grounded-reward-shop-event-v4",
            "actor_h21_collator": collator,
        })
        record["shared_actor_metrics"] = {
            "p2_h21_collator": collator,
            "requests": 32, "batches": 16,
            "host_collate_seconds": 0.4 if collator == "packed" else 4.0,
            "transfer_seconds": 0.05, "forward_seconds": 1.3,
            "p2_h21_cuda_peak_allocated_bytes": 1024,
            "p2_h21_cuda_peak_reserved_bytes": 2048,
        }
    return data


def test_h22_collator_abba_accepts_exact_h21_variable_shape_replay():
    data = h22_collator_cohorts()
    frozen = deepcopy(data)
    result = report_actor_abba(*data, experiment="h21-collator")
    assert data == frozen
    assert result["parity"]["passed"]
    assert result["public_schema_id"] == "prototype-ai-v1"
    assert result["actor_h21_collators"] == [
        "reference", "packed", "packed", "reference",
    ]
    assert result["ratio_of_total_baseline_to_candidate_wall"] == pytest.approx(
        37 / 25,
    )
    assert result["candidate_faster_in_both_pairs"]
    assert result["h21_shared_actor_stages"][0]["cuda_peak_memory_bytes"] == {
        "p2_h21_cuda_peak_allocated_bytes": 1024,
        "p2_h21_cuda_peak_reserved_bytes": 2048,
    }
    assert [x["single_actor_process_seconds"]["host_collate_seconds"]
            for x in result["h21_shared_actor_stages"]] == [
                4.0, 0.4, 0.4, 4.0,
            ]
    assert result["configurations"]["A_H21_reference"]["h21_collator"] == "reference"
    assert result["configurations"]["B_H21_packed"]["h21_collator"] == "packed"


@pytest.mark.parametrize(("arm", "change", "reason"), [
    (1, lambda p: p["identity"].update({"public_schema_id": "prototype-ai-v0"}),
     "public schema"),
    (1, lambda p: p["identity"].update({"representation": "h16"}),
     "public schema"),
    (1, lambda p: p["identity"].update({"actor_h21_collator": "reference"}),
     "collator"),
    (1, lambda p: p["shared_actor_metrics"].update({"p2_h21_collator": "reference"}),
     "actor collator receipt"),
    (2, lambda p: p["identity"].update({"actor_fixed_shape_batch_size": 4}),
     "inconsistent actor mode"),
    (2, lambda p: p["identity"].update({"actor_max_batch": 16}),
     "inconsistent actor mode"),
    (2, lambda p: p["identity"].update({"rollout_transport": "legacy"}),
     "provenance drift"),
    (2, lambda p: p["identity"].update({"checkpoint_sha256": "wrong"}),
     "provenance drift"),
    (3, lambda p: p["seed_reports"][0].update({"trace_sha256": "wrong"}),
     "teacher replay"),
    (3, lambda p: p["seed_reports"][0]["runs"][0].update({"search_transitions": 1}),
     "paired work drift"),
    (1, lambda p: p["identity"].update({"wire_profile_only": True}),
     "unexpected wire-only profiling"),
    (1, lambda p: p["shared_actor_metrics"].update({"host_collate_seconds": -1}),
     "invalid H21 actor stage"),
    (1, lambda p: p["shared_actor_metrics"].update({"requests": 0}),
     "missing H21 actor requests"),
    (1, lambda p: p["shared_actor_metrics"].update({
        "p2_h21_cuda_peak_allocated_bytes": 0,
    }), "missing CUDA memory receipt"),
    (1, lambda p: p["shared_actor_metrics"].update({
        "p2_h21_cuda_peak_reserved_bytes": 2,
    }), "invalid H21 CUDA allocator memory"),
])
def test_h22_collator_abba_rejects_drift(arm, change, reason):
    data = h22_collator_cohorts()
    change(data[arm])
    with pytest.raises(ValueError, match=reason):
        report_actor_abba(*data, experiment="h21-collator")


def test_h22_collator_abba_runner_changes_only_collator(tmp_path):
    args = Namespace(
        experiment="h21-collator", h21_max_batch=4,
        seeds=None, seed_prefix="p2-h22", seed_count=16, workers=15,
        max_decisions=4096, max_searches=2, simulations=16, depth=0,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0, macro_model=None,
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-v1-sha", checkpoint=Path("h22.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    calls, outputs = make_commands(args)
    assert outputs[-1].name == "p2-h22-collator-abba-report.json"
    assert [call[call.index("--actor-h21-collator") + 1]
            for call in calls[:4]] == [
                "reference", "packed", "packed", "reference",
            ]
    for command in calls[:4]:
        assert command[command.index("--representation") + 1] == "h21"
        assert command[command.index("--actor-max-batch") + 1] == "4"
        assert command[command.index("--actor-fixed-shape-batch-size") + 1] == "0"
        assert command[command.index("--max-decisions") + 1] == "4096"
        assert command[command.index("--depth") + 1] == "0"
        assert "--paired" in command and "--no-episode-profile" in command
        assert "--actor-profile" not in command
        assert "--execute" in command
        assert command[command.index("--rollout-transport") + 1] == "fused"
    assert calls[-1][calls[-1].index("--experiment") + 1] == "h21-collator"


def test_h22_report_cli_accepts_h21_collator_mode(tmp_path):
    """Exercise the CLI parser, not just the Python report function."""
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "tools/report_p2_actor_abba.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--a1", str(tmp_path / "a1.json"),
            "--b1", str(tmp_path / "b1.json"),
            "--b2", str(tmp_path / "b2.json"),
            "--a2", str(tmp_path / "a2.json"),
            "--experiment", "h21-collator",
            "--output", str(tmp_path / "report.json"),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "Missing successful P2 A/B/B/A profiles" in result.stderr
    assert "invalid choice" not in result.stderr


def h22_complete_cohorts() -> list[dict]:
    data = h22_collator_cohorts()
    for p in data:
        for row in p["seed_reports"]:
            for run in row["runs"]:
                run["complete"] = True
                run["outcome"] = "act_clear" if row["seed"] == "seed-a" else "defeat"
        p["completed_episodes"] = 4
        p["censored_episodes"] = 0
    return data


def test_h22_full_act_gate_verifies_complete_cuda_search_and_seed_count():
    inputs = h22_complete_cohorts()
    report = report_actor_abba(
        *inputs, experiment="h21-collator", require_complete=True,
        min_seed_count=2, require_cuda=True, require_search_roots=True,
    )
    assert report["full_act_gate"] == {
        "require_complete": True, "min_seed_count": 2,
        "require_cuda": True, "require_search_roots": True, "passed": True,
    }
    assert report["workload"]["censored_episode_executions_per_run"] == 0


def test_h22_full_act_gate_rejects_censoring_and_insufficient_seeds():
    with pytest.raises(ValueError, match="zero censored"):
        report_actor_abba(
            *h22_collator_cohorts(), experiment="h21-collator",
            require_complete=True, min_seed_count=2,
        )
    with pytest.raises(ValueError, match="at least 16 independent seeds"):
        report_actor_abba(
            *h22_complete_cohorts(), experiment="h21-collator",
            min_seed_count=16,
        )


def test_h22_full_act_gate_rejects_cpu_or_absent_real_search():
    data = h22_complete_cohorts()
    for row in data:
        row["identity"]["actor_device"] = "cpu"
    with pytest.raises(ValueError, match="requires CUDA"):
        report_actor_abba(*data, experiment="h21-collator", require_cuda=True)
    data = h22_complete_cohorts()
    for profile in data:
        for item in profile["seed_reports"]:
            for run in item["runs"]:
                run["search_roots"] = 0
                run["search_transitions"] = 0
        profile["search_transitions"] = 0
    with pytest.raises(ValueError, match="actual combat PUCT work"):
        report_actor_abba(
            *data, experiment="h21-collator", require_search_roots=True,
        )


def test_h22_full_act_runner_enforces_cuda_15_workers_and_full_decisions(tmp_path):
    args = Namespace(
        experiment="h21-collator", require_full_act=True, h21_max_batch=4,
        seeds=None, seed_prefix="h22-gpu", seed_count=16, workers=15,
        max_decisions=4096, max_searches=2, simulations=16, depth=0,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0, macro_model=None,
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="emulator-v1-sha", checkpoint=Path("h22.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    commands, _ = make_commands(args)
    assert all(command[command.index("--workers") + 1] == "15"
               for command in commands[:4])
    assert all(command[command.index("--seeds") + 1].count(",") == 15
               for command in commands[:4])
    assert all(command[command.index("--depth") + 1] == "0"
               for command in commands[:4])
    assert all(command[command.index("--max-decisions") + 1] == "4096"
               for command in commands[:4])
    for required in ("--require-complete", "--require-cuda",
                     "--require-search-roots", "--min-seed-count"):
        assert required in commands[-1]
    for attr, bad in (
        ("workers", 8), ("actor_device", "cpu"), ("seed_count", 8),
        ("max_decisions", 80), ("max_searches", 0),
        ("experiment", "batching"),
    ):
        candidate = Namespace(**vars(args))
        setattr(candidate, attr, bad)
        with pytest.raises(ValueError, match="Full-Act H21 acceptance"):
            make_commands(candidate)


def test_h22_checkpoint_preflight_detects_missing_path(tmp_path):
    with pytest.raises(ValueError, match="Missing frozen checkpoint"):
        preflight_checkpoint(Namespace(
            experiment="h21-collator", checkpoint=tmp_path / "absent.pt",
        ))


def test_h22_frozen_checkpoint_creation_and_schema_preflight(tmp_path):
    # Lightweight P2 contract CI does not install PyTorch; H22 CI runs the
    # same creation/checkpoint verification in its torch-enabled environment.
    torch = pytest.importorskip("torch")

    from sts2_ai.models.shared_transformer import H7Config, SharedObjectTransformer
    from sts2_ai.training.h16_joint_act_pilot import JointActPilotNet
    from tools.create_p2_h22_frozen_checkpoint import create_checkpoint

    torch.manual_seed(11)
    h16 = JointActPilotNet(SharedObjectTransformer(
        H7Config(width=32, layers=1, heads=4, feedforward=64)))
    source = tmp_path / "h16-r79.pt"
    output = tmp_path / "h22-frozen.pt"
    h16.save(source)
    provenance = create_checkpoint(source, output, seed=17)
    assert provenance["representation"] == "h21"
    assert provenance["trained_h21_steps"] == 0
    assert provenance["output_h22"] == str(output.resolve())
    assert output.is_file()
    loaded = JointActPilotNet.load(output)
    assert loaded.representation == "h21"
    preflight_checkpoint(Namespace(experiment="h21-collator", checkpoint=output))
    with pytest.raises(ValueError, match="expected H22 H21"):
        preflight_checkpoint(Namespace(experiment="h21-collator", checkpoint=source))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_checkpoint(source, output, seed=17)
    with pytest.raises(FileNotFoundError, match="Missing H16 source"):
        create_checkpoint(tmp_path / "absent-h16.pt", tmp_path / "new.pt")
    with pytest.raises(ValueError, match="Expected H16 source"):
        create_checkpoint(output, tmp_path / "new.pt")

    # Same source and seed reproduce identical initialized weights.
    second = tmp_path / "h22-frozen-copy.pt"
    create_checkpoint(source, second, seed=17)
    other = JointActPilotNet.load(second)
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, other.state_dict()[key]), key


def test_h22_deterministic_cuda_abba_flag_is_matched_across_arms(tmp_path):
    args = Namespace(
        experiment="h21-collator", actor_deterministic_cuda=True,
        h21_max_batch=4, require_full_act=True,
        seeds=None, seed_prefix="h22-det", seed_count=16, workers=15,
        max_decisions=4096, max_searches=2, simulations=16, depth=0,
        actor_max_delay_ms=2.0, horizon_lambda=0.411395,
        macro_prior_weight=0, macro_model=None,
        ascension=0, ai_root=Path("/tmp/ai"), emulator_root=Path("/tmp/emu"),
        emulator_revision="pinned-sha", checkpoint=Path("h22.pt"),
        actor_device="cuda", map_profile="generated", output_root=tmp_path,
    )
    commands, _ = make_commands(args)
    assert all("--actor-deterministic-cuda" in cmd for cmd in commands[:4])
    assert "--actor-deterministic-cuda" not in commands[-1]
    for experiment, device in (("batching", "cuda"), ("h21-collator", "cpu")):
        broken = Namespace(**vars(args))
        broken.experiment = experiment
        broken.actor_device = device
        with pytest.raises(ValueError, match="Invalid P2 A/B/B/A"):
            make_commands(broken)


def test_h22_deterministic_cuda_report_verifies_actor_receipts():
    data = h22_collator_cohorts()
    for record in data:
        record["identity"]["actor_deterministic_cuda"] = True
        record["shared_actor_metrics"]["p2_deterministic_cuda"] = True
        record["shared_actor_metrics"][
            "p2_cuda_cublas_workspace_config"
        ] = ":4096:8"
    report = report_actor_abba(*data, experiment="h21-collator")
    assert report["actor_deterministic_cuda"] is True

    wrong = deepcopy(data)
    wrong[1]["shared_actor_metrics"]["p2_deterministic_cuda"] = False
    with pytest.raises(ValueError, match="deterministic CUDA provenance"):
        report_actor_abba(*wrong, experiment="h21-collator")

    wrong = deepcopy(data)
    wrong[2]["shared_actor_metrics"]["p2_cuda_cublas_workspace_config"] = ":0:0"
    with pytest.raises(ValueError, match="invalid deterministic CUDA BLAS"):
        report_actor_abba(*wrong, experiment="h21-collator")

    wrong = deepcopy(data)
    wrong[3]["identity"]["actor_deterministic_cuda"] = False
    with pytest.raises(ValueError, match="deterministic CUDA provenance"):
        report_actor_abba(*wrong, experiment="h21-collator")
