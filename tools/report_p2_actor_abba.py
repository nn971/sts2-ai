#!/usr/bin/env python3
"""Strict four-run A1/B1/B2/A2 wall comparison of the H16 actor service.

Every run must have the same pinned model, emulator, legal-work and exact
public+PUCT-teacher replay. Run-level wall includes startup and teardown.
All four runs must disable actor, PUCT and JSONL profiling, while retaining
two within-seed action/teacher replay executions. This is evidence toward
P2.5, not a new training or full source-fidelity certification.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__:
    from .report_p2_cohort_comparison import MATCH_FIELDS, SCHEMA, _rows, _totals
else:
    from report_p2_cohort_comparison import MATCH_FIELDS, SCHEMA, _rows, _totals

REPORT_SCHEMA = "sts2-p2-actor-abba-acceptance-v1"
ORDER = ("A1", "B1", "B2", "A2")
WORK_FIELDS = (
    "outcome", "complete", "decisions", "max_floor", "search_roots",
    "search_simulations", "search_transitions", "real_transitions",
)
CHANGE_FIELDS = {"actor_max_batch", "actor_fixed_shape_batch_size"}
RUN_ARMS = ("control_a", "control_a2")


def _verified_profile(
    profile: Mapping[str, Any], arm: str, *,
    actor_profile: bool = False, wire_profile: bool = False,
    native_fused_profile: bool = False,
) -> dict[str, Any]:
    if profile.get("schema") != SCHEMA:
        raise ValueError(f"{arm}: unexpected P2 cohort schema")
    identity = profile.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{arm}: missing identity")
    if any(identity.get(flag) is not flag_value for flag, flag_value in (
        ("paired", True),
        ("actor_profile", actor_profile),
        ("profile_overhead", False),
        ("actor_batch_parity_probe", False),
        ("native_profile", False),
        ("episode_profile", False),
    )):
        raise ValueError(
            f"{arm}: need --paired --no-episode-profile with all other "
            "measurement instrumentation off"
        )
    if identity.get("wire_profile_only", False) is not wire_profile:
        raise ValueError(f"{arm}: unexpected wire-only profiling mode")
    if (identity.get("native_fused_profile_only", False)
            is not native_fused_profile):
        raise ValueError(f"{arm}: unexpected native-fused profiling mode")
    if identity.get("actor_inference") != "shared":
        raise ValueError(f"{arm}: require shared actor inference")
    workers = identity.get("worker_count")
    if type(workers) is not int or workers < 1:
        raise ValueError(f"{arm}: invalid worker count")
    rows = _rows(profile)
    if identity.get("seeds") != list(rows):
        raise ValueError(f"{arm}: seed-list/order mismatch")
    for seed, item in rows.items():
        runs = item["runs"]
        if tuple(run["arm"] for run in runs) != RUN_ARMS:
            raise ValueError(f"{arm}/{seed}: not exact unprofiled B/B arms")
        for run in runs:
            if (
                run.get("search_instrumentation") is not False
                or run.get("native_instrumentation") is not False
                or run.get("wire_instrumentation") is not wire_profile
                or (run.get("h16_native_fused_instrumentation", False)
                    is not native_fused_profile)
            ):
                raise ValueError(f"{arm}/{seed}: run was instrumented unexpectedly")
        for field in WORK_FIELDS:
            if runs[0].get(field) != runs[1].get(field):
                raise ValueError(f"{arm}/{seed}: paired work drift: {field}")

    actual = {
        "completed_episodes": sum(
            int(run["complete"]) for item in rows.values() for run in item["runs"]
        ),
        "censored_episodes": sum(
            int(not run["complete"]) for item in rows.values() for run in item["runs"]
        ),
        "search_transitions": sum(
            int(run["search_transitions"])
            for item in rows.values() for run in item["runs"]
        ),
        "real_transitions": sum(
            int(run["real_transitions"])
            for item in rows.values() for run in item["runs"]
        ),
    }
    totals = _totals(profile)
    for key, value in actual.items():
        if totals[key] != value:
            raise ValueError(f"{arm}: inconsistent reported {key}")
    return {"identity": identity, "rows": rows, "totals": totals}


def report_actor_abba(
    a1: Mapping[str, Any], b1: Mapping[str, Any],
    b2: Mapping[str, Any], a2: Mapping[str, Any], *,
    experiment: str = "batching",
) -> dict[str, Any]:
    if experiment not in (
        "batching", "actor-profile", "prefix-cache", "transport",
        "wire-profile", "native-fused-profile", "h21-collator",
    ):
        raise ValueError("Unsupported A/B/B/A experiment")
    profiles = (a1, b1, b2, a2)
    verified = [
        _verified_profile(
            profile, label,
            actor_profile=(experiment == "actor-profile"
                           and label in ("B1", "B2")),
            wire_profile=(experiment == "wire-profile"
                          and label in ("B1", "B2")),
            native_fused_profile=(experiment == "native-fused-profile"
                                  and label in ("B1", "B2")),
        )
        for label, profile in zip(ORDER, profiles, strict=True)
    ]
    identities = [p["identity"] for p in verified]
    reference = identities[0]
    cache_caps = [identity.get("p2_prefix_cache_handles", 0)
                  for identity in identities]
    if any(type(cap) is not int or cap < 0 or cap > 128
           for cap in cache_caps):
        raise ValueError("Invalid exact-prefix cache handle cap")
    if experiment == "prefix-cache" and any(
        "p2_prefix_cache_handles" not in identity for identity in identities
    ):
        raise ValueError("Missing P2.1 cache provenance")
    for label, identity in zip(ORDER[1:], identities[1:], strict=True):
        for field in (*MATCH_FIELDS, "ai_sha", "emulator_cli", "worker_count",
                      "torch", "platform", "cpu_count"):
            if (field in CHANGE_FIELDS or
                (field == "actor_profile" and experiment == "actor-profile") or
                (field == "rollout_transport" and experiment == "transport")):
                continue
            if field not in reference or field not in identity:
                raise ValueError(f"{label}: missing provenance {field}")
            if reference[field] != identity[field]:
                raise ValueError(f"{label}: provenance drift: {field}")
    for left, right, name in (
        (identities[0], identities[3], "A1/A2"),
        (identities[1], identities[2], "B1/B2"),
    ):
        for field in CHANGE_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(f"{name}: inconsistent actor mode: {field}")
    if experiment == "h21-collator":
        expected = ("reference", "packed", "packed", "reference")
        for arm, profile, identity, collator in zip(
            ORDER, profiles, identities, expected, strict=True,
        ):
            if (identity.get("representation") != "h21"
                    or identity.get("public_schema_id") != "prototype-ai-v1"
                    or identity.get("codec") != "h21-grounded-reward-shop-event-v4"
                    or identity.get("actor_h21_collator") != collator):
                raise ValueError(f"{arm}: wrong H21 public schema/collator")
            actor = profile.get("shared_actor_metrics")
            if not isinstance(actor, dict) or actor.get("p2_h21_collator") != collator:
                raise ValueError(f"{arm}: missing actual actor collator receipt")
            for field in ("representation", "public_schema_id", "codec"):
                if identity.get(field) != reference.get(field):
                    raise ValueError(f"{arm}: H21 provenance drift: {field}")
            if identity.get("native_fused_profile_only", False) or identity.get(
                "wire_profile_only", False
            ):
                raise ValueError(f"{arm}: instrumentation enabled")
        if any(i.get("actor_fixed_shape_batch_size") != 0 for i in identities):
            raise ValueError("H21 variable-shape actor must use fixed size 0")
        if any(i.get("actor_max_batch") not in (1, 4, 8, 16) for i in identities):
            raise ValueError("Unexpected H21 actor batch cap")
        if (any(i.get("actor_max_batch") != reference.get("actor_max_batch")
                for i in identities)
                or any(i.get("rollout_transport") != "fused" for i in identities)):
            raise ValueError("H21 batch/transport changed across arms")
    if cache_caps[0] != cache_caps[3] or cache_caps[1] != cache_caps[2]:
        raise ValueError("Inconsistent P2 prefix cache caps within A/B arms")
    if experiment != "prefix-cache" and any(cache_caps):
        raise ValueError("Unexpected P2 prefix cache in non-cache experiment")
    if experiment == "batching":
        if all(
            identities[0].get(field) == identities[1].get(field)
            for field in CHANGE_FIELDS
        ):
            raise ValueError("A/B actor configurations are identical")
        if identities[0].get("actor_fixed_shape_batch_size") != 0:
            raise ValueError("Baseline A must use ordinary single-row inference")
        if identities[0].get("actor_max_batch") != 1:
            raise ValueError("Baseline A must use --actor-max-batch 1")
        if not (
            identities[1].get("actor_fixed_shape_batch_size", 0) > 0
            and identities[1].get("actor_max_batch", 0)
            >= identities[1]["actor_fixed_shape_batch_size"]
        ):
            raise ValueError("Candidate B must use opt-in fixed-shape batching")
    else:
        for field in CHANGE_FIELDS:
            if identities[0].get(field) != identities[1].get(field):
                raise ValueError(
                    f"{experiment} experiment changed batching: {field}"
                )
        if experiment != "h21-collator":
            if identities[0].get("actor_fixed_shape_batch_size") != 4:
                raise ValueError("P2 control targets fixed-shape batch 4")
            if identities[0].get("actor_max_batch") != 16:
                raise ValueError("P2 control targets max batch 16")
        if experiment == "prefix-cache" and (
            cache_caps[0] != 0 or cache_caps[1] < 1
        ):
            raise ValueError("P2 prefix-cache control requires off/on cache arms")
        if experiment == "transport" and (
            (identities[0]["rollout_transport"],
             identities[1]["rollout_transport"],
             identities[2]["rollout_transport"],
             identities[3]["rollout_transport"])
            != ("legacy", "fused", "fused", "legacy")
        ):
            raise ValueError(
                "P2 transport control requires legacy/fused/fused/legacy"
            )
    first_rows = verified[0]["rows"]
    for label, compared in zip(ORDER[1:], verified[1:], strict=True):
        rows = compared["rows"]
        if list(rows) != list(first_rows):
            raise ValueError(f"{label}: seed set/order drift")
        for seed, original in first_rows.items():
            candidate = rows[seed]
            if original["trace_sha256"] != candidate["trace_sha256"]:
                raise ValueError(f"{label}/{seed}: exact public/PUCT teacher replay drift")
            for before, after in zip(
                original["runs"], candidate["runs"], strict=True
            ):
                for field in WORK_FIELDS:
                    if before.get(field) != after.get(field):
                        raise ValueError(f"{label}/{seed}: game/search drift: {field}")
    first = verified[0]["totals"]
    for label, candidate in zip(ORDER[1:], verified[1:], strict=True):
        for key in (
            "completed_episodes", "censored_episodes",
            "search_transitions", "real_transitions",
        ):
            if first[key] != candidate["totals"][key]:
                raise ValueError(f"{label}: aggregate work drift: {key}")

    wall = [x["totals"]["wall_seconds"] for x in verified]
    if any(not math.isfinite(value) or value <= 0 for value in wall):
        raise ValueError("Invalid measured wall time")
    a_wall, b_wall = wall[0] + wall[3], wall[1] + wall[2]
    h21_actor_stages: list[dict[str, Any]] | None = None
    if experiment == "h21-collator":
        h21_actor_stages = []
        for arm, profile, elapsed in zip(ORDER, profiles, wall, strict=True):
            metrics = profile["shared_actor_metrics"]
            for count_key in ("requests", "batches"):
                count = metrics.get(count_key)
                if type(count) is not int or count < 1:
                    raise ValueError(f"{arm}: missing H21 actor {count_key}")
            stages = {}
            for key in (
                "host_collate_seconds", "transfer_seconds", "forward_seconds",
            ):
                value = metrics.get(key)
                if (isinstance(value, bool) or not isinstance(value, (float, int))
                        or not math.isfinite(value) or value < 0):
                    raise ValueError(f"{arm}: invalid H21 actor stage {key}")
                stages[key] = float(value)
            h21_actor_stages.append({
                "arm": arm, "collator": metrics["p2_h21_collator"],
                "requests": metrics["requests"], "batches": metrics["batches"],
                "single_actor_process_seconds": stages,
                "host_collate_fraction_of_cohort_wall": (
                    stages["host_collate_seconds"] / elapsed
                ),
            })
    cache_diagnostics: list[dict[str, Any]] | None = None
    if experiment == "prefix-cache":
        available = ["p2_prefix_cache_diagnostics" in p for p in profiles]
        if any(available) and not all(available):
            raise ValueError("Partial P2 cache instrumentation across A/B/B/A")
        if all(available):
            cache_diagnostics = []
            for arm, profile in zip(ORDER, profiles, strict=True):
                stats = profile["p2_prefix_cache_diagnostics"]
                if not isinstance(stats, Mapping):
                    raise ValueError(f"{arm}: invalid cache diagnostics")
                for field in (
                    "roots", "hits", "misses", "emulator_step_calls_avoided",
                    "receipt_calls", "skipped_capacity", "skipped_provider",
                ):
                    v = stats.get(field)
                    if type(v) is not int or v < 0:
                        raise ValueError(f"{arm}: invalid cache count: {field}")
                receipt_seconds = stats.get("receipt_seconds")
                if (
                    isinstance(receipt_seconds, bool)
                    or not isinstance(receipt_seconds, (int, float))
                    or not math.isfinite(receipt_seconds)
                    or receipt_seconds < 0
                    or stats["emulator_step_calls_avoided"] != stats["hits"]
                ):
                    raise ValueError(f"{arm}: invalid cache step/receipt metrics")
                if arm in ("A1", "A2") and (
                    stats["hits"] or stats["receipt_calls"] or stats["roots"]
                ):
                    raise ValueError(f"{arm}: uncached arm performed cache work")
                cache_diagnostics.append({"arm": arm, **stats})
    transport_bridge: list[dict[str, Any]] | None = None
    if experiment == "transport":
        transport_bridge = []
        for arm, verified_profile in zip(ORDER, verified, strict=True):
            by_operation: dict[str, dict[str, float | int]] = {}
            for row in verified_profile["rows"].values():
                for run in row["runs"]:
                    per_operation = run.get("bridge")
                    if not isinstance(per_operation, Mapping):
                        raise ValueError(f"{arm}: missing JSONL bridge operation profile")
                    for operation, stats in per_operation.items():
                        if not isinstance(stats, Mapping):
                            raise ValueError(f"{arm}: invalid bridge stat for {operation}")
                        calls = stats.get("calls")
                        seconds = stats.get("total_seconds")
                        if (
                            type(calls) is not int or calls < 0
                            or isinstance(seconds, bool)
                            or not isinstance(seconds, (int, float))
                            or not math.isfinite(seconds) or seconds < 0
                        ):
                            raise ValueError(
                                f"{arm}: invalid bridge calls/time for {operation}"
                            )
                        bucket = by_operation.setdefault(
                            operation, {"calls": 0, "total_seconds": 0.0}
                        )
                        bucket["calls"] += calls
                        bucket["total_seconds"] += seconds
            transport_bridge.append({
                "arm": arm,
                "worker_summed_nonadditive_bridge_seconds": sum(
                    item["total_seconds"] for item in by_operation.values()
                ),
                "operations": dict(sorted(by_operation.items())),
            })
    wire_diagnostics: list[dict[str, Any]] | None = None
    if experiment == "wire-profile":
        wire_diagnostics = []
        for arm, verified_profile in zip(ORDER, verified, strict=True):
            per_opcode: dict[str, dict[str, int | float]] = {}
            for row in verified_profile["rows"].values():
                for run in row["runs"]:
                    profile_data = run.get("wire")
                    if not isinstance(profile_data, Mapping):
                        raise ValueError(f"{arm}: missing wire operation breakdown")
                    if arm in ("A1", "A2") and profile_data:
                        raise ValueError(f"{arm}: unprofiled control has wire samples")
                    for op, stats in profile_data.items():
                        if not isinstance(stats, Mapping):
                            raise ValueError(f"{arm}: invalid wire operation: {op}")
                        for key in ("calls", "request_bytes", "response_bytes"):
                            v = stats.get(key)
                            if type(v) is not int or v < 0:
                                raise ValueError(f"{arm}: invalid wire count {op}/{key}")
                        for key in (
                            "encode_seconds", "pipe_seconds", "decode_seconds",
                        ):
                            v = stats.get(key)
                            if (
                                isinstance(v, bool)
                                or not isinstance(v, (int, float))
                                or not math.isfinite(v) or v < 0
                            ):
                                raise ValueError(f"{arm}: invalid wire timer {op}/{key}")
                        bucket = per_opcode.setdefault(op, {
                            "calls": 0, "request_bytes": 0, "response_bytes": 0,
                            "encode_seconds": 0.0,
                            "pipe_seconds": 0.0, "decode_seconds": 0.0,
                        })
                        for key in bucket:
                            bucket[key] += stats[key]
            if arm in ("B1", "B2") and not per_opcode:
                raise ValueError(f"{arm}: missing wire-only timing evidence")
            wire_diagnostics.append({
                "arm": arm,
                "total_request_bytes": sum(
                    op["request_bytes"] for op in per_opcode.values()
                ),
                "total_response_bytes": sum(
                    op["response_bytes"] for op in per_opcode.values()
                ),
                "worker_summed_encode_seconds": sum(
                    op["encode_seconds"] for op in per_opcode.values()
                ),
                "worker_summed_pipe_seconds": sum(
                    op["pipe_seconds"] for op in per_opcode.values()
                ),
                "worker_summed_decode_seconds": sum(
                    op["decode_seconds"] for op in per_opcode.values()
                ),
                "operations": dict(sorted(per_opcode.items())),
            })
    native_fused_diagnostics: list[dict[str, Any]] | None = None
    if experiment == "native-fused-profile":
        native_fused_diagnostics = []
        for arm, verified_profile in zip(ORDER, verified, strict=True):
            per_opcode: dict[str, dict[str, int | float]] = {}
            for row in verified_profile["rows"].values():
                for run in row["runs"]:
                    profile_data = run.get("h16_native_fused")
                    if arm in ("A1", "A2"):
                        if profile_data is not None:
                            raise ValueError(
                                f"{arm}: unprofiled control has native samples"
                            )
                        continue
                    if (
                        not isinstance(profile_data, Mapping)
                        or profile_data.get("schemaId")
                        != "sts2-p2-h16-fused-native-profile-v1"
                        or profile_data.get("enabled") is not False
                        or not isinstance(profile_data.get("operations"), Mapping)
                    ):
                        raise ValueError(f"{arm}: missing versioned native profile")
                    for op, stats in profile_data["operations"].items():
                        if not isinstance(op, str) or not isinstance(stats, Mapping):
                            raise ValueError(f"{arm}: invalid native entry")
                        for key in ("count", "allocatedBytes"):
                            v = stats.get(key)
                            if type(v) is not int or v < 0:
                                raise ValueError(f"{arm}: invalid native count")
                        for key in ("totalSeconds", "maxSeconds"):
                            v = stats.get(key)
                            if (
                                isinstance(v, bool)
                                or not isinstance(v, (int, float))
                                or not math.isfinite(v) or v < 0
                            ):
                                raise ValueError(f"{arm}: invalid native timer")
                        bucket = per_opcode.setdefault(op, {
                            "count": 0, "totalSeconds": 0.0,
                            "allocatedBytes": 0,
                        })
                        for key in bucket:
                            bucket[key] += stats[key]
            if arm in ("B1", "B2") and not per_opcode:
                raise ValueError(f"{arm}: missing C# fused self-time")
            native_fused_diagnostics.append({
                "arm": arm, "operations": dict(sorted(per_opcode.items())),
            })
    return {
        "schema": REPORT_SCHEMA,
        "experiment": experiment,
        "evidence_level": (
            "order-balanced actor-profile overhead diagnostic"
            if experiment == "actor-profile" else
            "order-balanced P2.1 experimental exact-root-conditioned "
            "prefix reuse diagnostic; fair Nature acceptance separate"
            if experiment == "prefix-cache" else
            "order-balanced P2.2 existing legacy-vs-fused JSONL comparison; "
            "transport metrics are worker-summed, cohort wall is authoritative"
            if experiment == "transport" else
            "order-balanced P2.2 fused JSONL wire-only timing/overhead "
            "diagnostic with unchanged PUCT teacher"
            if experiment == "wire-profile" else
            "order-balanced source-versioned H16 native fused C# "
            "environment/serialization self-time diagnostic"
            if experiment == "native-fused-profile" else
            "order-balanced H22 public-v1 frozen checkpoint "
            "reference/packed collation, exact public/PUCT teacher replay"
            if experiment == "h21-collator" else
            "order-balanced uninstrumented P2.5 actor candidate; "
            "full episode/RNG acceptance separate"
        ),
        "source": {
            "ai_sha": reference["ai_sha"],
            "emulator_sha": reference["emulator_sha"],
            "emulator_cli": reference["emulator_cli"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "macro_sha256": reference["macro_sha256"],
        },
        "workload": {
            "seeds": reference["seeds"], "worker_count": reference["worker_count"],
            "ascension": reference["ascension"],
            "map_profile": reference["map_profile"],
            "search_simulations": reference["simulations"],
            "search_depth": reference["depth"],
            "max_searches": reference["max_searches"],
            "max_decisions": reference["max_decisions"],
            "search_transitions_per_run": first["search_transitions"],
            "real_transitions_per_run": first["real_transitions"],
            "completed_episode_executions_per_run": first["completed_episodes"],
            "censored_episode_executions_per_run": first["censored_episodes"],
        },
        "parity": {
            "passed": True, "seed_count": len(first_rows),
            "within_run_replay": True, "cross_run_public_teacher": True,
            "exact_search_and_episode_work": True,
        },
        "configurations": {
            ("A_unprofiled_fixed4" if experiment == "actor-profile"
             else "A_uncached_fixed4" if experiment == "prefix-cache"
             else "A_legacy_fixed4" if experiment == "transport"
             else "A_H21_reference" if experiment == "h21-collator"
             else "A_batch1"): {
                "actor_max_batch": identities[0]["actor_max_batch"],
                "h21_collator": identities[0].get("actor_h21_collator"),
                "fixed_shape_batch_size": identities[0]["actor_fixed_shape_batch_size"],
                "actor_profile": identities[0]["actor_profile"],
                "rollout_transport": identities[0]["rollout_transport"],
                "p2_prefix_cache_handles": cache_caps[0],
            },
            ("B_profiled_fixed4" if experiment == "actor-profile"
             else "B_cached_fixed4" if experiment == "prefix-cache"
             else "B_fused_fixed4" if experiment == "transport"
             else "B_H21_packed" if experiment == "h21-collator"
             else "B_fixed_shape"): {
                "actor_max_batch": identities[1]["actor_max_batch"],
                "h21_collator": identities[1].get("actor_h21_collator"),
                "fixed_shape_batch_size": identities[1]["actor_fixed_shape_batch_size"],
                "actor_profile": identities[1]["actor_profile"],
                "rollout_transport": identities[1]["rollout_transport"],
                "p2_prefix_cache_handles": cache_caps[1],
            },
        },
        "order": list(ORDER),
        "cohorts": [
            {"arm": arm, **verified[i]["totals"]}
            for i, arm in enumerate(ORDER)
        ],
        **(
            {"h21_shared_actor_stages": h21_actor_stages}
            if h21_actor_stages is not None else {}
        ),
        **(
            {"prefix_cache_diagnostics": cache_diagnostics}
            if cache_diagnostics is not None else {}
        ),
        **(
            {"transport_bridge_operation_totals": transport_bridge}
            if transport_bridge is not None else {}
        ),
        **(
            {"wire_profile_diagnostics": wire_diagnostics,
             "wire_profile_overhead_fraction": b_wall / a_wall - 1}
            if wire_diagnostics is not None else {}
        ),
        **(
            {"native_fused_diagnostics": native_fused_diagnostics,
             "native_fused_profile_overhead_fraction": b_wall / a_wall - 1}
            if native_fused_diagnostics is not None else {}
        ),
        "ratio_of_total_baseline_to_candidate_wall": a_wall / b_wall,
        **(
            {"public_schema_id": reference["public_schema_id"],
             "actor_collation_ratio_source": "matched unprofiled full cohort wall",
             "actor_h21_collators": [i["actor_h21_collator"] for i in identities]}
            if experiment == "h21-collator" else {}
        ),
        "wall_reduction_fraction": 1 - b_wall / a_wall,
        **(
            {
                "actor_profiled_to_unprofiled_wall_ratio": b_wall / a_wall,
                "actor_profile_overhead_fraction": b_wall / a_wall - 1,
            } if experiment == "actor-profile" else {}
        ),
        "order_pair_speedup_A1_over_B1": wall[0] / wall[1],
        "order_pair_speedup_A2_over_B2": wall[3] / wall[2],
        "candidate_faster_in_both_pairs": wall[1] < wall[0] and wall[2] < wall[3],
        "warnings": [
            "Two A/B pairs reduce order confounding but do not establish "
            "statistical confidence or a causal bound",
            "The replay checksum covers public actions/teachers, not hidden "
            "RNG receipts or native source fidelity",
            "Startup, warm-up and teardown contribute to each cohort wall",
            *(
                ["H21 model, search and emulator are frozen; "
                 "only shared-actor CPU collation changes",
                 "Offline 10.43x collation gain is not a full-cohort prediction"]
                if experiment == "h21-collator" else []
            ),
            *(
                ["Censored executions prevent complete-run-rate inference"]
                if first["censored_episodes"] else []
            ),
            *(
                ["B1/B2 enable actor service profiling with extra GPU "
                 "synchronizations; A1/A2 disable it",
                 "Signed overhead may be negative due to scheduling noise"]
                if experiment == "actor-profile" else []
            ),
            *(
                ["P2.1 cache is experimental root-conditioned Native only; "
                 "separate fair-history and Nature sampling bypass required",
                 "Per-root private receipt verification can dominate short prefixes"]
                if experiment == "prefix-cache" else []
            ),
            *(
                ["Native C# profiling is H16/public-v0 only, not native-v1",
                 "B1/B2 enable C# fused-only timers. Self-time excludes "
                 "Python actor and scheduling; compare to full cohort wall"]
                if experiment == "native-fused-profile" else []
            ),
            *(
                ["Fused transport stays enabled in all four arms; B1/B2 "
                 "collect JSONL wire-only timing. Pipe wait includes native "
                 "mechanics, native serialization and scheduling.",
                 "Wire profiling may add overhead, measured by order-balanced wall"]
                if experiment == "wire-profile" else []
            ),
            *(
                ["Legacy split versus already-existing fused calls are an "
                 "infrastructure baseline, not a new protocol optimization",
                 "Worker-summed bridge times overlap at multiple workers; "
                 "never substitute them for measured cohort wall"]
                if experiment == "transport" else []
            ),
            "Independent-seed confirmation and P2.1/P2.2 gates remain separate",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("a1", "b1", "b2", "a2"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--experiment", choices=("batching", "actor-profile", "prefix-cache",
                            "transport", "wire-profile",
                            "native-fused-profile", "h21-collator"),
                   default="batching")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = (args.a1, args.b1, args.b2, args.a2)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Missing successful P2 A/B/B/A profiles: " + ", ".join(missing))
    report = report_actor_abba(
        *[json.loads(path.read_text(encoding="utf-8")) for path in paths],
        experiment=args.experiment,
    )
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite A/B/B/A evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "parity": report["parity"],
        "experiment": args.experiment,
        **(
            {"actor_profile_overhead_fraction": report["actor_profile_overhead_fraction"]}
            if args.experiment == "actor-profile" else {
                "speedup": report["ratio_of_total_baseline_to_candidate_wall"],
                "wall_reduction_fraction": report["wall_reduction_fraction"],
            }
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
