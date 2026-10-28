"""Compare batch-size/window sweeps under exact H8 source and Nature replay.

This checks the source bank, frozen actor, declared law and replay for every
cohort before ranking full-combat throughput. It separates actor batch fill,
queue-to-batch wait, collation, forward/output, and response marshaling.

Use instrumented comparisons for relative ranking only. Verify the final
candidate against uninstrumented 1024-source baseline before changing defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SOURCE_KEYS = (
    "cell", "status", "is_steady", "outcome", "score", "exit_hp",
    "terminal_enemy_hp", "boundaries", "player_decisions", "nature_draws",
    "intents", "decision_sequence", "player_steps", "chance_steps",
    "reference_law_steps", "trace_digest", "reference_event_digest",
    "replay_verified", "mechanics_sha256", "journal_digest",
    "captured_player_decisions", "behavior_digest",
)
IMMUTABLE = (
    "model_sha256", "emulator_revision", "emulator_dll_sha256",
    "source_digest", "parameters", "chance_sampling",
    "warmup_per_worker", "sampled_mechanics_audit",
    "capture_player_training", "actor_collation",
    "actor_device", "actor_batch_trace",
)
COHORT = (
    "mode", "workers", "episodes", "source_digest", "statuses",
    "player_decisions", "all_boundaries", "replay_verified",
)


def compare(
    labeled: list[tuple[str, dict[str, Any]]],
    *, allow_censored: bool = False,
) -> dict[str, Any]:
    if len(labeled) < 2:
        raise ValueError("At least two source-equivalent timing runs required")
    reference = labeled[0][1]
    if len(reference.get("cohorts", [])) != 1:
        raise ValueError("Expected exactly one native H8 cohort per run")
    ref = reference["cohorts"][0]
    if ref["mode"] != "native" or ref["steady_state"] is None:
        raise ValueError("Expected synchronized native H8 steady-state cohort")
    source_bank = ref["trajectories"]
    statuses = {}
    rows = []
    for label, report in labeled:
        if label in statuses:
            raise ValueError("Duplicate run label " + label)
        statuses[label] = True
        for key in IMMUTABLE:
            if reference.get(key) != report.get(key):
                raise ValueError(f"Different frozen workload: {label} {key}")
        if len(report["cohorts"]) != 1:
            raise ValueError("Unexpected multiple cohorts in " + label)
        cohort = report["cohorts"][0]
        for key in COHORT:
            if ref.get(key) != cohort.get(key):
                raise ValueError(f"Cohort drift {label}: {key}")
        if set(source_bank) != set(cohort["trajectories"]):
            raise ValueError("Missing/extra source in " + label)
        for seed, source in source_bank.items():
            other = cohort["trajectories"][seed]
            for key in SOURCE_KEYS:
                if source.get(key) != other.get(key):
                    raise ValueError(
                        f"Player/Nature/replay mismatch {label}: {seed} {key}")
        steady = cohort["steady_state"]
        base_steady = ref["steady_state"]
        for key in ("measured_episodes", "terminal_episodes",
                    "warmup_episodes", "accepted_for_complete_combat_target"):
            if steady.get(key) != base_steady.get(key):
                raise ValueError(f"Steady-state parity drift {label}: {key}")
        if (not steady["accepted_for_complete_combat_target"]
                and not allow_censored):
            raise ValueError(
                "Incomplete combat coverage; no valid throughput claim")
        actor = cohort["actor_metrics"]
        batch_trace = actor.get("batch_trace")
        if batch_trace is None:
            raise ValueError("Missing opt-in per-batch trace in " + label)
        if (batch_trace["batches"] != actor["batches"] or
                batch_trace["requests"] != actor["requests"] or
                steady["actor_requests"] != actor["requests"]):
            raise ValueError("Actor service trace counters diverge in " + label)
        entries = batch_trace["batch_records"]
        if len(entries) != actor["batches"]:
            raise ValueError("Truncated batch records in " + label)
        n = actor["batches"]
        def total_ms(key: str, records: list = entries) -> float:
            return sum(float(x[key]) for x in records)
        rows.append({
            "label": label,
            "actor_max_batch": report["actor_max_batch"],
            "effective_batch_cap": min(
                int(report["actor_max_batch"]), cohort["workers"]),
            "actor_delay_ms": report["actor_delay_ms"],
            "terminal_combats": steady["terminal_episodes"],
            "steady_wall_seconds": steady["wall_seconds"],
            "completed_combats_per_second":
                steady["terminal_combats_per_second"],
            "batches": n,
            "requests": actor["requests"],
            "mean_batch_size": actor["requests"] / n,
            "max_batch_size_seen": actor["max_batch_observed"],
            "padding_object_slots": sum(
                x["count"] * x["padded_objects"] -
                x["valid_objects"] for x in entries),
            "padding_action_slots": sum(
                x["count"] * x["padded_actions"] -
                x["valid_actions"] for x in entries),
            "mean_padded_objects": sum(
                x["count"] * x["padded_objects"] for x in entries) /
                actor["requests"],
            "mean_padded_actions": sum(
                x["count"] * x["padded_actions"] for x in entries) /
                actor["requests"],
            "queue_to_batch_mean_ms": sum(
                x["count"] * x["queue_to_batch_mean_ms"] for x in entries) /
                actor["requests"],
            "batch_fill_ms": total_ms("fill_ms"),
            "collate_seconds": actor["collate_seconds"],
            "forward_seconds": actor["forward_seconds"],
            "reply_ms": total_ms("reply_ms"),
            "forward_ms_per_batch": actor["forward_seconds"] * 1000 / n,
            "batch_fill_ms_per_batch": total_ms("fill_ms") / n,
            "by_batch_size": batch_trace["by_batch_size"],
        })
    ranked = sorted(rows, key=lambda r: r["completed_combats_per_second"],
                    reverse=True)
    return {
        "schema": "sts2-h8-actor-batch-window-sweep-v1",
        "exact_source_player_nature_replay_parity": True,
        "complete_combat_throughput_accepted": bool(
            ref["steady_state"]["accepted_for_complete_combat_target"]),
        "source_count": len(source_bank),
        "frozen_model_sha256": reference["model_sha256"],
        "emulator_revision": reference["emulator_revision"],
        "actor_collation": reference["actor_collation"],
        "device": reference["actor_device"],
        "runs": rows,
        "ranked": [x["label"] for x in ranked],
        "fastest_observed_label": ranked[0]["label"],
        "interpretation": (
            "Measured after synchronized warmup. Each worker has only one "
            "outstanding request, bounding batch size by worker count. "
            "Queue-to-batch includes IPC serialization and batch fill; "
            "service/worker times overlap across concurrent processes. "
            "A single sweep is exploratory; confirm a candidate with "
            "interleaved repeated unprofiled fixed-source cohorts."),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path,
                   help="At least two full H8 JSON report paths")
    p.add_argument("--allow-censored", action="store_true",
                   help="CI only: compare exact source parity if unsupported")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    labeled = [(f"{i+1:02}-{path.stem}", json.loads(path.read_text()))
               for i, path in enumerate(args.reports)]
    report = compare(labeled, allow_censored=args.allow_censored)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    # Print a compact summary, not thousands of individual batch records.
    print(json.dumps({
        "source_parity": report["exact_source_player_nature_replay_parity"],
        "source_count": report["source_count"],
        "complete": report["complete_combat_throughput_accepted"],
        "runs": [{
            key: row[key] for key in (
                "label", "actor_max_batch", "effective_batch_cap",
                "actor_delay_ms", "completed_combats_per_second",
                "mean_batch_size", "queue_to_batch_mean_ms",
                "batch_fill_ms_per_batch", "forward_ms_per_batch",
                "padding_object_slots", "padding_action_slots",
            )} for row in report["runs"]],
    }, indent=2))


if __name__ == "__main__":
    main()
