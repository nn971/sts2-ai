#!/usr/bin/env python3
"""Report exact-replay profile-on/off overhead from a P2 A/B/B/A cohort.

This measures *per-episode PUCT/JSONL instrumentation*, NOT the cost of the
cohort-wide shared actor's --actor-profile mode.  Every seed must have four
arms and its complete public/teacher replay checksum verified.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__:
    from .report_p2_cohort_comparison import SCHEMA, _rows
else:
    from report_p2_cohort_comparison import SCHEMA, _rows

REPORT_SCHEMA = "sts2-p2-profile-overhead-report-v1"
ARMS = ("control_a", "profile_b1", "profile_b2", "control_a2")


def report_profile_overhead(profile: Mapping[str, Any]) -> dict[str, Any]:
    if profile.get("schema") != SCHEMA:
        raise ValueError("Unsupported P2 profile schema")
    identity = profile.get("identity")
    if not isinstance(identity, dict) or identity.get("profile_overhead") is not True:
        raise ValueError("Requires --profile-overhead A/B/B/A cohort")
    if identity.get("actor_batch_parity_probe") is True:
        raise ValueError("Numerical parity probe performs extra model forwards")
    if identity.get("paired") is not True:
        # The flag can be false even though the A/B/B/A runner actually
        # verifies all arms; require explicit paired contract here.
        raise ValueError("Requires explicit --paired replay contract")

    rows = _rows(profile)
    details = []
    control_total = profile_total = 0.0
    for seed, item in rows.items():
        if item.get("public_teacher_parity") is not True:
            raise ValueError(f"Missing exact replay proof: {seed}")
        runs = item["runs"]
        if tuple(r["arm"] for r in runs) != ARMS:
            raise ValueError(f"Unexpected A/B/B/A arms for {seed}")
        for run in runs:
            expected = run["arm"].startswith("profile_")
            if (run.get("search_instrumentation") is not expected
                    or run.get("wire_instrumentation") is not expected):
                raise ValueError(f"Incorrect instrumentation for {seed}/{run['arm']}")
            if bool(run.get("native_instrumentation")) != (
                bool(identity.get("native_profile")) and expected
            ):
                raise ValueError(f"Incorrect C# instrumentation for {seed}/{run['arm']}")
        for key in (
            "outcome", "complete", "decisions", "max_floor", "search_roots",
            "search_simulations", "search_transitions", "real_transitions",
        ):
            if len({json.dumps(r.get(key), sort_keys=True) for r in runs}) != 1:
                raise ValueError(f"Workload mismatch: {seed}/{key}")
        control = sum(float(r["episode_wall_seconds"]) for r in (
            runs[0], runs[3],
        ))
        measured = sum(float(r["episode_wall_seconds"]) for r in (
            runs[1], runs[2],
        ))
        if not math.isfinite(control) or not math.isfinite(measured) or (
            control <= 0 or measured <= 0
        ):
            raise ValueError(f"Invalid episode wall: {seed}")
        control_total += control
        profile_total += measured
        details.append({
            "seed": seed,
            "unprofiled_mean_seconds": control / 2,
            "profiled_mean_seconds": measured / 2,
            "profiling_wall_ratio": measured / control,
            "extra_wall_fraction": (measured - control) / control,
            "completed": bool(runs[0]["complete"]),
        })
    ratios = [r["profiling_wall_ratio"] for r in details]
    return {
        "schema": REPORT_SCHEMA,
        "evidence_level": "diagnostic overhead; not optimization acceptance",
        "source": {
            "ai_sha": identity["ai_sha"], "emulator_sha": identity["emulator_sha"],
            "model_sha256": identity["checkpoint_sha256"],
            "worker_count": identity["worker_count"],
            "actor_profile": identity["actor_profile"],
            "native_profile": identity["native_profile"],
        },
        "parity": {"passed": True, "seed_count": len(details)},
        "ratio_of_paired_totals": profile_total / control_total,
        "paired_mean_overhead_fraction": profile_total / control_total - 1,
        "per_seed_ratio_median": statistics.median(ratios),
        "per_seed_ratio_min": min(ratios),
        "per_seed_ratio_max": max(ratios),
        "per_seed": details,
        "limitations": [
            "Actor service profiling is cohort-wide and NOT toggled by A/B/B/A arms",
            "A1/B1/B2/A2 order mitigates but does not eliminate time-varying system load",
            "Episode wall excludes startup/teardown; compare complete cohort wall separately",
            "Only the supplied seeds and recorded search configuration were verified",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = report_profile_overhead(json.loads(args.profile.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "parity": report["parity"],
        "overhead_fraction": report["paired_mean_overhead_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
