#!/usr/bin/env python3
"""Compare two pinned P2 cohorts, with strict workload and replay parity checks."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "sts2-p2-parallel-cohort-profile-v1"
REPORT_SCHEMA = "sts2-p2-matched-cohort-comparison-v1"
MATCH_FIELDS = (
    "emulator_sha", "checkpoint_sha256", "macro_sha256", "macro_prior_weight",
    "horizon_lambda", "ascension", "map_profile", "simulations", "no_search",
    "depth", "max_simulation_steps", "max_searches", "max_decisions",
    "rollout_transport", "native_profile", "paired", "profile_overhead",
    "actor_inference", "actor_device", "actor_max_batch", "actor_max_delay_ms",
    "actor_profile", "actor_batch_parity_probe", "seeds",
)


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid numeric field: {name}")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"Invalid nonnegative field: {name}")
    return result


def _rows(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    seeds = profile.get("seed_reports")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Missing seed_reports")
    for item in seeds:
        if not isinstance(item, dict):
            raise ValueError("Invalid seed report")
        seed = item.get("seed")
        if not isinstance(seed, str) or not seed or seed in result:
            raise ValueError("Missing/duplicate seed")
        if not isinstance(item.get("trace_sha256"), str):
            raise ValueError(f"Missing replay digest for {seed}")
        runs = item.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"No runs for {seed}")
        arms = [r.get("arm") for r in runs if isinstance(r, dict)]
        if len(arms) != len(runs) or len(set(arms)) != len(arms):
            raise ValueError(f"Invalid/duplicate run arms for {seed}")
        if len(runs) > 1 and item.get("public_teacher_parity") is not True:
            raise ValueError(f"Missing within-seed replay verification for {seed}")
        for run in runs:
            if run.get("seed") != seed:
                raise ValueError(f"Seed mismatch in run for {seed}")
            _number(run.get("episode_wall_seconds"), "episode_wall_seconds", positive=True)
        result[seed] = item
    return result


def _totals(profile: Mapping[str, Any]) -> dict[str, float]:
    wall = _number(profile.get("cohort_wall_seconds"), "cohort_wall_seconds", positive=True)
    complete = _number(profile.get("completed_episodes"), "completed_episodes")
    censored = _number(profile.get("censored_episodes"), "censored_episodes")
    search = _number(profile.get("search_transitions"), "search_transitions")
    real = _number(profile.get("real_transitions"), "real_transitions")
    return {
        "wall_seconds": wall, "completed_episodes": complete,
        "censored_episodes": censored, "search_transitions": search,
        "real_transitions": real, "completed_episodes_per_second": complete / wall,
        "search_transitions_per_second": search / wall,
        "real_transitions_per_second": real / wall,
    }


def compare_cohorts(single: Mapping[str, Any], parallel: Mapping[str, Any]) -> dict[str, Any]:
    """Reject mismatched inputs before publishing a scalability number."""
    if single.get("schema") != SCHEMA or parallel.get("schema") != SCHEMA:
        raise ValueError("Unsupported cohort profile schema")
    a, b = single.get("identity"), parallel.get("identity")
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("Missing cohort identity")
    if a.get("worker_count") != 1 or b.get("worker_count") != 15:
        raise ValueError("Expected exactly 1-worker and 15-worker profiles in that order")
    if a.get("actor_batch_parity_probe") or b.get("actor_batch_parity_probe"):
        raise ValueError(
            "Actor batch numerical probe adds extra forwards: "
            "cannot report a measured throughput speedup"
        )
    for field in MATCH_FIELDS:
        if field not in a or field not in b or a[field] != b[field]:
            raise ValueError(f"Cohort identity mismatch: {field}")
    if a.get("emulator_cli") != b.get("emulator_cli"):
        raise ValueError("Different compiled emulator binary identities")
    if a.get("ai_sha") != b.get("ai_sha"):
        raise ValueError("Different AI source revisions")
    left, right = _rows(single), _rows(parallel)
    if set(left) != set(right):
        raise ValueError("Different episode seed set")
    if list(left) != list(right):
        raise ValueError("Different episode seed order")
    for seed in left:
        old, new = left[seed], right[seed]
        if old["trace_sha256"] != new["trace_sha256"]:
            raise ValueError(f"PUCT/public action replay drift: {seed}")
        runs_a, runs_b = old["runs"], new["runs"]
        if [x["arm"] for x in runs_a] != [x["arm"] for x in runs_b]:
            raise ValueError(f"Different profile arms: {seed}")
        for x, y in zip(runs_a, runs_b, strict=True):
            for field in (
                "outcome", "complete", "decisions", "max_floor", "search_roots",
                "search_simulations", "search_transitions", "real_transitions",
            ):
                if x.get(field) != y.get(field):
                    raise ValueError(f"Gameplay/search drift: {seed}/{x['arm']}/{field}")
    c1, c15 = _totals(single), _totals(parallel)
    for field in (
        "completed_episodes", "censored_episodes",
        "search_transitions", "real_transitions",
    ):
        if c1[field] != c15[field]:
            raise ValueError(f"Cohort count drift: {field}")
    speedup = c1["wall_seconds"] / c15["wall_seconds"]
    return {
        "schema": REPORT_SCHEMA,
        "evidence_level": "diagnostic paired scalability, not optimization acceptance",
        "identity": {key: a[key] for key in MATCH_FIELDS},
        "source": {"ai_sha": a["ai_sha"], "emulator_sha": a["emulator_sha"]},
        "parity": {
            "passed": True, "seed_count": len(left), "trace_digest_match": True,
            "same_completed_censored_and_search_work": True,
        },
        "one_worker": c1, "fifteen_workers": c15,
        "wall_speedup_15_vs_1": speedup,
        "parallel_efficiency_vs_15": speedup / 15,
        "maximum_concurrently_active_seed_tasks": min(15, len(left)),
        "actor": {
            "one_worker": single.get("shared_actor_metrics"),
            "fifteen_workers": parallel.get("shared_actor_metrics"),
        },
        "single_worker_critical_path": single.get("sequential_worker_wall_attribution"),
        "warnings": [
            "Speedup includes actor/process startup, warm-up and teardown",
            "A/B/B/A arms contribute equally; this compares worker count, not an optimization",
            "Worker-summed RPC, actor and search durations overlap",
            "Fewer than 15 independent seeds cannot saturate the 15-worker pool",
            "Search transitions/s, real transitions/s and completed episodes/s are distinct",
            "Zero completed episodes yields no completed-run throughput estimate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    single = json.loads(args.single.read_text(encoding="utf-8"))
    parallel = json.loads(args.parallel.read_text(encoding="utf-8"))
    report = compare_cohorts(single, parallel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "parity": report["parity"],
        "wall_speedup_15_vs_1": report["wall_speedup_15_vs_1"],
        "parallel_efficiency_vs_15": report["parallel_efficiency_vs_15"],
    }, indent=2))


if __name__ == "__main__":
    main()
