#!/usr/bin/env python3
"""Strict, same-revision P2 actor A/B comparison with two worker counts.

Accepts four already paired 1/15-worker profiles.  Each configuration must
pass internal exact replay, AND the baseline/candidate public teacher digest
must match for every seed.  This is diagnostic, not full P2.5 A/B/B/A.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__:
    from .report_p2_cohort_comparison import (
        MATCH_FIELDS,
        _rows,
        compare_cohorts,
    )
else:
    from report_p2_cohort_comparison import (
        MATCH_FIELDS,
        _rows,
        compare_cohorts,
    )

REPORT_SCHEMA = "sts2-p2-actor-candidate-comparison-v1"
ALLOWED_BY_EXPERIMENT = {
    "batching": {"actor_max_batch", "actor_fixed_shape_batch_size"},
    "actor-profile": {"actor_profile"},
}
SAME_RUN_FIELDS = (
    "outcome", "complete", "decisions", "max_floor", "search_roots",
    "search_simulations", "search_transitions", "real_transitions",
)


def compare_actor_candidate(
    baseline_one: Mapping[str, Any], baseline_many: Mapping[str, Any],
    candidate_one: Mapping[str, Any], candidate_many: Mapping[str, Any],
    *, experiment: str = "batching",
) -> dict[str, Any]:
    if experiment not in ALLOWED_BY_EXPERIMENT:
        raise ValueError("Unsupported actor A/B experiment")
    allowed_differences = ALLOWED_BY_EXPERIMENT[experiment]
    base = compare_cohorts(baseline_one, baseline_many)
    trial = compare_cohorts(candidate_one, candidate_many)
    ids = [p["identity"] for p in (
        baseline_one, baseline_many, candidate_one, candidate_many,
    )]
    if any(identity.get("paired") is not True for identity in ids):
        raise ValueError("Actor A/B requires explicit --paired exact replay")
    if any(identity.get("profile_overhead") for identity in ids):
        raise ValueError("Actor A/B requires ordinary B/B cohorts, not profile-overhead")
    for original, changed in (
        (baseline_one, candidate_one),
        (baseline_many, candidate_many),
    ):
        a, b = original["identity"], changed["identity"]
        for key in (*MATCH_FIELDS, "ai_sha", "emulator_cli", "worker_count"):
            if key in allowed_differences:
                continue
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"Actor candidate identity mismatch: {key}")
        old_seeds = _rows(original)
        new_seeds = _rows(changed)
        if list(old_seeds) != list(new_seeds):
            raise ValueError("Actor A/B seed/order mismatch")
        for seed in old_seeds:
            earlier = old_seeds[seed]
            later = new_seeds[seed]
            if earlier["trace_sha256"] != later["trace_sha256"]:
                raise ValueError(
                    f"Actor A/B exact teacher/public replay drift: "
                    f"{seed} / workers {a['worker_count']}"
                )
            before_runs, after_runs = earlier["runs"], later["runs"]
            if [x["arm"] for x in before_runs] != [
                x["arm"] for x in after_runs
            ]:
                raise ValueError(f"Actor A/B arm mismatch: {seed}")
            for before, after in zip(before_runs, after_runs, strict=True):
                for field in SAME_RUN_FIELDS:
                    if before.get(field) != after.get(field):
                        raise ValueError(f"Actor A/B workload drift: {seed}/{field}")
    old_many, new_many = base["fifteen_workers"], trial["fifteen_workers"]
    return {
        "schema": REPORT_SCHEMA,
        "experiment": experiment,
        "evidence_level": "same-revision exact-replay actor diagnostic, not P2.5 acceptance",
        "parity": {
            "passed": True, "seed_count": base["parity"]["seed_count"],
            "within_configuration": True, "cross_configuration": True,
        },
        "baseline": {
            "actor_profile": ids[0]["actor_profile"],
            "actor_max_batch": ids[0]["actor_max_batch"],
            "fixed_shape_batch_size": ids[0]["actor_fixed_shape_batch_size"],
            "one_worker": base["one_worker"],
            "fifteen_workers": old_many,
            "actor": base["actor"],
        },
        "candidate": {
            "actor_profile": ids[2]["actor_profile"],
            "actor_max_batch": ids[2]["actor_max_batch"],
            "fixed_shape_batch_size": ids[2]["actor_fixed_shape_batch_size"],
            "one_worker": trial["one_worker"],
            "fifteen_workers": new_many,
            "actor": trial["actor"],
        },
        "wall_speedup_baseline_over_candidate_15_workers": (
            old_many["wall_seconds"] / new_many["wall_seconds"]
        ),
        "search_throughput_ratio_candidate_over_baseline_15_workers": (
            new_many["search_transitions_per_second"]
            / old_many["search_transitions_per_second"]
            if old_many["search_transitions_per_second"] > 0 else None
        ),
        "source": base["source"],
        "warnings": [
            "Each configuration must pass complete B/B and 1/15 replay",
            "This is one baseline/candidate measurement, not multi-order A/B/B/A",
            *(
                ["Fewer than 15 independent seeds cannot saturate 15 workers"]
                if base["parity"]["seed_count"] < 15 else []
            ),
            "Compare profile-on and profile-off timings separately",
            *(
                ["Zero completed episodes yields no completed-run speedup estimate"]
                if old_many["completed_episodes"] == 0
                or new_many["completed_episodes"] == 0 else []
            ),
            "Censored episodes limit completed-run throughput conclusions",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("baseline-one", "baseline-fifteen",
                 "candidate-one", "candidate-fifteen"):
        p.add_argument(f"--{flag}", type=Path, required=True)
    p.add_argument("--experiment", choices=tuple(ALLOWED_BY_EXPERIMENT),
                   default="batching")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = (
        args.baseline_one, args.baseline_fifteen,
        args.candidate_one, args.candidate_fifteen,
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(
            "P2 actor comparison requires four successful cohort profiles. "
            "Missing: " + ", ".join(missing)
        )
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    report = compare_actor_candidate(*inputs, experiment=args.experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "parity": report["parity"],
        "speedup_15_workers": report["wall_speedup_baseline_over_candidate_15_workers"],
    }, indent=2))


if __name__ == "__main__":
    main()
