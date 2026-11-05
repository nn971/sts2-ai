#!/usr/bin/env python3
"""P2 closure gate for an exact-parity, full-Act H22 A/B/B/A benchmark.

Consumes only the strict A1/B1/B2/A2 aggregate report. It cannot infer
CUDA throughput, full-Act completion, or replay correctness from a short
censored pilot. A slower candidate is a valid measured *negative* outcome:
it remains opt-in and the investigation can be closed without promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "sts2-p2-actor-abba-acceptance-v1"
CLOSURE_SCHEMA = "sts2-p2-h22-fullact-closure-v1"
ORDER = ("A1", "B1", "B2", "A2")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assess(
    report: dict[str, Any], *, checkpoint_sha256: str | None = None,
    minimum_promotion_speedup: float = 1.10,
) -> dict[str, Any]:
    """Validate evidence and decide whether packed collation merits promotion."""
    _require(
        math.isfinite(minimum_promotion_speedup)
        and minimum_promotion_speedup >= 1,
        "Invalid packed-promotion threshold",
    )
    _require(
        report.get("schema") == REPORT_SCHEMA
        and report.get("experiment") == "h21-collator",
        "Expected strict H22 H21-collator A/B/B/A report",
    )
    parity = report.get("parity", {})
    for field in (
        "passed", "within_run_replay", "cross_run_public_teacher",
        "exact_search_and_episode_work",
    ):
        _require(parity.get(field) is True, f"Missing exact parity: {field}")
    gate = report.get("full_act_gate", {})
    _require(
        gate.get("passed") is True
        and gate.get("require_complete") is True
        and gate.get("require_cuda") is True
        and gate.get("require_search_roots") is True
        and type(gate.get("min_seed_count")) is int
        and gate["min_seed_count"] >= 16,
        "Report lacks the strict CUDA/full-Act P2.5 acceptance gate",
    )
    workload = report.get("workload", {})
    seeds = workload.get("seeds")
    _require(
        isinstance(seeds, list)
        and len(seeds) >= 16
        and all(isinstance(s, str) and bool(s) for s in seeds)
        and len(set(seeds)) == len(seeds)
        and parity.get("seed_count") == len(seeds)
        and workload.get("worker_count", 0) >= 15
        and workload.get("max_decisions", 0) >= 1024
        and workload.get("search_simulations", 0) >= 8
        and workload.get("max_searches", 0) >= 1
        and workload.get("search_transitions_per_run", 0) > 0
        and workload.get("real_transitions_per_run", 0) > 0
        and workload.get("ascension") == 0
        and workload.get("map_profile") == "generated",
        "Full-Act H22 workload is insufficient or seed set is invalid",
    )
    _require(
        workload.get("censored_episode_executions_per_run") == 0
        and workload.get("completed_episode_executions_per_run")
        == 2 * len(seeds),
        "Full-Act acceptance requires two complete runs per seed",
    )
    source = report.get("source", {})
    for field in ("ai_sha", "emulator_sha", "checkpoint_sha256"):
        _require(
            isinstance(source.get(field), str) and bool(source[field]),
            f"Missing source provenance: {field}",
        )
    cli = source.get("emulator_cli")
    _require(
        isinstance(cli, dict) and bool(cli.get("sha256")),
        "Missing pinned emulator CLI digest",
    )
    if checkpoint_sha256 is not None:
        _require(
            source["checkpoint_sha256"] == checkpoint_sha256,
            "Full-Act benchmark used a different H22 checkpoint",
        )
    _require(
        report.get("public_schema_id") == "prototype-ai-v1"
        and report.get("actor_deterministic_cuda") is True
        and report.get("actor_h21_collators")
        == ["reference", "packed", "packed", "reference"],
        "Wrong H22 representation, CUDA setting or collator order",
    )
    configs = report.get("configurations", {})
    for name, collator in (
        ("A_H21_reference", "reference"), ("B_H21_packed", "packed"),
    ):
        conf = configs.get(name, {})
        _require(
            conf.get("h21_collator") == collator
            and conf.get("actor_max_batch") == 4
            and conf.get("fixed_shape_batch_size") == 4
            and conf.get("rollout_transport") == "fused"
            and conf.get("p2_prefix_cache_handles") == 0
            and conf.get("actor_profile") is False,
            f"Unmatched P2.5 actor/transport configuration: {name}",
        )
    cohorts = report.get("cohorts", [])
    _require(
        isinstance(cohorts, list)
        and len(cohorts) == 4
        and [c.get("arm") for c in cohorts] == list(ORDER),
        "Expected four order-balanced A/B/B/A cohorts",
    )
    wall: list[float] = []
    for arm, cohort in zip(ORDER, cohorts, strict=True):
        seconds = cohort.get("wall_seconds")
        _require(
            isinstance(seconds, (float, int))
            and not isinstance(seconds, bool)
            and math.isfinite(seconds) and seconds > 0
            and cohort.get("censored_episodes") == 0
            and cohort.get("completed_episodes") == 2 * len(seeds)
            and cohort.get("search_transitions", 0) > 0,
            f"{arm}: incomplete full-Act workload or invalid wall receipt",
        )
        wall.append(float(seconds))
    a_wall, b_wall = wall[0] + wall[3], wall[1] + wall[2]
    ratio = a_wall / b_wall
    _require(
        math.isclose(
            report.get("ratio_of_total_baseline_to_candidate_wall", -1),
            ratio, rel_tol=1e-9,
        )
        and math.isclose(
            report.get("order_pair_speedup_A1_over_B1", -1),
            wall[0] / wall[1], rel_tol=1e-9,
        )
        and math.isclose(
            report.get("order_pair_speedup_A2_over_B2", -1),
            wall[3] / wall[2], rel_tol=1e-9,
        ),
        "Benchmark speedup disagrees with cohort wall measurements",
    )
    faster_in_both = wall[1] < wall[0] and wall[2] < wall[3]
    _require(
        report.get("candidate_faster_in_both_pairs") is faster_in_both,
        "Candidate pairwise-speedup flag disagrees with raw wall",
    )
    promote = faster_in_both and ratio >= minimum_promotion_speedup
    return {
        "schema": CLOSURE_SCHEMA,
        "status": "full_act_acceptance_complete",
        "p2_5_gate_passed": True,
        "full_act_episodes_per_arm": 2 * len(seeds),
        "source": source,
        "search_depth": workload["search_depth"],
        "workers": workload["worker_count"],
        "seed_count": len(seeds),
        "strict_replay_parity": True,
        "candidate_faster_in_both_pairs": faster_in_both,
        "wall_speedup": ratio,
        "wall_reduction_fraction": 1 - 1 / ratio,
        "promotion_threshold": minimum_promotion_speedup,
        "packed_collator_decision": (
            "eligible_for_default_promotion" if promote
            else "keep_opt_in_measured_benefit_insufficient"
        ),
        "scope": (
            "H22/public-v1, deterministic CUDA, fixed-4, "
            f"depth-{workload['search_depth']} search, generated Silent A0"
        ),
        "caveat": (
            "Two order-balanced pairs provide a measured workload comparison; "
            "native installed-game parity and other Nature providers remain separate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-promotion-speedup", type=float, default=1.10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Closure output already exists; preserve existing evidence")
    try:
        raw = args.checkpoint.read_bytes()
        report = json.loads(args.report.read_text())
        result = assess(
            report, checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
            minimum_promotion_speedup=args.minimum_promotion_speedup,
        )
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
