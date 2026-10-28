"""Validate and summarize three source-identical H8 actor-layout pilot reports.

For a fixed model + 1,024 source combat bank, reject any change in player
actions, event-scoped Chance Law, stochastic draws, terminal state or replay.
This is a single three-way comparison; use ABCCBA repetitions to distinguish
scheduling drift from a small true effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODES = ("legacy", "packed", "packed-contiguous")
SOURCE_KEYS = (
    "cell", "status", "is_steady", "outcome", "score", "exit_hp",
    "terminal_enemy_hp", "boundaries", "player_decisions", "nature_draws",
    "intents", "decision_sequence", "player_steps", "chance_steps",
    "reference_law_steps", "trace_digest", "reference_event_digest",
    "replay_verified", "mechanics_sha256", "journal_digest",
    "captured_player_decisions", "behavior_digest",
)
TOP_KEYS = (
    "model_sha256", "parameters", "emulator_revision",
    "emulator_dll_sha256", "chance_sampling", "source_digest",
    "warmup_per_worker", "sampled_mechanics_audit",
    "capture_player_training",
)
COHORT_KEYS = (
    "mode", "workers", "episodes", "source_digest", "statuses",
    "player_decisions", "all_boundaries", "replay_verified",
)
STEADY_KEYS = (
    "warmup_episodes", "measured_episodes", "terminal_episodes",
    "accepted_for_complete_combat_target",
)


def compare(
    reports: list[dict[str, Any]], *, allow_censored: bool = False,
) -> dict[str, Any]:
    if len(reports) != 3:
        raise ValueError("Expected three actor variants")
    for report, mode in zip(reports, MODES, strict=True):
        if report.get("actor_collation", "legacy") != mode:
            raise ValueError("Wrong collation report for " + mode)
        if len(report.get("cohorts", [])) != 1:
            raise ValueError("Need one native H8 cohort for " + mode)
    old = reports[0]
    baseline = old["cohorts"][0]
    if baseline.get("mode") != "native":
        raise ValueError("Only native source-valid H8 cohorts are comparable")
    complete = bool(
        baseline["steady_state"]["accepted_for_complete_combat_target"])
    if not complete and not allow_censored:
        raise ValueError("Legacy source coverage incomplete")
    bases = baseline["trajectories"]
    for mode, report in zip(MODES[1:], reports[1:], strict=True):
        for key in TOP_KEYS:
            if old.get(key) != report.get(key):
                raise ValueError(f"Input mismatch {mode}: {key}")
        candidate = report["cohorts"][0]
        for key in COHORT_KEYS:
            if baseline.get(key) != candidate.get(key):
                raise ValueError(f"Cohort mismatch {mode}: {key}")
        if (baseline["actor_metrics"]["device"] !=
                candidate["actor_metrics"]["device"]):
            raise ValueError("Cannot compare CPU and GPU service time")
        for key in STEADY_KEYS:
            if (baseline["steady_state"].get(key) !=
                    candidate["steady_state"].get(key)):
                raise ValueError(f"Measured-phase mismatch {mode}: {key}")
        if set(bases) != set(candidate["trajectories"]):
            raise ValueError("Missing or extra H8 source in " + mode)
        for seed, source in bases.items():
            other = candidate["trajectories"][seed]
            for key in SOURCE_KEYS:
                if source.get(key) != other.get(key):
                    raise ValueError(
                        f"Frozen model/Nature/replay drift {mode} {seed}: {key}")
        if baseline["actor_metrics"]["requests"] != (
                candidate["actor_metrics"]["requests"]):
            raise ValueError("Number of player inference calls changed")

    metrics = {}
    for mode, report in zip(MODES, reports, strict=True):
        cohort = report["cohorts"][0]
        actor = cohort["actor_metrics"]
        s = cohort["steady_state"]
        n = actor["batches"]
        if n <= 0 or actor["collate_seconds"] <= 0:
            raise ValueError("Invalid actor service timings in " + mode)
        metrics[mode] = {
            "steady_wall_seconds": s["wall_seconds"],
            "steady_completed_combats_per_second":
                s["terminal_combats_per_second"],
            "steady_completed_combats": s["terminal_episodes"],
            "actor_batches": n,
            "actor_requests": actor["requests"],
            "mean_batch_size": actor["requests"] / n,
            "actor_collate_seconds": actor["collate_seconds"],
            "actor_forward_seconds": actor["forward_seconds"],
            "collate_ms_per_batch": 1000 * actor["collate_seconds"] / n,
            "forward_ms_per_batch": 1000 * actor["forward_seconds"] / n,
            "combined_actor_service_seconds": (
                actor["collate_seconds"] + actor["forward_seconds"]),
        }
    return {
        "schema": "sts2-h8-three-way-actor-layout-comparison-v1",
        "exact_source_player_nature_replay_parity": True,
        "complete_combat_throughput_accepted": complete,
        "source_count": len(bases),
        "model_sha256": old["model_sha256"],
        "emulator_revision": old["emulator_revision"],
        "device": baseline["actor_metrics"]["device"],
        "variants": metrics,
        "packed_vs_legacy_rollout_speedup": (
            metrics["legacy"]["steady_wall_seconds"] /
            metrics["packed"]["steady_wall_seconds"]),
        "contiguous_vs_packed_rollout_speedup": (
            metrics["packed"]["steady_wall_seconds"] /
            metrics["packed-contiguous"]["steady_wall_seconds"]),
        "contiguous_vs_packed_forward_ms_per_batch_speedup": (
            metrics["packed"]["forward_ms_per_batch"] /
            metrics["packed-contiguous"]["forward_ms_per_batch"]),
        "interpretation": (
            "Three sequential cohorts with identical sources, model and "
            "reference Nature; repeat ABCCBA before attributing small "
            "speedups to metadata strides. Batch distribution and GPU "
            "clock/host load are separate potential confounders."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("legacy", type=Path)
    ap.add_argument("packed", type=Path)
    ap.add_argument("packed_contiguous", type=Path)
    ap.add_argument("--allow-censored", action="store_true",
                    help="CI-only: verify exact source parity even if some "
                         "encounters are unsupported; no throughput claim")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    reports = [json.loads(path.read_text())
               for path in (a.legacy, a.packed, a.packed_contiguous)]
    result = compare(reports, allow_censored=a.allow_censored)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
