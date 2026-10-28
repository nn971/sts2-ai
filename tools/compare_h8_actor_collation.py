"""Compare fixed-source H8 1,024-combat legacy vs packed actor reports.

Reject different source/world/checkpoint/decisions, including event-scoped
Nature selections; only then report observed steady-state throughput.
Both reports must use the same emulator commit and worker count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    for name in (
        "model_sha256", "emulator_revision", "emulator_dll_sha256",
        "source_digest", "chance_sampling", "warmup_per_worker",
    ):
        if old.get(name) != new.get(name):
            raise ValueError("Different actor comparison input: " + name)
    left, right = old["cohorts"][0], new["cohorts"][0]
    for name in ("mode", "workers", "episodes", "source_digest", "statuses"):
        if left[name] != right[name]:
            raise ValueError("Different H8 cohort: " + name)
    if left["mode"] != "native":
        raise ValueError("Only a source-valid native H8 cohort is comparable")
    if old.get("actor_collation", "legacy") != "legacy":
        raise ValueError("Baseline must use legacy actor collation")
    if new.get("actor_collation") != "packed":
        raise ValueError("Candidate must use packed actor collation")
    for source, oldrow in left["trajectories"].items():
        newrow = right["trajectories"].get(source)
        if newrow is None:
            raise ValueError("Missing source from candidate: " + source)
        for name in (
            "cell", "status", "is_steady", "outcome", "score", "exit_hp",
            "player_decisions", "nature_draws", "intents", "decision_sequence",
            "reference_law_steps", "trace_digest", "reference_event_digest",
            "replay_verified", "mechanics_sha256", "journal_digest",
        ):
            if oldrow.get(name) != newrow.get(name):
                raise ValueError(
                    f"Frozen actor or stochastic-law drift: {source} {name}")
    if len(left["trajectories"]) != len(right["trajectories"]):
        raise ValueError("Unexpected extra candidate source")
    old_steady = left.get("steady_state")
    new_steady = right.get("steady_state")
    if old_steady is None or new_steady is None:
        raise ValueError("Both cohorts must contain steady-state epochs")
    for name in ("measured_episodes", "terminal_episodes",
                 "accepted_for_complete_combat_target"):
        if old_steady[name] != new_steady[name]:
            raise ValueError("Different steady-state coverage: " + name)
    if not old_steady["accepted_for_complete_combat_target"]:
        raise ValueError("Incomplete source coverage cannot yield combats/s gain")
    old_seconds = old_steady["wall_seconds"]
    new_seconds = new_steady["wall_seconds"]
    if old_seconds <= 0 or new_seconds <= 0:
        raise ValueError("Invalid steady-state wall-clock timing")
    oldactor = left["actor_metrics"]
    newactor = right["actor_metrics"]
    for name in ("requests",):
        if oldactor[name] != newactor[name]:
            raise ValueError("Actor request count drift: " + name)
    return {
        "schema": "sts2-h8-packed-actor-1024-comparison-v1",
        "source_and_decision_parity": True,
        "sources": len(left["trajectories"]),
        "steady_fights": old_steady["terminal_episodes"],
        "emulator_revision": old["emulator_revision"],
        "model_sha256": old["model_sha256"],
        "old_seconds": old_seconds,
        "packed_seconds": new_seconds,
        "old_combats_per_second": old_steady["terminal_combats_per_second"],
        "packed_combats_per_second": new_steady["terminal_combats_per_second"],
        "steady_rollout_speedup": old_seconds / new_seconds,
        "old_actor_batch_count": oldactor["batches"],
        "packed_actor_batch_count": newactor["batches"],
        "old_collation_seconds": oldactor["collate_seconds"],
        "packed_collation_seconds": newactor["collate_seconds"],
        "collation_speedup": (
            oldactor["collate_seconds"] / newactor["collate_seconds"]),
        "old_forward_seconds": oldactor["forward_seconds"],
        "packed_forward_seconds": newactor["forward_seconds"],
        "interpretation": (
            "A single sequential A/B comparison; repeat interleaved "
            "A-B-B-A before accepting small performance differences."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("packed", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.legacy.read_text()),
                     json.loads(args.packed.read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
