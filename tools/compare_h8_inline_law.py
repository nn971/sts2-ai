"""Compare pinned frozen H8 source decisions across standalone/inline Chance Law.

The two JSON reports must use the same model, emulator build,
source seeds and reference-law event decisions. No speedup is claimed
unless every terminal source and sampled replay agrees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "model_sha256", "emulator_revision", "emulator_dll_sha256",
        "source_digest", "chance_sampling", "warmup_per_worker",
        "actor_collation",
    ):
        if left.get(key) != right.get(key):
            raise ValueError("Different H8 reference-law input " + key)
    if left.get("inline_reference_law"):
        raise ValueError("Expected standalone law baseline")
    if not right.get("inline_reference_law"):
        raise ValueError("Expected inline law candidate")
    before = left["cohorts"][0]
    after = right["cohorts"][0]
    for key in ("mode", "workers", "episodes", "statuses", "replay_verified"):
        if before[key] != after[key]:
            raise ValueError("H8 law-cohort mismatch " + key)
    if before["mode"] != "native":
        raise ValueError("Compare native H8 source cohorts")
    old_sources, new_sources = before["trajectories"], after["trajectories"]
    if set(old_sources) != set(new_sources):
        raise ValueError("Missing source seed in Chance Law comparison")
    for seed, first in old_sources.items():
        second = new_sources[seed]
        for key in (
            "cell", "status", "outcome", "score", "exit_hp",
            "player_steps", "chance_steps", "decision_sequence",
            "reference_law_steps", "reference_event_digest",
            "trace_digest", "replay_verified", "journal_digest",
            "mechanics_sha256",
        ):
            if first.get(key) != second.get(key):
                raise ValueError(f"Chance Law drift at {seed}: {key}")
    if not all(x["status"] == "terminal" for x in old_sources.values()):
        raise ValueError("Incomplete combats cannot establish target throughput")
    before_steady = before["steady_state"]
    after_steady = after["steady_state"]
    if (before_steady is None) != (after_steady is None):
        raise ValueError("Mismatched steady-state epochs")
    if before_steady:
        n = before_steady["terminal_episodes"]
        if n != after_steady["terminal_episodes"]:
            raise ValueError("Steady-state terminal count changed")
        before_t = before_steady["wall_seconds"]
        after_t = after_steady["wall_seconds"]
    else:
        n = before["episodes"]
        before_t = before["wall_seconds"]
        after_t = after["wall_seconds"]
    return {
        "schema": "sts2-h8-inline-declared-law-comparison-v1",
        "exact_source_player_nature_replay_parity": True,
        "combat_count": n,
        "standalone_law_calls": (
            (before_steady or {}).get("emulator_rpc_calls") or {}
        ).get("h8_reference_law"),
        "inline_law_calls": (
            (after_steady or {}).get("emulator_rpc_calls") or {}
        ).get("h8_reference_law"),
        "standalone_seconds": before_t,
        "inline_seconds": after_t,
        "speedup": before_t / after_t,
        "notes": "One A/B measurement; require A-B-B-A to accept small gains.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("standalone", type=Path)
    p.add_argument("inline", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = compare(json.loads(args.standalone.read_text()),
                     json.loads(args.inline.read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
