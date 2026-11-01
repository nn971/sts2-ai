#!/usr/bin/env python3
"""Replay H16 held-out checkpoints and report per-seed Act-1 floor progress.

Read-only relative to the training run: no optimizer state or progress.json is
modified. Replays provide observed floor histories for saved checkpoints; they
are not serialized records of the original evaluation trajectories.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sts2_ai.training.h16_joint_act_pilot import PilotEpisode


def episode_floor_record(episode: PilotEpisode) -> dict[str, Any]:
    """Extract reached/completed floors and first post-room HP observations."""
    seen_floors: list[int] = []
    completed: dict[int, dict[str, int]] = {}
    for state in episode.frames:
        floor = state.get("floor")
        if type(floor) is int and 0 <= floor <= 16:
            seen_floors.append(floor)
        visits = state.get("completed_rooms")
        if not isinstance(visits, list):
            continue
        for visit in visits:
            if not isinstance(visit, dict) or visit.get("act") != 1:
                continue
            room_floor = visit.get("floor")
            if (type(room_floor) is int and 1 <= room_floor <= 16
                    and room_floor not in completed):
                hp, maximum = state.get("hp"), state.get("max_hp")
                if type(hp) is int and type(maximum) is int:
                    completed[room_floor] = {"hp": hp, "max_hp": maximum}
    final_state = episode.frames[-1]
    final_floor = final_state.get("floor")
    return {
        "seed": episode.seed,
        "outcome": episode.outcome,
        "max_floor_reached": max(seen_floors, default=None),
        "max_floor_completed": max(completed, default=0),
        "terminal_floor": final_floor if type(final_floor) is int else None,
        "terminal_hp": final_state.get("hp"),
        "terminal_max_hp": final_state.get("max_hp"),
        "post_room_hp": {str(k): v for k, v in sorted(completed.items())},
        "decisions": len(episode.decisions),
    }


def summarize_floor_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [r["max_floor_reached"] for r in records
               if r["max_floor_reached"] is not None]
    return {
        "episodes": len(records),
        "clears": sum(r["outcome"] == "act_clear" for r in records),
        "mean_max_floor_reached": statistics.mean(reached) if reached else None,
        "median_max_floor_reached": statistics.median(reached) if reached else None,
        "reached_floors": {
            str(floor): sum((r["max_floor_reached"] or 0) >= floor
                            for r in records)
            for floor in (5, 10, 12, 15, 16)
        },
        "completed_floors": {
            str(floor): sum(r["max_floor_completed"] >= floor
                            for r in records)
            for floor in (5, 10, 12, 15, 16)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--emulator-root", type=Path, required=True)
    parser.add_argument("--macro-model", type=Path, required=True)
    parser.add_argument("--rounds", nargs="+", type=int,
                        help="Checkpoint rounds; default is the latest round")
    parser.add_argument("--workers", type=int, default=2,
                        help="Use few workers to avoid disrupting ongoing training")
    parser.add_argument("--actor-inference", choices=("cpu-worker", "shared"),
                        default="cpu-worker")
    parser.add_argument("--actor-device", default="cuda")
    parser.add_argument("--actor-max-batch", type=int, default=16)
    parser.add_argument("--actor-max-delay-ms", type=float, default=2.0)
    parser.add_argument("--output", type=Path,
                        help="Default: <progress directory>/floor-replay-heldout.json")
    args = parser.parse_args()
    if args.workers <= 0 or args.actor_max_batch <= 0 or args.actor_max_delay_ms < 0:
        parser.error("workers and batch must be positive; delay must be nonnegative")
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    rows = progress["rounds"]
    if not rows:
        parser.error("No saved rounds to replay")
    selected = list(dict.fromkeys(args.rounds or [rows[-1]["round"]]))
    by_round = {row["round"]: row for row in rows}
    if any(index not in by_round for index in selected):
        parser.error("Requested round has no checkpoint in progress.json")

    # Import from the neighboring entrypoint so its original collector,
    # seeded action RNG, and emulator provenance remain unchanged.
    from train_h16_joint_act_pilot import collect, sha256, write_json

    source = progress["experiment"]
    options = source["options"]
    if sha256(args.macro_model) != source["macro_sha256"]:
        parser.error("Macro-model digest does not match the original run")
    shared = SimpleNamespace(
        ai_root=Path.cwd(),
        emulator_root=args.emulator_root,
        emulator_revision=source["emulator_revision"],
        macro_model=args.macro_model,
        temperature=options["temperature"],
        macro_prior_weight=options["macro_prior_weight"],
        max_decisions=options["max_decisions"],
        search_probability=0.0,
        max_searches=0,
        simulations=options["simulations"],
        search_depth=options["search_depth"],
        study_seed=options["study_seed"],
        rollout_transport=options["rollout_transport"],
        actor_inference=args.actor_inference,
        actor_device=args.actor_device,
        actor_max_batch=args.actor_max_batch,
        actor_max_delay_ms=args.actor_max_delay_ms,
        workers=args.workers,
    )
    seeds = [
        f"h16-fixedroute-heldout-{options['study_seed']}-{i}"
        for i in range(options["heldout_runs"])
    ]
    results: list[dict[str, Any]] = []
    print("Round  Seed  Outcome       MaxFloor  Completed  TerminalHP")
    for index in selected:
        row = by_round[index]
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        if not checkpoint.is_file():
            parser.error(f"Missing checkpoint: {checkpoint}")
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            parser.error(f"Checkpoint digest mismatch for round {index}")
        episodes = collect(
            shared, checkpoint, seeds=seeds,
            horizon_lambda=row["horizon_lambda"], search=False,
        )
        per_seed = [episode_floor_record(ep) for ep in episodes]
        for i, record in enumerate(per_seed):
            hp = record["terminal_hp"]
            cap = record["terminal_max_hp"]
            print(f"{index:5d}  {i:4d}  {record['outcome']:<12}"
                  f"  {str(record['max_floor_reached']):>8}"
                  f"  {record['max_floor_completed']:>9}"
                  f"  {hp}/{cap}")
        original = row["heldout"]
        decision_count = sum(len(ep.decisions) for ep in episodes)
        agreement = {
            "clears": sum(ep.outcome == "act_clear" for ep in episodes)
            == original["cleared"],
            "decisions": decision_count == original["decisions"],
            "censored": sum(ep.outcome == "censored" for ep in episodes)
            == original["censored"],
        }
        summary = summarize_floor_records(per_seed)
        print(f"R{index}: mean max floor {summary['mean_max_floor_reached']:.2f}; "
              f"reach floor 10: {summary['reached_floors']['10']}/{len(episodes)}; "
              f"replay match: {agreement}")
        results.append({
            "round": index,
            "checkpoint": str(checkpoint),
            "summary": summary,
            "original_aggregate_agreement": agreement,
            "runs": per_seed,
        })
    output = args.output or args.progress.parent / "floor-replay-heldout.json"
    write_json(output, {
        "schema": "sts2-h16-heldout-floor-replay-v1",
        "source_progress": str(args.progress),
        "emulator_revision": source["emulator_revision"],
        "sampling": "checkpoint-independent-h16-player-v2",
        "note": "Replayed saved checkpoint on held-out seeds; not original serialized runs.",
        "rounds": results,
    })
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
