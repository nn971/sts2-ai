#!/usr/bin/env python3
"""Paired H16 checkpoint evaluation on fresh, shared Act-1 seeds.

Runs each frozen checkpoint with both the sampled actor and 16x8 local PUCT.
The same seed and common critic horizon are used for every arm. Reports are
descriptive: PUCT uses privileged exact-state forks, not a fair posterior.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sts2_ai.training.h16_episode_metrics import combat_results, episode_floor_record

SCHEMA = "sts2-h16-paired-diagnostic-v1"
MODES = ("policy", "puct")
DEFAULT_ROUNDS = (39, 79, 107)


def make_jobs(rounds: list[int], prefix: str, count: int) -> list[tuple[int, str, str]]:
    return [
        (round_number, mode, f"{prefix}-{index:04d}")
        for round_number in rounds for mode in MODES for index in range(count)
    ]


def extract_episode(episode: Any) -> dict[str, Any]:
    floor = episode_floor_record(
        episode.frames, episode.seed, episode.outcome, len(episode.decisions),
    )
    hp7 = floor["post_room_hp"].get("7")
    lost = [
        {"floor": result.floor, "damage": result.enemy_progress}
        for result in combat_results(episode.frames, episode.outcome)
        if not result.won
    ]
    return {
        "seed": episode.seed,
        "outcome": episode.outcome,
        "max_floor_reached": floor["max_floor_reached"],
        "death_floor": floor["death_floor"],
        "floor7_reached": (floor["max_floor_reached"] or 0) >= 7,
        "floor7_won": floor["max_floor_completed"] >= 7,
        "post_floor7_hp": hp7["hp"] / max(1, hp7["max_hp"]) if hp7 else None,
        "lost_fights": lost,
        "searches": episode.searches,
        "search_transitions": episode.search_transitions,
        "search_overrides": episode.search_overrides,
        "decisions": len(episode.decisions),
    }


def summarize_runs(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_floor: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        for loss in rec["lost_fights"]:
            by_floor[str(loss["floor"])].append(loss["damage"])
    reached = sum(rec["floor7_reached"] for rec in records)
    survived = sum(rec["floor7_won"] for rec in records)
    hp7 = [rec["post_floor7_hp"] for rec in records
           if rec["post_floor7_hp"] is not None]
    floors = [rec["max_floor_reached"] for rec in records
              if rec["max_floor_reached"] is not None]
    searches = sum(rec["searches"] for rec in records)
    return {
        "runs": len(records),
        "act_clears": sum(rec["outcome"] == "act_clear" for rec in records),
        "censored": sum(rec["outcome"] == "censored" for rec in records),
        "mean_max_floor_reached": sum(floors) / len(floors) if floors else None,
        "floor7_reached": reached,
        "floor7_won": survived,
        "floor7_win_rate": survived / reached if reached else None,
        "post_floor7_hp_mean": sum(hp7) / len(hp7) if hp7 else None,
        "defeat_damage_by_floor": {
            floor: {"defeats": len(values), "mean": sum(values) / len(values)}
            for floor, values in sorted(by_floor.items(), key=lambda item: int(item[0]))
        },
        "search_roots": searches,
        "search_transitions": sum(rec["search_transitions"] for rec in records),
        "search_overrides": sum(rec["search_overrides"] for rec in records),
        "search_override_rate": (
            sum(rec["search_overrides"] for rec in records) / searches
            if searches else None
        ),
    }


def paired_comparison(left: list[dict[str, Any]],
                      right: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired outcomes: right is the candidate, left is the reference."""
    a = {row["seed"]: row for row in left}
    b = {row["seed"]: row for row in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("Paired cohorts must contain exactly the same unique seeds")
    ordered = sorted(a)
    delta = [
        b[s]["max_floor_reached"] - a[s]["max_floor_reached"]
        for s in ordered if (a[s]["max_floor_reached"] is not None
                             and b[s]["max_floor_reached"] is not None)
    ]
    return {
        "seeds": len(ordered),
        "mean_floor_delta_right_minus_left": (
            sum(delta) / len(delta) if delta else None
        ),
        "floor7_only_right": sum(b[s]["floor7_won"] and not a[s]["floor7_won"]
                                 for s in ordered),
        "floor7_only_left": sum(a[s]["floor7_won"] and not b[s]["floor7_won"]
                                for s in ordered),
        "floor7_both": sum(a[s]["floor7_won"] and b[s]["floor7_won"] for s in ordered),
        "clears_only_right": sum(
            b[s]["outcome"] == "act_clear" and a[s]["outcome"] != "act_clear"
            for s in ordered
        ),
        "clears_only_left": sum(
            a[s]["outcome"] == "act_clear" and b[s]["outcome"] != "act_clear"
            for s in ordered
        ),
    }


def build_report(rows: list[dict[str, Any]], rounds: list[int]) -> dict[str, Any]:
    grouped = {(number, mode): [] for number in rounds for mode in MODES}
    for row in rows:
        grouped[(row["round"], row["mode"])].append(row["result"])
    summary = {
        f"r{number}-{mode}": summarize_runs(grouped[number, mode])
        for number in rounds for mode in MODES
    }
    comparisons = {
        "puct_vs_policy": {
            str(number): paired_comparison(grouped[number, "policy"],
                                          grouped[number, "puct"])
            for number in rounds
        },
        "checkpoint_progress": {
            mode: {
                f"r{a}_to_r{b}": paired_comparison(grouped[a, mode],
                                                 grouped[b, mode])
                for a, b in zip(rounds, rounds[1:])
            }
            for mode in MODES
        },
    }
    return {"arms": summary, "paired": comparisons}


def read_journal(path: Path, identity: dict[str, Any],
                 jobs: list[tuple[int, str, str]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        first = source.readline()
        if not first or json.loads(first) != {"schema": SCHEMA, "identity": identity}:
            raise ValueError("Journal identity differs; choose another output path")
        rows = [json.loads(line) for line in source if line.strip()]
    if len(rows) > len(jobs):
        raise ValueError("Journal contains more jobs than requested")
    for actual, expected in zip(rows, jobs):
        if (actual["round"], actual["mode"], actual["seed"]) != expected:
            raise ValueError("Journal has a missing, reordered, or duplicated job")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--macro-model", type=Path)
    p.add_argument("--rounds", nargs="+", type=int, default=list(DEFAULT_ROUNDS))
    p.add_argument("--seed-prefix", default="h16-v5-fresh-paired-20261102")
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--actor-inference", choices=("shared", "cpu-worker"),
                   default="shared")
    p.add_argument("--actor-device", default="cuda")
    p.add_argument("--actor-max-batch", type=int, default=16)
    p.add_argument("--actor-max-delay-ms", type=float, default=2.0)
    p.add_argument("--horizon-lambda", type=float,
                   help="Common critic horizon; defaults to latest selected round")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true",
                   help="Actually evaluate; omitted means preflight only")
    args = p.parse_args()

    if (min(args.seeds, args.batch_size, args.workers, args.actor_max_batch) < 1
            or not math.isfinite(args.actor_max_delay_ms)
            or args.actor_max_delay_ms < 0
            or not args.seed_prefix.strip()
            or args.seed_prefix.startswith(("h16-fixedroute-train-",
                                            "h16-fixedroute-heldout-"))):
        p.error("Invalid evaluation configuration or reused training/heldout prefix")
    rounds = list(dict.fromkeys(args.rounds))
    if len(rounds) < 2 or any(n < 1 for n in rounds):
        p.error("Select at least two distinct positive checkpoint rounds")
    rounds.sort()

    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    source = progress["experiment"]
    if not source["schema"].endswith("-v5"):
        p.error("This diagnostic expects H16 v5 progress")
    options = source["options"]
    by_round = {row["round"]: row for row in progress["rounds"]}
    if any(n not in by_round for n in rounds):
        p.error("Requested checkpoint missing from progress file")
    horizon = (args.horizon_lambda if args.horizon_lambda is not None
               else by_round[rounds[-1]]["horizon_lambda"])
    if not math.isfinite(horizon) or not 0 <= horizon <= 1:
        p.error("Common horizon lambda must lie in [0, 1]")

    # Reuse the original H16 collector; verify every frozen input first.
    from train_h16_joint_act_pilot import collect, sha256, write_json

    if (args.macro_model is None) != (source["macro_sha256"] is None):
        p.error("Provide the same macro model as the original H16 experiment")
    if args.macro_model is not None and (
            not args.macro_model.is_file()
            or sha256(args.macro_model) != source["macro_sha256"]):
        p.error("Macro model SHA256 differs from original H16 run")
    checkpoints = {}
    for number in rounds:
        row = by_round[number]
        path = Path(row["checkpoint"])
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file() or sha256(path) != row["checkpoint_sha256"]:
            p.error(f"Missing/changed frozen checkpoint R{number}: {path}")
        checkpoints[str(number)] = {
            "path": str(path.resolve()), "sha256": row["checkpoint_sha256"],
            "training_horizon_lambda": row["horizon_lambda"],
        }
    identity = {
        "progress_fingerprint": progress["fingerprint"],
        "checkpoints": checkpoints,
        "macro_sha256": source["macro_sha256"],
        "emulator_revision": source["emulator_revision"],
        "map_profile": source["map_profile"],
        "puct_provenance": source["puct_provenance"],
        "horizon_lambda": horizon,
        "seed_prefix": args.seed_prefix, "seeds": args.seeds,
        "batch_size": args.batch_size, "workers": args.workers,
        "actor_inference": args.actor_inference,
        "actor_device": args.actor_device,
        "actor_max_batch": args.actor_max_batch,
        "actor_max_delay_ms": args.actor_max_delay_ms,
        "options": {key: options[key] for key in (
            "study_seed", "temperature", "macro_prior_weight",
            "max_decisions", "max_searches", "search_probability",
            "simulations", "search_depth", "rollout_transport",
            "combat_credit_weight", "loss_progress_cap",
        )},
    }
    jobs = make_jobs(rounds, args.seed_prefix, args.seeds)
    journal = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    old_rows = read_journal(journal, identity, jobs)
    print(json.dumps({
        "schema": SCHEMA, "arms": len(rounds) * 2,
        "episodes": len(jobs), "already_completed": len(old_rows),
        "horizon_lambda": horizon, "search": {
            "simulations": options["simulations"],
            "depth": options["search_depth"],
            "probability": options["search_probability"],
            "max_searches": options["max_searches"],
        }, "output": str(args.output), "execute": args.execute,
    }, indent=2), flush=True)
    if not args.execute:
        return
    if args.output.exists():
        p.error("Output already exists; choose a new path")
    if options["search_probability"] != 1.0 or options["max_searches"] != 0:
        p.error("Expected full-search H16 v5 configuration (p=1, no cap)")

    shared = SimpleNamespace(
        ai_root=Path.cwd(), emulator_root=args.emulator_root,
        emulator_revision=source["emulator_revision"],
        macro_model=args.macro_model,
        temperature=options["temperature"],
        macro_prior_weight=options["macro_prior_weight"],
        combat_credit_weight=options["combat_credit_weight"],
        loss_progress_cap=options["loss_progress_cap"],
        max_decisions=options["max_decisions"],
        search_probability=options["search_probability"],
        max_searches=options["max_searches"],
        simulations=options["simulations"],
        search_depth=options["search_depth"],
        study_seed=options["study_seed"],
        rollout_transport=options["rollout_transport"],
        actor_inference=args.actor_inference, actor_device=args.actor_device,
        actor_max_batch=args.actor_max_batch,
        actor_max_delay_ms=args.actor_max_delay_ms,
        workers=args.workers,
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    if not journal.exists():
        with journal.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": SCHEMA, "identity": identity},
                                    sort_keys=True) + "\n")
    rows = old_rows[:]
    while len(rows) < len(jobs):
        number, mode, _ = jobs[len(rows)]
        batch = []
        for job in jobs[len(rows):len(rows) + args.batch_size]:
            if job[:2] != (number, mode):
                break
            batch.append(job)
        episodes = collect(
            shared, Path(checkpoints[str(number)]["path"]),
            seeds=[job[2] for job in batch], horizon_lambda=horizon,
            search=(mode == "puct"),
        )
        if [ep.seed for ep in episodes] != [job[2] for job in batch]:
            raise RuntimeError("Episode order differs from requested seed order")
        with journal.open("a", encoding="utf-8") as stream:
            for (_, _, seed), episode in zip(batch, episodes, strict=True):
                record = {
                    "round": number, "mode": mode, "seed": seed,
                    "result": extract_episode(episode),
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                rows.append(record)
            stream.flush()
        print(f"[h16-diagnostic] {len(rows)}/{len(jobs)} "
              f"R{number} {mode}", flush=True)
    report = {
        "schema": SCHEMA, "identity": identity,
        "analysis": build_report(rows, rounds), "runs": rows,
        "limitations": [
            "PUCT uses exact-state forks; results are not public-history-fair.",
            "Seed prefix must be new; repeated evaluations are not independent.",
            "No-search is stochastic actor sampling at the recorded temperature.",
        ],
    }
    write_json(args.output, report)
    print(f"[h16-diagnostic] complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
