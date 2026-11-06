#!/usr/bin/env python3
"""H23 paired boss-entry test of frozen H22 policy vs depth-8/full-combat PUCT.

Replays the same new Act-1 seeds in every arm. The opt-in search_floor=16
gate suppresses *all search RNG draws* before floor 16; a strict pre-boss
public-state and action-prefix check rejects any divergent boss entries.
The model, macro policy, stochastic provider and training outputs are unchanged.
This is a diagnostic of the existing root-conditioned PUCT, not fair Nature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sts2_ai.training.h16_episode_metrics import combat_results
from sts2_ai.training.h16_joint_act_pilot import H22_SCHEMA

try:
    from tools.train_h16_joint_act_pilot import (
        collect, seed_namespace, sha256, write_json,
    )
except ModuleNotFoundError:
    from train_h16_joint_act_pilot import collect, seed_namespace, sha256, write_json


SCHEMA = "sts2-h23-boss-puct-paired-v1"
ARMS = {
    "policy": (False, 16, 8),
    "depth8_16": (True, 16, 8),
    "full16": (True, 16, 0),
    "full64": (True, 64, 0),
}


def stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8"),
    ).hexdigest()


def describe_episode(episode: Any) -> dict[str, Any]:
    """Extract per-seed boss data without saving hidden state or giant episodes."""
    if not episode.complete:
        raise ValueError(f"Censored boss diagnostic: {episode.seed}")
    entries = [
        i for i, state in enumerate(episode.frames)
        if state.get("floor") == 16
        and isinstance(state.get("combat"), dict)
    ]
    index = entries[0] if entries else None
    action_ids = [
        decision.legal_actions[decision.chosen_index].action_id
        for decision in episode.decisions
    ]
    prefix = action_ids if index is None else action_ids[:index]
    first = episode.frames[index] if index is not None else None
    boss_fights = [
        result for result in combat_results(episode.frames, episode.outcome)
        if result.floor == 16
    ]
    if len(boss_fights) > 1:
        raise ValueError("Multiple floor-16 boss fights are unsupported")
    boss = boss_fights[0] if boss_fights else None
    return {
        "seed": episode.seed,
        "outcome": episode.outcome,
        "complete": episode.complete,
        "reached_boss": first is not None,
        "boss_entry_public_hash": stable_digest(first) if first is not None else None,
        "preboss_action_prefix_hash": stable_digest(prefix),
        "preboss_action_count": len(prefix),
        "boss_entry_hp": first.get("hp") if first else None,
        "boss_entry_max_hp": first.get("max_hp") if first else None,
        "boss_damage_fraction": (boss.enemy_progress if boss is not None else None),
        "boss_won": (boss.won if boss is not None else None),
        "first_boss_action": (
            action_ids[index] if index is not None and index < len(action_ids)
            else None
        ),
        "search_roots": episode.searches,
        "search_transitions": episode.search_transitions,
        "search_overrides": episode.search_overrides,
        "total_decisions": len(action_ids),
    }


def check_preboss_parity(reference: list[dict[str, Any]],
                         candidate: list[dict[str, Any]]) -> None:
    if [r["seed"] for r in reference] != [r["seed"] for r in candidate]:
        raise ValueError("Paired boss diagnostic seeds are not identical")
    for a, b in zip(reference, candidate, strict=True):
        for field in ("reached_boss", "boss_entry_public_hash",
                      "preboss_action_prefix_hash", "preboss_action_count"):
            if a[field] != b[field]:
                raise ValueError(
                    f"Preboss parity failure at seed {a['seed']}, field={field}"
                )
        if not a["reached_boss"] and (
            a["outcome"] != b["outcome"]
            or a["total_decisions"] != b["total_decisions"]
        ):
            raise ValueError(
                f"Unsearched non-boss episode drift at seed {a['seed']}"
            )


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reached = [r for r in rows if r["reached_boss"]]
    wins = [r for r in reached if r["outcome"] == "act_clear"]
    hp = [r["boss_entry_hp"] / r["boss_entry_max_hp"] for r in reached
          if type(r["boss_entry_hp"]) is int
          and type(r["boss_entry_max_hp"]) is int
          and r["boss_entry_max_hp"] > 0]
    losses = [r["boss_damage_fraction"] for r in reached
              if r["outcome"] == "defeat"
              and r["boss_damage_fraction"] is not None]
    return {
        "episodes": len(rows),
        "boss_reached": len(reached),
        "boss_wins": len(wins),
        "boss_clear_rate_given_entry": len(wins) / len(reached) if reached else None,
        "act_clears": sum(r["outcome"] == "act_clear" for r in rows),
        "mean_boss_entry_hp_fraction": statistics.mean(hp) if hp else None,
        "mean_boss_enemy_hp_removed_on_defeat": (
            statistics.mean(losses) if losses else None
        ),
        "boss_damage_samples": len(losses),
        "search_roots": sum(r["search_roots"] for r in rows),
        "search_transitions": sum(r["search_transitions"] for r in rows),
        "search_overrides": sum(r["search_overrides"] for r in rows),
    }


def compare_arms(a: list[dict[str, Any]],
                 b: list[dict[str, Any]]) -> dict[str, Any]:
    check_preboss_parity(a, b)
    paired = [(x, y) for x, y in zip(a, b, strict=True) if x["reached_boss"]]
    return {
        "boss_entry_seeds": len(paired),
        "boss_wins_only_right": sum(
            y["outcome"] == "act_clear" and x["outcome"] != "act_clear"
            for x, y in paired
        ),
        "boss_wins_only_left": sum(
            x["outcome"] == "act_clear" and y["outcome"] != "act_clear"
            for x, y in paired
        ),
        "boss_action_changed": sum(
            x["first_boss_action"] != y["first_boss_action"]
            for x, y in paired
        ),
        "both_lost_boss_damage_delta_right_minus_left": (
            statistics.mean(
                y["boss_damage_fraction"] - x["boss_damage_fraction"]
                for x, y in paired
                if x["outcome"] == y["outcome"] == "defeat"
                and x["boss_damage_fraction"] is not None
                and y["boss_damage_fraction"] is not None
            )
            if any(
                x["outcome"] == y["outcome"] == "defeat"
                and x["boss_damage_fraction"] is not None
                and y["boss_damage_fraction"] is not None
                for x, y in paired
            ) else None
        ),
    }


def run_boss_diagnosis(
    *, progress_path: Path, emulator_root: Path,
    output_root: Path, round_number: int, seed_count: int,
    tag: str, workers: int, execute: bool,
    arms: tuple[str, ...] = ("policy", "depth8_16", "full16", "full64"),
) -> dict[str, Any]:
    if (seed_count <= 0 or workers <= 0 or len(arms) < 2
            or len(set(arms)) != len(arms) or "policy" not in arms
            or any(arm not in ARMS for arm in arms)):
        raise ValueError("Invalid paired search cohort/arms")
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
                      for c in tag):
        raise ValueError("Unsafe seed tag")
    if "final" in tag:
        raise ValueError("Use only diagnostic seeds, not final-audit seeds")
    progress = json.loads(progress_path.read_text())
    source = progress["experiment"]
    opts = source["options"]
    if (source["schema"] != H22_SCHEMA
            or opts["representation"] != "h21"
            or opts["actor_inference"] != "shared"
            or opts["actor_h21_collator"] != "packed"
            or opts["actor_deterministic_cuda"] is not True
            or opts["actor_fixed_shape_batch_size"] != 4
            or opts.get("macro_prior_weight") != 0
            or opts.get("map_profile") != "generated"):
        raise ValueError("Boss diagnostic requires pinned deterministic H22 run")
    if round_number < 1 or round_number > len(progress["rounds"]):
        raise ValueError("Checkpoint round is outside committed range")
    row = progress["rounds"][round_number - 1]
    checkpoint = progress_path.parent / "models" / f"joint-r{round_number:04d}.pt"
    if not checkpoint.is_file() or sha256(checkpoint) != row["checkpoint_sha256"]:
        raise ValueError("Boss diagnostic checkpoint digest mismatch")
    namespace = seed_namespace(opts["ascension"], opts["map_profile"])
    seeds = [
        f"{namespace}-h23boss-{opts['study_seed']}-{tag}-{i}"
        for i in range(seed_count)
    ]
    reserved = {
        r["seed"] for record in progress["rounds"]
        for group in (record["train"], record.get("heldout"))
        if group is not None
        for r in group.get("floor_progress", {}).get("runs", [])
    }
    if not set(seeds).isdisjoint(reserved):
        raise ValueError("Boss diagnostic reuses training/heldout seeds")
    evaluation_horizon = opts.get("evaluation_horizon_lambda")
    if evaluation_horizon is None:
        evaluation_horizon = row["horizon_lambda"]
    if not math.isfinite(evaluation_horizon) or not 0 <= evaluation_horizon < 1:
        raise ValueError("Bad saved evaluation horizon")
    identity = {
        "schema": SCHEMA,
        "training_fingerprint": progress["fingerprint"],
        "emulator_revision": source["emulator_revision"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "round": round_number, "seeds": seeds, "workers": workers,
        "search_floor": 16, "max_searches": 1, "max_simulation_steps": 512,
        "search_rng": "independent_deterministic_v1", 
        "nature": source.get("chance_provenance"),
        "horizon_lambda": evaluation_horizon,
        "arms": list(arms),
    }
    if not execute:
        return {"execute": False, "identity": identity, "output_root": str(output_root)}

    shared = {
        **opts, "ai_root": Path.cwd(),
        "emulator_root": emulator_root,
        "emulator_revision": source["emulator_revision"],
        "macro_model": None, "workers": workers,
        "search_probability": 1.0, "max_searches": 1,
        "h17_adaptive": False, "search_floor": 16,
        "independent_search_rng": True,
        "card_exposure": {},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        path = output_root / f"{arm}.json"
        expected = {**identity, "arm": arm, "budget": ARMS[arm][1],
                    "depth": ARMS[arm][2]}
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("identity") != expected:
                raise ValueError(f"Existing {arm} arm identity mismatch")
            rows = saved["rows"]
            if [r["seed"] for r in rows] != seeds:
                raise ValueError(f"Existing {arm} arm seed mismatch")
        else:
            search, budget, depth = ARMS[arm]
            args = SimpleNamespace(
                **{**shared, "simulations": budget, "search_depth": depth},
            )
            print(f"[h23-boss] {arm}: {seed_count} Act episodes", flush=True)
            episodes = collect(
                args, checkpoint, seeds=seeds,
                horizon_lambda=evaluation_horizon, search=search,
            )
            rows = [describe_episode(ep) for ep in episodes]
            if [r["seed"] for r in rows] != seeds:
                raise ValueError(f"Unexpected {arm} seed order")
            if arm == "policy" and any(r["search_roots"] for r in rows):
                raise ValueError("Search-free control executed PUCT")
            if arm != "policy" and any(r["search_roots"] > 1 for r in rows):
                raise ValueError("More than one boss search per episode")
            if "policy" in results:
                check_preboss_parity(results["policy"], rows)
            write_json(path, {"identity": expected, "rows": rows})
        results[arm] = rows
        print(f"[h23-boss] {arm}: {summarize_arm(rows)}", flush=True)

    control = results["policy"]
    comparisons = {
        arm: compare_arms(control, results[arm])
        for arm in arms if arm != "policy"
    }
    report = {
        "identity": identity,
        "summary": {arm: summarize_arm(rows) for arm, rows in results.items()},
        "vs_policy": comparisons,
        "limitations": [
            "Root-conditioned Native PUCT is privileged and not a fair Nature search.",
            "Same boss entry is verified by public snapshot and preboss action trace.",
            "Identical prefix plus native seed conditions the hidden RNG continuation.",
            "Search simulation seeds are deterministic and independent of behavior sampling.",
            "A searched root can change subsequent actions and future boss states.",
            "An unfinished full-combat 512-step search fails, not a terminal reward.",
            "The cohort is diagnostic; preserve independent validation seeds.",
        ],
    }
    final = output_root / "boss-comparison.json"
    if final.exists():
        raise ValueError("Boss comparison exists; use a fresh output root")
    write_json(final, report)
    return {"execute": True, "report": str(final),
            "summary": report["summary"], "vs_policy": comparisons}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--round", type=int, default=120)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--tag", default="h23-boss-pilot-v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--arm", action="append", dest="arms",
                   choices=sorted(ARMS))
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    try:
        result = run_boss_diagnosis(
            progress_path=args.progress, emulator_root=args.emulator_root,
            output_root=args.output_root, round_number=args.round,
            seed_count=args.seeds, tag=args.tag,
            workers=args.workers, execute=args.execute,
            arms=tuple(args.arms or ARMS),
        )
    except (ValueError, KeyError, FileNotFoundError) as error:
        p.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
