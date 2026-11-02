#!/usr/bin/env python3
"""H16 R79 paired full-combat PUCT evaluation: 16 and 64 simulations.

Both budgets use full PUCT tree expansion through the combat boundary.
Previous depth-8 policy/16/64 results are immutable baselines, not rerun.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__:
    from .diagnose_h16_paired_checkpoints import (
        extract_episode,
        paired_comparison,
        summarize_runs,
    )
    from .diagnose_h16_puct_budget import (
        exact_paired_binary_p,
        floor_delta_interval,
        reference_arms,
    )
else:
    from diagnose_h16_paired_checkpoints import (
        extract_episode,
        paired_comparison,
        summarize_runs,
    )
    from diagnose_h16_puct_budget import (
        exact_paired_binary_p,
        floor_delta_interval,
        reference_arms,
    )

SCHEMA = "sts2-h16-full-combat-puct-budget-v1"
BUDGET_SCHEMA = "sts2-h16-puct-budget-ablation-v1"
BUDGETS = (16, 64)


def verify_depth8_budget(report: dict[str, Any], *,
                         reference_sha256: str,
                         reference_identity: dict[str, Any],
                         reference_seeds: list[str],
                         round_number: int) -> list[dict[str, Any]]:
    if report.get("schema") != BUDGET_SCHEMA:
        raise ValueError("Expected completed prior 64-simulation depth-8 report")
    identity = report["identity"]
    expected = {
        "reference_sha256": reference_sha256,
        "round": round_number,
        "simulations": 64,
        "baseline_simulations": 16,
        "horizon_lambda": reference_identity["horizon_lambda"],
        "emulator_revision": reference_identity["emulator_revision"],
        "macro_sha256": reference_identity["macro_sha256"],
        "checkpoint_sha256": reference_identity["checkpoints"][str(round_number)]["sha256"],
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Prior depth-8 budget report is from a different experiment")
    records = report["puct64_runs"]
    if ([entry["seed"] for entry in records] != reference_seeds
            or any(entry["result"]["seed"] != entry["seed"] for entry in records)):
        raise ValueError("Prior depth-8 budget report has missing/reordered seeds")
    return [entry["result"] for entry in records]


def read_arm_journal(path: Path, identity: dict[str, Any],
                     seeds: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        header = source.readline()
        if not header or json.loads(header) != {"schema": SCHEMA, "identity": identity}:
            raise ValueError("Full-combat journal identity mismatch")
        rows = [json.loads(line) for line in source if line.strip()]
    if (len(rows) > len(seeds)
            or [row["seed"] for row in rows] != seeds[:len(rows)]
            or any(row["result"]["seed"] != row["seed"] for row in rows)):
        raise ValueError("Full-combat journal has missing, duplicate or reordered seeds")
    return rows


def compare(left: list[dict[str, Any]],
            right: list[dict[str, Any]]) -> dict[str, Any]:
    paired = paired_comparison(left, right)
    return {
        **paired,
        "floor7_exact_two_sided_p": exact_paired_binary_p(
            paired["floor7_only_right"], paired["floor7_only_left"],
        ),
        "mean_floor_delta_bootstrap_95pct": list(floor_delta_interval(left, right)),
    }


def build_analysis(reference: dict[str, list[dict[str, Any]]],
                   depth8_64: list[dict[str, Any]],
                   full16: list[dict[str, Any]],
                   full64: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(full16)
    if n == 0 or len(full64) != n:
        raise ValueError("Full-combat arms must have identical nonzero lengths")
    if [row["seed"] for row in full16] != [row["seed"] for row in full64]:
        raise ValueError("Full-combat runs must use identical seed order")
    policy = reference["policy"][:n]
    capped16 = reference["puct"][:n]
    capped64 = depth8_64[:n]
    return {
        "summary": {
            "policy": summarize_runs(policy),
            "depth8_puct16": summarize_runs(capped16),
            "depth8_puct64": summarize_runs(capped64),
            "full_puct16": summarize_runs(full16),
            "full_puct64": summarize_runs(full64),
        },
        "paired": {
            "full16_vs_depth8_16": compare(capped16, full16),
            "full64_vs_depth8_64": compare(capped64, full64),
            "full64_vs_full16": compare(full16, full64),
            "full16_vs_policy": compare(policy, full16),
            "full64_vs_policy": compare(policy, full64),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True, type=Path,
                   help="Original paired-diagnostic.json (policy and depth8-16)")
    p.add_argument("--depth8-budget", required=True, type=Path,
                   help="Completed puct16-vs64-r79.json (depth8-64)")
    p.add_argument("--progress", required=True, type=Path)
    p.add_argument("--emulator-root", required=True, type=Path)
    p.add_argument("--macro-model", type=Path)
    p.add_argument("--round", type=int, default=79)
    p.add_argument("--max-simulation-steps", type=int, default=512,
                   help="Hard watchdog per PUCT simulation, not a value cutoff")
    p.add_argument("--limit", type=int, default=64,
                   help="Complete this many seeds in EACH budget arm")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true",
                   help="Evaluate; omit for frozen-input validation only")
    args = p.parse_args()
    if min(args.round, args.max_simulation_steps, args.limit,
           args.batch_size, args.workers) < 1:
        p.error("Invalid nonpositive configuration")

    reference_bytes = args.reference.read_bytes()
    reference_sha = hashlib.sha256(reference_bytes).hexdigest()
    reference_file = json.loads(reference_bytes)
    old_identity = reference_file["identity"]
    old_options = old_identity["options"]
    reference = reference_arms(reference_file, args.round)
    expected_seeds = [item["seed"] for item in reference["policy"]]
    if args.limit > len(expected_seeds):
        p.error("Requested seed count exceeds original paired study")
    if (old_options["simulations"] != 16
            or old_options["search_depth"] != 8
            or old_options["search_probability"] != 1.0
            or old_options["max_searches"] != 0):
        p.error("Reference is not the original 16-by-depth8 full-search cohort")
    depth8_64 = verify_depth8_budget(
        json.loads(args.depth8_budget.read_text(encoding="utf-8")),
        reference_sha256=reference_sha, reference_identity=old_identity,
        reference_seeds=expected_seeds, round_number=args.round,
    )
    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    source = progress["experiment"]
    by_round = {row["round"]: row for row in progress["rounds"]}
    if args.round not in by_round:
        p.error("Missing requested checkpoint in progress")
    checkpoint = Path(by_round[args.round]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint
    from train_h16_joint_act_pilot import collect, sha256, write_json

    if (not checkpoint.is_file()
            or sha256(checkpoint) != by_round[args.round]["checkpoint_sha256"]
            or sha256(checkpoint) !=
            old_identity["checkpoints"][str(args.round)]["sha256"]):
        p.error("Frozen checkpoint absent or SHA256 mismatch")
    if (args.macro_model is None) != (source["macro_sha256"] is None):
        p.error("Provide exactly the original H16 macro prior checkpoint")
    if args.macro_model is not None and (
            not args.macro_model.is_file()
            or sha256(args.macro_model) != source["macro_sha256"]):
        p.error("Frozen macro-model SHA256 mismatch")
    if (source["macro_sha256"] != old_identity["macro_sha256"]
            or source["emulator_revision"] != old_identity["emulator_revision"]
            or source["map_profile"] != old_identity["map_profile"]
            or progress["fingerprint"] != old_identity["progress_fingerprint"]
            or any(source["options"].get(key) != value
                   for key, value in old_options.items())):
        p.error("Progress and reference identity disagree")

    identity = {
        "reference_sha256": reference_sha,
        "depth8_budget_sha256": sha256(args.depth8_budget),
        "progress_fingerprint": progress["fingerprint"],
        "checkpoint_sha256": sha256(checkpoint),
        "macro_sha256": source["macro_sha256"],
        "round": args.round,
        "budgets": list(BUDGETS),
        "search_mode": "full_combat_puct",
        "max_simulation_steps": args.max_simulation_steps,
        "horizon_lambda": old_identity["horizon_lambda"],
        "seed_prefix": old_identity["seed_prefix"],
        "total_seeds": len(expected_seeds),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "actor_inference": old_identity["actor_inference"],
        "actor_device": old_identity["actor_device"],
        "actor_max_batch": old_identity["actor_max_batch"],
        "actor_max_delay_ms": old_identity["actor_max_delay_ms"],
        "emulator_revision": old_identity["emulator_revision"],
    }
    journals = {
        budget: args.output.with_suffix(
            args.output.suffix + f".s{budget}.partial.jsonl"
        ) for budget in BUDGETS
    }
    arms = {
        budget: read_arm_journal(journals[budget], {
            **identity, "simulations": budget,
        }, expected_seeds)
        for budget in BUDGETS
    }
    print(json.dumps({
        "schema": SCHEMA, "round": args.round, "budgets": BUDGETS,
        "tree_expansion": "full_depth_to_combat_end",
        "watchdog_steps": args.max_simulation_steps,
        "reference_seeds": len(expected_seeds), "target_per_arm": args.limit,
        "completed": {str(b): len(rows) for b, rows in arms.items()},
        "horizon_lambda": old_identity["horizon_lambda"],
        "execute": args.execute, "output": str(args.output),
    }, indent=2), flush=True)
    if not args.execute:
        return
    if args.output.exists():
        p.error("Completed output already exists; use a new output path")

    shared = SimpleNamespace(
        ai_root=Path.cwd(), emulator_root=args.emulator_root,
        emulator_revision=source["emulator_revision"],
        macro_model=args.macro_model,
        temperature=old_options["temperature"],
        macro_prior_weight=old_options["macro_prior_weight"],
        combat_credit_weight=old_options["combat_credit_weight"],
        loss_progress_cap=old_options["loss_progress_cap"],
        max_decisions=old_options["max_decisions"],
        search_probability=old_options["search_probability"],
        max_searches=old_options["max_searches"],
        simulations=None,
        search_depth=None,
        max_simulation_steps=args.max_simulation_steps,
        study_seed=old_options["study_seed"],
        rollout_transport=old_options["rollout_transport"],
        actor_inference=old_identity["actor_inference"],
        actor_device=old_identity["actor_device"],
        actor_max_batch=old_identity["actor_max_batch"],
        actor_max_delay_ms=old_identity["actor_max_delay_ms"],
        workers=args.workers,
    )
    for budget in BUDGETS:
        arm_identity = {**identity, "simulations": budget}
        journal = journals[budget]
        journal.parent.mkdir(parents=True, exist_ok=True)
        if not journal.exists():
            journal.write_text(
                json.dumps({"schema": SCHEMA, "identity": arm_identity},
                           sort_keys=True) + "\n", encoding="utf-8",
            )
        rows = arms[budget]
        shared.simulations = budget
        target = max(args.limit, len(rows))
        while len(rows) < target:
            seed_batch = expected_seeds[len(rows):min(len(rows) + args.batch_size, target)]
            episodes = collect(
                shared, checkpoint, seeds=seed_batch,
                horizon_lambda=old_identity["horizon_lambda"], search=True,
            )
            if [episode.seed for episode in episodes] != seed_batch:
                raise RuntimeError("Full-combat episode seed order mismatch")
            with journal.open("a", encoding="utf-8") as stream:
                for episode in episodes:
                    record = {
                        "seed": episode.seed, "result": extract_episode(episode),
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    rows.append(record)
                stream.flush()
            paired = compare(
                reference["puct"][:len(rows)],
                [entry["result"] for entry in rows],
            )
            print(
                f"[h16-full] S{budget} {len(rows)}/{len(expected_seeds)} "
                f"mean-floor vs capped16: "
                f"{paired['mean_floor_delta_right_minus_left']:+.3f} "
                f"search transitions="
                f"{sum(item['result']['search_transitions'] for item in rows)}",
                flush=True,
            )
    if all(len(arms[b]) == len(expected_seeds) for b in BUDGETS):
        full16 = [entry["result"] for entry in arms[16]]
        full64 = [entry["result"] for entry in arms[64]]
        write_json(args.output, {
            "schema": SCHEMA, "identity": identity,
            "analysis": build_analysis(reference, depth8_64, full16, full64),
            "runs": {"full16": arms[16], "full64": arms[64]},
            "limitations": [
                "Same seeds do not imply identical combat states after divergent actions.",
                "Root-conditioned PUCT uses privileged exact-emulator forks.",
                "The 512-step watchdog is a failure, not a reward or value estimate.",
                "R79 was previously selected using these seeds; use a fresh-seed confirmation.",
            ],
        })
        print(f"[h16-full] complete: {args.output}", flush=True)
    else:
        print(f"[h16-full] partial: {journals}", flush=True)


if __name__ == "__main__":
    main()
