#!/usr/bin/env python3
"""Read-only H16 PUCT budget ablation: replay 64 simulations against saved 16.

Reuses the per-seed R79 policy and PUCT16 records from an existing H16 paired
diagnostic, and evaluates only PUCT64. Can resume a 16-seed pilot to 64 seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Running as a script puts tools/ (rather than the repo root) on sys.path.
try:
    from tools.diagnose_h16_paired_checkpoints import (
        SCHEMA as REFERENCE_SCHEMA,
        extract_episode,
        paired_comparison,
        summarize_runs,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from diagnose_h16_paired_checkpoints import (
        SCHEMA as REFERENCE_SCHEMA,
        extract_episode,
        paired_comparison,
        summarize_runs,
    )

SCHEMA = "sts2-h16-puct-budget-ablation-v1"


def reference_arms(reference: dict[str, Any], round_number: int
                   ) -> dict[str, list[dict[str, Any]]]:
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("Expected an H16 paired diagnostic reference")
    identity = reference["identity"]
    expected = [
        f"{identity['seed_prefix']}-{index:04d}"
        for index in range(identity["seeds"])
    ]
    result: dict[str, list[dict[str, Any]]] = {}
    for mode in ("policy", "puct"):
        matching = [
            row for row in reference["runs"]
            if row["round"] == round_number and row["mode"] == mode
        ]
        seeds = [row["seed"] for row in matching]
        if seeds != expected or any(
                row["result"]["seed"] != row["seed"] for row in matching):
            raise ValueError(
                f"Reference R{round_number} {mode} has incomplete or reordered seeds"
            )
        result[mode] = [row["result"] for row in matching]
    return result


def load_journal(path: Path, identity: dict[str, Any],
                 expected_seeds: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        line = source.readline()
        if not line or json.loads(line) != {"schema": SCHEMA, "identity": identity}:
            raise ValueError("Journal identity mismatch; choose a different output")
        records = [json.loads(line) for line in source if line.strip()]
    if (len(records) > len(expected_seeds)
            or [row["seed"] for row in records] != expected_seeds[:len(records)]
            or any(row["result"]["seed"] != row["seed"] for row in records)):
        raise ValueError("Journal contains missing, duplicate or reordered seeds")
    return records


def exact_paired_binary_p(gained: int, lost: int) -> float:
    """Exact two-sided sign/McNemar test on discordant paired outcomes."""
    n = gained + lost
    if n == 0:
        return 1.0
    return min(
        1.0, 2.0 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2**n,
    )


def floor_delta_interval(left: list[dict[str, Any]],
                         right: list[dict[str, Any]]) -> tuple[float, float]:
    if len(left) != len(right) or any(
            a["seed"] != b["seed"] for a, b in zip(left, right, strict=True)):
        raise ValueError("Bootstrap requires matching paired seeds")
    deltas = [
        b["max_floor_reached"] - a["max_floor_reached"]
        for a, b in zip(left, right, strict=True)
    ]
    if not deltas:
        raise ValueError("Empty paired data")
    rng = random.Random(2026110206)
    size = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(4000)
    )
    return (means[int(0.025 * len(means))],
            means[int(0.975 * len(means))])


def analyze(arms: dict[str, list[dict[str, Any]]],
            sixty_four: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(sixty_four)
    if count < 1:
        raise ValueError("No completed 64-simulation evaluations")
    ref16, policy = arms["puct"][:count], arms["policy"][:count]
    pair16 = paired_comparison(ref16, sixty_four)
    pair_policy = paired_comparison(policy, sixty_four)
    return {
        "summary": {
            "policy": summarize_runs(policy),
            "puct16": summarize_runs(ref16),
            "puct64": summarize_runs(sixty_four),
        },
        "paired": {
            "puct64_vs_puct16": {
                **pair16,
                "floor7_exact_two_sided_p": exact_paired_binary_p(
                    pair16["floor7_only_right"], pair16["floor7_only_left"],
                ),
                "mean_floor_delta_bootstrap_95pct": list(
                    floor_delta_interval(ref16, sixty_four)
                ),
            },
            "puct64_vs_policy": {
                **pair_policy,
                "floor7_exact_two_sided_p": exact_paired_binary_p(
                    pair_policy["floor7_only_right"], pair_policy["floor7_only_left"],
                ),
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True,
                   help="Completed 64-seed paired-diagnostic.json from previous run")
    p.add_argument("--progress", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--macro-model", type=Path)
    p.add_argument("--round", type=int, default=79)
    p.add_argument("--simulations", type=int, default=64)
    p.add_argument("--limit", type=int, default=64,
                   help="How many seeds to evaluate *so far*. Increase to resume.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true",
                   help="Run emulator; otherwise only validate input manifest")
    args = p.parse_args()
    if min(args.round, args.simulations, args.limit, args.workers, args.batch_size) <= 0:
        p.error("Round, simulations, limit, workers and batch-size must be positive")

    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    arms = reference_arms(reference, args.round)
    previous = reference["identity"]
    options = previous["options"]
    if (options["simulations"] != 16 or args.simulations <= 16
            or options["search_probability"] != 1.0
            or options["max_searches"] != 0):
        p.error("Expected complete PUCT16 reference; new budget must exceed 16")
    if args.limit > previous["seeds"]:
        p.error("Limit exceeds reference seed count")

    progress = json.loads(args.progress.read_text(encoding="utf-8"))
    source = progress["experiment"]
    by_round = {row["round"]: row for row in progress["rounds"]}
    if args.round not in by_round or str(args.round) not in previous["checkpoints"]:
        p.error("Requested checkpoint absent from progress or reference")
    row = by_round[args.round]
    checkpoint = Path(row["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = Path.cwd() / checkpoint

    from train_h16_joint_act_pilot import collect, sha256, write_json

    if (not checkpoint.is_file()
            or sha256(checkpoint) != row["checkpoint_sha256"]
            or row["checkpoint_sha256"] !=
            previous["checkpoints"][str(args.round)]["sha256"]):
        p.error("Checkpoint missing or SHA256 differs from original paired evaluation")
    if (args.macro_model is None) != (source["macro_sha256"] is None):
        p.error("Provide the original macro model if the experiment used one")
    if args.macro_model is not None and (
            not args.macro_model.is_file()
            or sha256(args.macro_model) != source["macro_sha256"]):
        p.error("Original macro-model SHA256 mismatch")
    if (source["macro_sha256"] != previous["macro_sha256"]
            or source["emulator_revision"] != previous["emulator_revision"]
            or source["map_profile"] != previous["map_profile"]
            or progress["fingerprint"] != previous["progress_fingerprint"]
            or any(source["options"].get(k) != v
                   for k, v in options.items())):
        p.error("Reference belongs to a different H16 training experiment")
    expected_seeds = [
        f"{previous['seed_prefix']}-{i:04d}" for i in range(previous["seeds"])
    ]
    identity = {
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "round": args.round,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "simulations": args.simulations,
        "baseline_simulations": options["simulations"],
        "horizon_lambda": previous["horizon_lambda"],
        "emulator_revision": previous["emulator_revision"],
        "macro_sha256": previous["macro_sha256"],
        "seed_prefix": previous["seed_prefix"],
        "total_seeds": previous["seeds"],
        "batch_size": args.batch_size, "workers": args.workers,
        "actor_inference": previous["actor_inference"],
        "actor_device": previous["actor_device"],
        "actor_max_batch": previous["actor_max_batch"],
        "actor_max_delay_ms": previous["actor_max_delay_ms"],
    }
    journal = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    rows = load_journal(journal, identity, expected_seeds)
    print(json.dumps({
        "schema": SCHEMA, "round": args.round,
        "reference_simulations": 16, "test_simulations": args.simulations,
        "reference_seeds": len(arms["puct"]), "target_this_call": args.limit,
        "already_completed": len(rows),
        "horizon_lambda": previous["horizon_lambda"],
        "output": str(args.output), "execute": args.execute,
    }, indent=2), flush=True)
    if not args.execute:
        return
    if args.output.exists():
        p.error("Completed output already exists; use a new output")
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
        simulations=args.simulations,
        search_depth=options["search_depth"],
        study_seed=options["study_seed"],
        rollout_transport=options["rollout_transport"],
        actor_inference=previous["actor_inference"],
        actor_device=previous["actor_device"],
        actor_max_batch=previous["actor_max_batch"],
        actor_max_delay_ms=previous["actor_max_delay_ms"],
        workers=args.workers,
    )
    journal.parent.mkdir(parents=True, exist_ok=True)
    if not journal.exists():
        with journal.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(
                {"schema": SCHEMA, "identity": identity}, sort_keys=True,
            ) + "\n")
    target = max(args.limit, len(rows))
    while len(rows) < target:
        seed_batch = expected_seeds[len(rows):min(len(rows) + args.batch_size, target)]
        episodes = collect(
            shared, checkpoint, seeds=seed_batch,
            horizon_lambda=previous["horizon_lambda"], search=True,
        )
        if [episode.seed for episode in episodes] != seed_batch:
            raise RuntimeError("Collected episodes differ from intended seeds")
        with journal.open("a", encoding="utf-8") as stream:
            for episode in episodes:
                record = {"seed": episode.seed, "result": extract_episode(episode)}
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                rows.append(record)
            stream.flush()
        partial = analyze(arms, [x["result"] for x in rows])
        difference = partial["paired"]["puct64_vs_puct16"]
        print(f"[h16-budget] {len(rows)}/{previous['seeds']} "
              f"mean-floor delta={difference['mean_floor_delta_right_minus_left']:+.3f} "
              f"floor7 gained/lost={difference['floor7_only_right']}/"
              f"{difference['floor7_only_left']}", flush=True)
    if len(rows) == previous["seeds"]:
        output = {
            "schema": SCHEMA, "identity": identity,
            "analysis": analyze(arms, [x["result"] for x in rows]),
            "puct64_runs": rows,
            "note": ("16-simulation and policy reference records are reused from "
                     "the supplied paired diagnostic on identical run seeds. "
                     "The resulting trajectories can diverge after the first action."),
            "limitations": [
                "Same run seeds do not guarantee identical later combat states.",
                "PUCT uses privileged exact-state forks rather than public-history-fair search.",
                "R79 was previously selected using these seeds; confirm on new seeds.",
            ],
        }
        write_json(args.output, output)
        print(f"[h16-budget] complete: {args.output}", flush=True)
    else:
        print(f"[h16-budget] partial journal saved: {journal}", flush=True)


if __name__ == "__main__":
    main()
