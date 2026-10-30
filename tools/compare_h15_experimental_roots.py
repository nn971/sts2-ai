#!/usr/bin/env python3
"""Counterfactual root-action diagnostic using unverified independent Nature.

Not fair search. Do not train the H15 policy/value network from this data.
Run separately from the pinned long PPO job, using an exact-replay corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import load_recipes
from sts2_ai.models.h15_policy_value import H15CombatEvaluator
from sts2_ai.search.h15_experimental_root_compare import compare_root_actions
from sts2_ai.training.h15_experimental_rollout import RolloutLimits
from probe_h15_native_candidates import is_ancestor, reenter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emulator-root", type=Path, required=True)
    ap.add_argument("--emulator-revision", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--candidate-actions", type=int, default=3)
    ap.add_argument("--samples-per-action", type=int, default=4)
    ap.add_argument("--max-player-actions", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=32)
    ap.add_argument("--max-nature", type=int, default=1024)
    ap.add_argument("--seed-prefix", default="h15-experimental-paired-root-v1")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    if min(args.limit, args.candidate_actions, args.samples_per_action) < 1:
        ap.error("All research budgets must be positive")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a root-comparison report")
    if not is_ancestor(
        args.emulator_root, args.source_revision, args.emulator_revision,
    ):
        raise ValueError("Emulator revision is not a descendant of the corpus")
    records = load_recipes(args.train_corpus)
    if any(recipe.emulator_revision != args.source_revision for recipe in records):
        raise ValueError("Mixed emulator source-revision corpus")
    limits = RolloutLimits(
        args.max_player_actions, args.max_turns, args.max_nature,
    )
    limits.validate()
    teacher = H15CombatEvaluator.load(args.teacher, device=args.device)
    counts: Counter[str] = Counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with JsonlEmulatorBackend(
        emulator_root=args.emulator_root,
        expected_emulator_revision=args.emulator_revision,
        build=args.build,
    ) as backend, args.output.open("x", encoding="utf-8") as report:
        for index, recipe in enumerate(records[:args.limit]):
            source = reenter(backend, recipe)
            try:
                seed_prefix = hashlib.sha256(
                    (args.seed_prefix + "\n" + recipe.seed + "\n" +
                     recipe.exact_hash).encode("utf-8")
                ).hexdigest()
                result = compare_root_actions(
                    backend, source, teacher, seed_prefix=seed_prefix,
                    candidate_limit=args.candidate_actions,
                    samples_per_action=args.samples_per_action,
                    limits=limits,
                )
            finally:
                backend.release_many((source,))
            row = {
                "index": index,
                "source_seed": recipe.seed,
                "source_exact_hash": recipe.exact_hash,
                "source_floor": recipe.floor,
                "source_tier": recipe.tier,
                "source_emulator_revision": args.source_revision,
                "experimental_emulator_revision": args.emulator_revision,
                "teacher_sha256": teacher.checkpoint_sha256,
                "teacher_codec": teacher.codec_schema,
                **result,
            }
            counts["roots"] += 1
            counts["all_candidates_complete"] += int(
                result["comparable_all_completed"]
            )
            for candidate in result["candidates"]:
                counts.update(candidate["statuses"])
            report.write(json.dumps(row, sort_keys=True) + "\n")
            report.flush()
            print("[h15-experimental-compare] " + json.dumps({
                "index": index,
                "floor": recipe.floor,
                "tier": recipe.tier,
                "candidate_count": result["candidate_count"],
                "all_completed": result["comparable_all_completed"],
                "preferred": result["experimental_preferred_action_id"],
                "statuses": {
                    candidate["policy_rank"]: candidate["statuses"]
                    for candidate in result["candidates"]
                },
            }, sort_keys=True), flush=True)

    print("[h15-experimental-compare-summary] " + json.dumps({
        "counts": dict(counts),
        "fair_search_roots": 0, "training_targets": 0,
        "probability_status": "UNVERIFIED",
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
