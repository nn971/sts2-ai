#!/usr/bin/env python3
"""Run frozen H9/H15 tactical-policy multi-turn combat on an unverified Nature.

Diagnostic coverage only: never write PUCT, critic or policy targets.
The real native-structure/prototype-RNG combat source is exact-replayed
before creating each separately-seeded experimental Nature session.
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
from sts2_ai.training.h15_experimental_rollout import (
    RolloutLimits, run_policy_combat,
)
from probe_h15_native_candidates import is_ancestor, reenter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emulator-root", required=True, type=Path)
    ap.add_argument("--emulator-revision", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--train-corpus", required=True, type=Path)
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--max-player-actions", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=32)
    ap.add_argument("--max-nature", type=int, default=1024)
    ap.add_argument("--seed-prefix", default="h15-multiturn-policy-diagnostic-v1")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    limits = RolloutLimits(
        args.max_player_actions, args.max_turns, args.max_nature,
    )
    limits.validate()
    if args.limit < 1:
        ap.error("limit must be positive")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing research report")
    if not is_ancestor(
        args.emulator_root, args.source_revision, args.emulator_revision,
    ):
        raise ValueError("Rollout must use a descendant emulator revision")
    recipes = load_recipes(args.train_corpus)
    if any(r.emulator_revision != args.source_revision for r in recipes):
        raise ValueError("Corpus source emulator revision differs")

    teacher = H15CombatEvaluator.load(args.teacher, device=args.device)
    counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with JsonlEmulatorBackend(
        emulator_root=args.emulator_root,
        expected_emulator_revision=args.emulator_revision,
        build=args.build,
    ) as backend, args.output.open("x", encoding="utf-8") as writer:
        for i, recipe in enumerate(recipes[:args.limit]):
            source = reenter(backend, recipe)
            try:
                entropy = hashlib.sha256(
                    (args.seed_prefix + "\n" + recipe.exact_hash)
                    .encode("utf-8")
                ).digest()
                report = run_policy_combat(
                    backend, source,
                    independent_seed=entropy.hex(),
                    teacher=teacher, limits=limits,
                )
            finally:
                backend.release_many((source,))
            report.update({
                "index": i,
                "source_seed": recipe.seed,
                "source_exact_hash": recipe.exact_hash,
                "source_floor": recipe.floor,
                "source_tier": recipe.tier,
                "source_emulator_revision": args.source_revision,
                "experimental_emulator_revision": args.emulator_revision,
                "teacher_sha256": teacher.checkpoint_sha256,
                "teacher_codec": teacher.codec_schema,
            })
            counts[report["status"]] += 1
            writer.write(json.dumps(report, sort_keys=True) + "\n")
            writer.flush()
            print("[h15-multiturn-policy] " + json.dumps({
                "index": i, "floor": recipe.floor, "tier": recipe.tier,
                "status": report["status"],
                "outcome": report.get("outcome"),
                "player_actions": report["player_actions"],
                "turns_ended": report["turns_ended"],
                "nature_decisions": report["nature_decisions"],
                "rejection_count": sum(report["rejections"].values()),
                "reason": report.get("reason"),
            }, sort_keys=True), flush=True)
    print("[h15-multiturn-policy-summary] " + json.dumps({
        "counts": dict(counts),
        "fair_search_roots": 0,
        "probability_status": "UNVERIFIED",
        "teacher_sha256": teacher.checkpoint_sha256,
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
