#!/usr/bin/env python3
"""Real A10 entry public-history conditioning under an UNVERIFIED candidate prior.

This is a corpus-level integration/coverage diagnostic, not fair PUCT.
The exact source combat is reentered from CombatSnapshotRecipe and used
only by the offline public-history observer. Candidate Nature outcomes
are sampled exclusively by the emulator ReferenceNature provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.h15_experimental_posterior import (
    ExperimentalConditioningUnavailable,
    H15ExperimentalReferencePosterior,
    capture_source_public_history,
)
from sts2_ai.evaluation.combat_snapshots import load_recipes
from probe_h15_native_candidates import is_ancestor, reenter

SCHEMA = "sts2-h15-real-entry-experimental-reference-conditioning-v1"


def probe_entry(
    backend: JsonlEmulatorBackend, source: str, *,
    action_mode: str, random_seed: int, max_candidates: int,
    max_nature: int, samples: int,
) -> dict:
    if action_mode not in ("root_only", "end_turn", "first_card"):
        raise ValueError("Unsupported public-history action mode")
    action_ids: tuple[str, ...] = ()
    if action_mode != "root_only":
        menu = tuple(backend.legal_actions(source))
        kind = "end_turn" if action_mode == "end_turn" else "play_card"
        eligible = [a.action_id for a in menu if a.kind == kind]
        if not eligible:
            return {
                "status": "source_action_unavailable",
                "reason": "no source-legal " + kind,
                "public_frames": 0,
            }
        action_ids = (eligible[0],)
    try:
        history = capture_source_public_history(
            backend, source, action_ids,
        )
    except ExperimentalConditioningUnavailable as exc:
        return {
            "status": "source_prefix_incomplete",
            "reason": str(exc), "public_frames": 0,
        }
    posterior = H15ExperimentalReferencePosterior(
        backend, source, max_candidates=max_candidates,
        max_nature_events=max_nature,
    )
    accepted: tuple[str, ...] = ()
    try:
        accepted = posterior.sample_conditioned(
            history, search_rng=random.Random(random_seed), count=samples,
        )
        status, reason = "conditioned", None
    except ExperimentalConditioningUnavailable as exc:
        status, reason = "unavailable", str(exc)
    finally:
        if accepted:
            backend.h15_experimental_release_many(accepted)
    stats = posterior.last_stats
    return {
        "status": status, "reason": reason,
        "public_frames": len(history),
        "public_observation_hashes": [
            s.observation.observation_hash for s in history
        ],
        "action_ids": list(action_ids),
        "attempted": stats.attempted,
        "accepted": stats.accepted if status == "conditioned" else 0,
        "mismatched": stats.mismatched,
        "nature_events": stats.sampled_nature_events,
        "candidate_prior": posterior.model_id,
        "probability_status": posterior.probability_status,
        "fair_search_roots": 0, "training_targets": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emulator-root", type=Path, required=True)
    ap.add_argument("--emulator-revision", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument(
        "--action-mode", choices=("root_only", "end_turn", "first_card"),
        default="end_turn",
    )
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=128)
    ap.add_argument("--max-nature", type=int, default=1200)
    ap.add_argument("--seed-prefix", default="h15-experimental-condition-a10-v1")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if min(args.limit, args.samples, args.max_candidates, args.max_nature) < 1:
        ap.error("All budgets must be positive")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite existing H15 diagnostic")
    if not is_ancestor(
        args.emulator_root, args.source_revision, args.emulator_revision,
    ):
        raise ValueError("Experimental emulator must descend from source revision")
    recipes = load_recipes(args.train_corpus)
    if any(r.emulator_revision != args.source_revision for r in recipes):
        raise ValueError("Mixed emulator source provenance")

    counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with JsonlEmulatorBackend(
        emulator_root=args.emulator_root,
        expected_emulator_revision=args.emulator_revision,
        build=args.build,
    ) as backend, args.output.open("x", encoding="utf-8") as out:
        for index, recipe in enumerate(recipes[:args.limit]):
            source = reenter(backend, recipe)
            try:
                seed_bytes = hashlib.sha256(
                    (args.seed_prefix + "\n" + recipe.exact_hash)
                    .encode("utf-8")
                ).digest()
                result = probe_entry(
                    backend, source, action_mode=args.action_mode,
                    random_seed=int.from_bytes(seed_bytes[:8], "big"),
                    max_candidates=args.max_candidates,
                    max_nature=args.max_nature, samples=args.samples,
                )
            finally:
                backend.release_many((source,))
            row = {
                "schema": SCHEMA, "index": index,
                "source_seed": recipe.seed,
                "source_exact_hash": recipe.exact_hash,
                "source_floor": recipe.floor, "source_tier": recipe.tier,
                "source_emulator_revision": args.source_revision,
                "experimental_emulator_revision": args.emulator_revision,
                "action_mode": args.action_mode,
                "fair_search_roots": 0, "training_targets": 0,
                "probability_status": "UNVERIFIED",
                **result,
            }
            counts[result["status"]] += 1
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            print("[h15-condition] " + json.dumps({
                "index": index, "floor": recipe.floor, "tier": recipe.tier,
                "status": result["status"],
                "attempted": result.get("attempted"),
                "accepted": result.get("accepted"),
                "reason": result.get("reason"),
            }, sort_keys=True), flush=True)
    print("[h15-condition-summary] " + json.dumps({
        "counts": dict(counts), "output": str(args.output),
        "fair_search_roots": 0, "training_targets": 0,
        "probability_status": "UNVERIFIED",
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
