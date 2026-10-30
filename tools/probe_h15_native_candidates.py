#!/usr/bin/env python3
"""Probe native-structure combat entries against the independent-root builder.

This diagnostic has no fair PUCT continuation capability. Its returned
candidate never leaves the emulator; no search targets can be generated.
The original native-overgrowth corpus has native map/room structure but
prototype combat RNG. Distinguish that from opt-in native-combat RNG.

The corpus was captured on a pinned emulator ancestor. Cross-revision replay
is allowed ONLY with an explicitly named source revision, and every entry
must match its recorded exact state hash before it can be probed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run
from sts2_ai.evaluation.combat_snapshots import (
    POLICY, CombatSnapshotRecipe, _verify, load_recipes,
)

PROBE_SCHEMA = "sts2-h15-native-combat-candidate-probe-v1"
CANDIDATE_MODELS = {
    "prototype-combat-rng":
        "h15-native-structure-prototype-rng-candidate-unverified-v1",
    "native-combat-rng":
        "h15-native-combat-entry-independent-candidate-unverified-v1",
}


def is_ancestor(root: Path, ancestor: str, head: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor",
         ancestor, head], capture_output=True, check=False,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            "Cannot verify recorded emulator revision ancestry: " +
            proc.stderr.decode(errors="replace")
        )
    return proc.returncode == 0


def reenter(
    backend: JsonlEmulatorBackend, recipe: CombatSnapshotRecipe,
) -> str:
    """Replay a pinned native run and verify the *complete* exact root hash."""
    current = reset_training_run(
        backend, recipe.seed, NATIVE_OVERGROWTH, recipe.ascension,
    )
    try:
        for action_id in recipe.action_ids:
            legal = backend.legal_actions(current)
            selected = next(
                (action for action in legal if action.action_id == action_id),
                None,
            )
            if selected is None:
                raise ValueError(
                    "Source action missing in descendant replay: " + action_id
                )
            child = backend.step(current, selected).child
            backend.release_many((current,))
            current = child
        _verify(backend, current, recipe)
        return current
    except BaseException:
        backend.release_many((current,))
        raise


def probe(
    backend: JsonlEmulatorBackend, recipe: CombatSnapshotRecipe,
    *, search_seed: str,
) -> dict[str, Any]:
    handle = reenter(backend, recipe)
    try:
        original = backend.observe(handle, POLICY)
        result = backend.probe_h15_native_combat_candidate(
            handle, independent_seed=search_seed,
        )
        if (result["candidate_model"] != CANDIDATE_MODELS.get(result["provider_class"])
                or result["observation_hash"] != original.observation_hash):
            raise ValueError("Native candidate altered the public entry observation")
        return {
            "source_seed": recipe.seed,
            "entry_hash": recipe.exact_hash,
            "family": list(recipe.enemy_ids),
            "tier": recipe.tier,
            "floor": recipe.floor,
            **result,
        }
    finally:
        backend.release_many((handle,))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emulator-root", type=Path, required=True)
    ap.add_argument("--emulator-revision", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--study-seed", default="h15-native-entry-probe-v1")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.limit < 1 or not args.study_seed:
        ap.error("Invalid native candidate probe settings")
    if not (len(args.source_revision) == len(args.emulator_revision) == 40):
        ap.error("Require full source and descendant emulator revision identities")
    if not is_ancestor(
        args.emulator_root, args.source_revision, args.emulator_revision,
    ):
        raise ValueError(
            "Probe emulator must descend from the exact corpus source revision"
        )
    recipes = load_recipes(args.train_corpus)
    if any(recipe.emulator_revision != args.source_revision
           for recipe in recipes):
        raise ValueError("Corpus contains an unapproved source emulator revision")
    if args.output.exists():
        raise FileExistsError("H15 diagnostic report already exists")
    attempted = passed = 0
    provider_counts = {key: 0 for key in CANDIDATE_MODELS}
    with JsonlEmulatorBackend(
        emulator_root=args.emulator_root,
        expected_emulator_revision=args.emulator_revision,
        build=args.build,
    ) as backend:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            for recipe in recipes[:args.limit]:
                attempted += 1
                search_seed = hashlib.sha256((
                    args.study_seed + "\n" + recipe.seed + "\n" +
                    recipe.exact_hash
                ).encode()).hexdigest()
                try:
                    outcome = probe(backend, recipe, search_seed=search_seed)
                    passed += 1
                    provider_counts[outcome["provider_class"]] += 1
                    record: dict[str, Any] = {
                        "schema": PROBE_SCHEMA,
                        "status": "candidate_constructed",
                        **outcome,
                    }
                except Exception as exc:
                    record = {
                        "schema": PROBE_SCHEMA,
                        "status": "rejected",
                        "source_seed": recipe.seed,
                        "entry_hash": recipe.exact_hash,
                        "reason": type(exc).__name__ + ": " + str(exc),
                    }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                print("[h15-native-root-probe] " +
                      json.dumps(record, sort_keys=True), flush=True)

    print("[h15-native-root-summary] " + json.dumps({
        "attempted": attempted, "candidate_constructed": passed,
        "rejected": attempted - passed, "fair_search_roots": 0,
        "candidate_provider_classes": provider_counts,
        "source_revision": args.source_revision,
        "emulator_revision": args.emulator_revision,
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
