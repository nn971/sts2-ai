#!/usr/bin/env python3
"""Exercise real-run H15 end-turn/Nature continuations on exact-replayed entries.

Experimental coverage probe only; NO fair PUCT samples or training targets.
Failures from unowned stochastic sites are recorded, not silently replaced by
prototype or native RNG sampling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import load_recipes
from probe_h15_native_candidates import is_ancestor, reenter

SCHEMA = "sts2-h15-experimental-controlled-endturn-coverage-v1"
NATURE_KINDS = frozenset((
    "enemy_action", "enemy_intent", "draw_card", "random_target",
))


def exercise(
    backend: JsonlEmulatorBackend, recipe: Any,
    *, study_seed: str, max_nature: int,
) -> dict[str, Any]:
    source = reenter(backend, recipe)
    session_handle: str | None = None
    kinds: list[str] = []
    try:
        independent_seed = hashlib.sha256(
            (study_seed + "\n" + recipe.seed + "\n" + recipe.exact_hash)
            .encode("utf-8")
        ).hexdigest()
        status = backend.h15_experimental_begin(
            source, independent_seed=independent_seed,
        )
        session_handle = status["h15Handle"]
        if status["boundaryKind"] != "player":
            raise ValueError("H15 entry does not expose a player decision")
        end_turn_actions = [
            action["actionId"] for action in status.get("legalActions") or []
            if action.get("kind") == "end_turn"
        ]
        if len(end_turn_actions) != 1:
            raise ValueError("Expected one legal first-turn end_turn action")
        status = backend.h15_experimental_end_turn(
            session_handle, status["version"], end_turn_actions[0],
        )
        while status["boundaryKind"] in NATURE_KINDS:
            if len(kinds) >= max_nature:
                return {
                    "status": "censored", "reason": "maximum_nature_boundaries",
                    "nature_boundaries": kinds,
                    "final_boundary": status["boundaryKind"],
                }
            kinds.append(status["boundaryKind"])
            law = status["chanceLaw"]
            if law["probabilityStatus"] != "UNVERIFIED":
                raise ValueError("H15 returned a falsely verified chance prior")
            outcome = law["outcomes"][0]["outcomeId"]
            status = backend.h15_experimental_choose(
                session_handle, status["version"], law, outcome,
            )
        return {
            "status": ("next_player_or_exit"
                       if status["boundaryKind"] in ("player", "terminal")
                       else "unsupported_boundary"),
            "nature_boundaries": kinds,
            "final_boundary": status["boundaryKind"],
            "combat_exit": status["combatExit"],
        }
    except Exception as exc:
        return {
            "status": "rejected",
            "nature_boundaries": kinds,
            "reason": type(exc).__name__ + ": " + str(exc),
        }
    finally:
        if session_handle is not None:
            backend.h15_experimental_release_many((session_handle,))
        backend.release_many((source,))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emulator-root", type=Path, required=True)
    ap.add_argument("--emulator-revision", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--max-nature", type=int, default=40)
    ap.add_argument("--study-seed", default="h15-controlled-turn-v1")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.limit < 1 or args.max_nature < 1:
        ap.error("limit and max-nature must be positive")
    if not is_ancestor(
        args.emulator_root, args.source_revision, args.emulator_revision,
    ):
        raise ValueError(
            "H15 controlled-turn probe requires a descendant source revision"
        )
    recipes = load_recipes(args.train_corpus)
    if any(recipe.emulator_revision != args.source_revision
           for recipe in recipes):
        raise ValueError("Incompatible recorded emulator revisions")
    if args.output.exists():
        raise FileExistsError("H15 experimental coverage report already exists")
    counts: dict[str, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with JsonlEmulatorBackend(
        emulator_root=args.emulator_root,
        expected_emulator_revision=args.emulator_revision,
        build=args.build,
    ) as backend, args.output.open("x", encoding="utf-8") as stream:
        for recipe in recipes[:args.limit]:
            report = exercise(
                backend, recipe, study_seed=args.study_seed,
                max_nature=args.max_nature,
            )
            record = {
                "schema": SCHEMA, "source_seed": recipe.seed,
                "entry_hash": recipe.exact_hash,
                "tier": recipe.tier, "floor": recipe.floor,
                "fair_search_roots": 0, **report,
            }
            counts[report["status"]] = counts.get(report["status"], 0) + 1
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print("[h15-controlled-turn] " +
                  json.dumps(record, sort_keys=True), flush=True)
    print("[h15-controlled-turn-summary] " + json.dumps({
        "counts": counts, "fair_search_roots": 0,
        "source_revision": args.source_revision,
        "emulator_revision": args.emulator_revision,
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
