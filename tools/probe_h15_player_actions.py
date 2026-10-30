#!/usr/bin/env python3
"""Exercise real-run audited player actions followed by controlled end-turn.

Samples from the exact replayed Act 1 native-structure corpus. The report
is diagnostic only: independent Nature outcomes are UNVERIFIED and must not
supply fair PUCT or outcome-prediction training targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import load_recipes
from probe_h15_native_candidates import is_ancestor, reenter

SCHEMA = "sts2-h15-experimental-real-player-action-coverage-v1"
NATURE_KINDS = frozenset((
    "enemy_action", "enemy_intent", "draw_card", "random_target",
))


def exercise(
    backend: JsonlEmulatorBackend,
    recipe: Any,
    *, study_seed: str, maximum_actions: int = 12,
    maximum_nature: int = 48,
) -> dict[str, Any]:
    source = reenter(backend, recipe)
    handle: str | None = None
    attempted: list[dict[str, str]] = []
    events: list[str] = []
    try:
        independent_seed = hashlib.sha256(
            (study_seed + "\n" + recipe.seed + "\n" + recipe.exact_hash)
            .encode()
        ).hexdigest()
        frame = backend.h15_experimental_begin(
            source, independent_seed=independent_seed,
        )
        handle = frame["h15Handle"]
        if frame["boundaryKind"] != "player":
            raise ValueError("H15 live root is not a player boundary")
        legal = frame.get("legalActions") or []
        player_actions = [
            action for action in legal
            if action.get("kind") in ("play_card", "use_potion")
            and isinstance(action.get("actionId"), str)
        ]
        selected: dict[str, str] | None = None
        for action in player_actions[:maximum_actions]:
            kind = action["kind"]
            action_id = action["actionId"]
            try:
                next_frame = backend.h15_experimental_player_action(
                    handle, frame["version"], action_id,
                )
            except Exception as exc:
                attempted.append({
                    "kind": kind,
                    "rejection": type(exc).__name__ + ": " + str(exc),
                })
                # The server must retain the old state after a rejected
                # action. Validate the version before trying another.
                unchanged = backend.h15_experimental_status(handle)
                if (unchanged["version"] != frame["version"] or
                        unchanged["observationHash"] != frame["observationHash"]):
                    raise ValueError(
                        "H15 rejected action mutated its live session"
                    ) from exc
                continue
            selected = {"kind": kind, "action_id": action_id}
            frame = next_frame
            break
        if selected is None:
            return {
                "status": "no_supported_player_action",
                "candidate_actions": len(player_actions),
                "attempted": attempted,
                "final_boundary": frame["boundaryKind"],
            }

        if frame["boundaryKind"] == "player":
            ending = [
                action["actionId"]
                for action in frame.get("legalActions") or []
                if action.get("kind") == "end_turn"
            ]
            if len(ending) != 1:
                raise ValueError("Player action has no unique legal end_turn")
            frame = backend.h15_experimental_end_turn(
                handle, frame["version"], ending[0],
            )
        while frame["boundaryKind"] in NATURE_KINDS:
            if len(events) >= maximum_nature:
                return {
                    "status": "censored", "selected": selected,
                    "attempted": attempted,
                    "nature_boundaries": events,
                    "final_boundary": frame["boundaryKind"],
                }
            events.append(frame["boundaryKind"])
            law = frame["chanceLaw"]
            frame = backend.h15_experimental_choose(
                handle, frame["version"], law,
                law["outcomes"][0]["outcomeId"],
            )
        return {
            "status": ("next_player_or_exit"
                       if frame["boundaryKind"] in ("player", "combat_exit")
                       else "unsupported_boundary"),
            "selected": selected, "attempted": attempted,
            "candidate_actions": len(player_actions),
            "nature_boundaries": events,
            "final_boundary": frame["boundaryKind"],
            "combat_outcome": frame.get("combatOutcome"),
        }
    except Exception as exc:
        return {
            "status": "rejected", "attempted": attempted,
            "nature_boundaries": events,
            "reason": type(exc).__name__ + ": " + str(exc),
        }
    finally:
        if handle is not None:
            backend.h15_experimental_release_many((handle,))
        backend.release_many((source,))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emulator-root", type=Path, required=True)
    ap.add_argument("--emulator-revision", required=True)
    ap.add_argument("--source-revision", required=True)
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--max-actions", type=int, default=12)
    ap.add_argument("--max-nature", type=int, default=48)
    ap.add_argument("--study-seed", default="h15-real-player-actions-v1")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if min(args.limit, args.max_actions, args.max_nature) < 1:
        ap.error("All positive limits are required")
    if not is_ancestor(
        args.emulator_root, args.source_revision, args.emulator_revision,
    ):
        raise ValueError("H15 player-action probe requires source ancestry")
    recipes = load_recipes(args.train_corpus)
    if any(r.emulator_revision != args.source_revision for r in recipes):
        raise ValueError("Corpus contains a different recorded emulator revision")
    if args.output.exists():
        raise FileExistsError("Player-action coverage report already exists")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with JsonlEmulatorBackend(
        emulator_root=args.emulator_root,
        expected_emulator_revision=args.emulator_revision,
        build=args.build,
    ) as backend, args.output.open("x", encoding="utf-8") as output:
        for recipe in recipes[:args.limit]:
            report = exercise(
                backend, recipe, study_seed=args.study_seed,
                maximum_actions=args.max_actions,
                maximum_nature=args.max_nature,
            )
            record = {
                "schema": SCHEMA,
                "seed": recipe.seed, "entry_hash": recipe.exact_hash,
                "tier": recipe.tier, "floor": recipe.floor,
                "fair_search_roots": 0, **report,
            }
            counts[report["status"]] += 1
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            print("[h15-player-action] " +
                  json.dumps(record, sort_keys=True), flush=True)
    print("[h15-player-action-summary] " + json.dumps({
        "counts": dict(counts), "fair_search_roots": 0,
        "output": str(args.output),
        "emulator_revision": args.emulator_revision,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
