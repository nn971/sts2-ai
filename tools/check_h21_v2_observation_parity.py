#!/usr/bin/env python3
"""Lockstep public observation and action parity for opt-in enemy power v2.

Compares independent same-seed generated-map states through the public JSONL
bridge. The old public enemy status/power split is normalized into canonical
power names before comparisons. No hidden emulator state, future RNG, or
training actor is accessed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from probe_h21_live_public import FAIR_POLICY_ID, PublicJsonl, choose
from sts2_ai.training.h19_resource_codec import (
    EMULATOR_PUBLIC_SCHEMA_V1,
    EMULATOR_PUBLIC_SCHEMA_V2,
)

_EFFECT_IDS = {
    "proto.status.poison": "proto.power.poison",
    "proto.status.weak": "proto.power.weak",
    "proto.status.vulnerable": "proto.power.vulnerable",
}
_CANONICAL = frozenset(_EFFECT_IDS.values())


def normalize_legacy(observation: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the legacy enemy effect split, preserving every other field."""
    state = copy.deepcopy(observation)
    combat = state.get("combat")
    if not isinstance(combat, dict):
        return state
    for enemy in combat.get("enemies", []):
        if not isinstance(enemy, dict):
            raise ValueError("Malformed public enemy in legacy observation")
        statuses = enemy.get("statuses")
        powers = enemy.get("powers")
        if not isinstance(statuses, dict) or not isinstance(powers, list):
            raise ValueError("Legacy enemy statuses/powers are malformed")
        merged: dict[str, int] = {}
        other = []
        for power in powers:
            pid, stacks = power["power_id"], power["stacks"]
            if pid in _CANONICAL:
                merged[pid] = merged.get(pid, 0) + stacks
            else:
                other.append(power)
        for status, stacks in statuses.items():
            if status not in _EFFECT_IDS:
                raise ValueError("Unrecognized legacy enemy status: " + status)
            pid = _EFFECT_IDS[status]
            merged[pid] = merged.get(pid, 0) + stacks
        other.extend(
            {"power_id": pid, "stacks": stacks}
            for pid, stacks in sorted(merged.items()) if stacks > 0
        )
        enemy["statuses"] = {}
        enemy["powers"] = other
    return state


def first_difference(left: Any, right: Any, path: str = "observation") -> str | None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return path + ".keys: " + repr(sorted(left)) + " vs " + repr(sorted(right))
        for key in sorted(left):
            difference = first_difference(left[key], right[key], path + "." + key)
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path + ".length: " + str(len(left)) + " vs " + str(len(right))
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(a, b, path + f"[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return path + ": " + repr(left)[:200] + " vs " + repr(right)[:200]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--seed", action="append", dest="seeds")
    parser.add_argument("--max-decisions", type=int, default=100)
    args = parser.parse_args()
    if args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    seeds = args.seeds or ["p1-v2-parity-a"]
    checked = 0
    canonical_frames = 0
    highest_floor = 0
    errors: list[dict[str, Any]] = []
    public = PublicJsonl(args.emulator_dll)
    try:
        hello = public.call("hello")
        if (hello.get("aiSchemaId") != EMULATOR_PUBLIC_SCHEMA_V1
                or hello.get("experimentalAiPowerSchemaId")
                != EMULATOR_PUBLIC_SCHEMA_V2):
            raise RuntimeError("Emulator does not advertise v1/v2 public frames")
        for seed_index, seed in enumerate(seeds):
            first = public.call("reset_native_overgrowth",
                                seed=seed, ascension=0,
                                compact_card_selection=True)["stateHandle"]
            second = public.call("reset_native_overgrowth",
                                 seed=seed, ascension=0,
                                 compact_card_selection=True,
                                 canonical_enemy_effect_powers=True)["stateHandle"]
            for step in range(args.max_decisions):
                old = public.call("observe", state_handle=first,
                                  policy_id=FAIR_POLICY_ID)
                new = public.call("observe", state_handle=second,
                                  policy_id=FAIR_POLICY_ID)
                legacy_observation = json.loads(old["payloadJson"])
                canonical_observation = json.loads(new["payloadJson"])
                is_combat = isinstance(canonical_observation.get("combat"), dict)
                expected_schema = (EMULATOR_PUBLIC_SCHEMA_V2 if is_combat
                                   else EMULATOR_PUBLIC_SCHEMA_V1)
                if (old["schemaId"] != EMULATOR_PUBLIC_SCHEMA_V1
                        or new["schemaId"] != expected_schema):
                    errors.append({"seed": seed, "step": step,
                                   "error": "Incorrect public schema IDs"})
                    break
                canonical_frames += int(is_combat)
                floor = canonical_observation.get("floor")
                if type(floor) is int:
                    highest_floor = max(highest_floor, floor)
                difference = first_difference(
                    normalize_legacy(legacy_observation), canonical_observation)
                if difference is not None:
                    errors.append({"seed": seed, "step": step, "error": difference})
                    break
                old_actions = public.call(
                    "legal_actions", state_handle=first)["actions"]
                new_actions = public.call(
                    "legal_actions", state_handle=second)["actions"]
                if old_actions != new_actions:
                    errors.append({"seed": seed, "step": step,
                                   "error": "Different public legal actions"})
                    break
                if not old_actions:
                    break
                checked += 1
                action = choose(
                    old_actions, public_state=legacy_observation,
                    seed=seed, step=step, route_variant=seed_index)
                first_step = public.call(
                    "step", state_handle=first, action_id=action["actionId"])
                second_step = public.call(
                    "step", state_handle=second, action_id=action["actionId"])
                first, second = first_step["child"], second_step["child"]
                if first_step["terminal"] != second_step["terminal"]:
                    errors.append({"seed": seed, "step": step,
                                   "error": "Different terminal decision"})
                    break
                if first_step["terminal"]:
                    break
    finally:
        public.close()

    print(json.dumps({
        "seeds": seeds, "checked_decisions": checked,
        "canonical_v2_frames": canonical_frames,
        "highest_floor": highest_floor, "errors": errors[:15],
        "total_errors": len(errors),
    }, indent=2, sort_keys=True), flush=True)
    if errors or checked < len(seeds) or canonical_frames == 0:
        raise AssertionError(
            "Canonical/legacy public observation parity failed: "
            + repr(errors[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
