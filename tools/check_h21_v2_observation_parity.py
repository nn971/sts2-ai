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
    parser.add_argument(
        "--snapshot-every", type=int, default=0,
        help="Periodically save/load and replay the same next public action",
    )
    args = parser.parse_args()
    if args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    if args.snapshot_every < 0:
        parser.error("--snapshot-every must be nonnegative")
    seeds = args.seeds or ["p1-v2-parity-a"]
    checked = 0
    canonical_frames = 0
    snapshot_roundtrips = 0
    snapshot_replayed_actions = 0
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
                replay_handles: list[tuple[str, str]] = []
                if args.snapshot_every and step % args.snapshot_every == 0:
                    for handle, frame, actions in (
                        (first, old, old_actions),
                        (second, new, new_actions),
                    ):
                        saved = public.call("save_snapshot", state_handle=handle)
                        loaded = public.call(
                            "load_snapshot", snapshot=saved["snapshot"])
                        restored_handle = loaded["stateHandle"]
                        roundtrip = public.call(
                            "save_snapshot", state_handle=restored_handle)
                        if saved["snapshot"] != roundtrip["snapshot"]:
                            raise AssertionError(
                                "RunSnapshot save/load/save bytes differ")
                        replay_frame = public.call(
                            "observe", state_handle=restored_handle,
                            policy_id=FAIR_POLICY_ID)
                        replay_actions = public.call(
                            "legal_actions", state_handle=restored_handle)["actions"]
                        if (replay_frame["schemaId"] != frame["schemaId"]
                                or replay_frame["payloadJson"] != frame["payloadJson"]
                                or replay_actions != actions):
                            raise AssertionError(
                                "RunSnapshot roundtrip changed the public frame/actions")
                        replay_handles.append((handle, restored_handle))
                        snapshot_roundtrips += 1
                first_step = public.call(
                    "step", state_handle=first, action_id=action["actionId"])
                second_step = public.call(
                    "step", state_handle=second, action_id=action["actionId"])
                first, second = first_step["child"], second_step["child"]
                for original, restored in replay_handles:
                    replay_step = public.call(
                        "step", state_handle=restored,
                        action_id=action["actionId"])
                    expected = (first_step if original == replay_handles[0][0]
                                else second_step)
                    if replay_step["terminal"] != expected["terminal"]:
                        raise AssertionError(
                            "RunSnapshot replay changed terminal decision")
                    replay_frame = public.call(
                        "observe", state_handle=replay_step["child"],
                        policy_id=FAIR_POLICY_ID)
                    expected_frame = public.call(
                        "observe", state_handle=expected["child"],
                        policy_id=FAIR_POLICY_ID)
                    if (replay_frame["schemaId"] != expected_frame["schemaId"]
                            or replay_frame["payloadJson"] != expected_frame["payloadJson"]):
                        raise AssertionError(
                            "RunSnapshot replay changed next public observation")
                    snapshot_replayed_actions += 1
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
        "snapshot_roundtrips": snapshot_roundtrips,
        "snapshot_replayed_actions": snapshot_replayed_actions,
        "highest_floor": highest_floor, "errors": errors[:15],
        "total_errors": len(errors),
    }, indent=2, sort_keys=True), flush=True)
    if (args.snapshot_every and snapshot_replayed_actions == 0):
        raise AssertionError("No RunSnapshot replay actions verified")
    if errors or checked < len(seeds) or canonical_frames == 0:
        raise AssertionError(
            "Canonical/legacy public observation parity failed: "
            + repr(errors[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
