#!/usr/bin/env python3
"""Audit H21 against genuine v1 emulator observations and legal actions.

This is a finite deterministic test probe, never a training session.
No private RunState, RNG stream or hidden deck order is read by the policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from sts2_ai.training.h21_action_alias_audit import audit_h21_action_aliases
from sts2_ai.training.h21_field_coverage import audit_public_observation
from sts2_ai.training.h21_macro_offer_codec import encode_h21_macro_decision

FAIR_POLICY_ID = "prototype-fair-v0"


class PublicJsonl:
    def __init__(self, dll: Path) -> None:
        self.process = subprocess.Popen(
            ["dotnet", str(dll), "prototype-ai-jsonl"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.request_index = 0

    def call(self, op: str, **kwargs: Any) -> dict[str, Any]:
        self.request_index += 1
        request = {"op": op, "request_id": str(self.request_index), **kwargs}
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Public JSONL process pipes closed")
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        result = self.process.stdout.readline()
        if not result:
            code = self.process.poll()
            raise RuntimeError(f"Emulator JSONL ended before reply (exit={code})")
        data = json.loads(result)
        if data.get("requestId") != str(self.request_index) or not data.get("ok"):
            raise RuntimeError("Emulator public RPC failed: " + repr(data))
        return data

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        if self.process.returncode:
            lines = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(
                "Emulator JSONL process failed: " + lines[-2000:])


# PrototypeRoomType source enum order. These are public map categories.
_ROOM_NAMES = ("Combat", "Elite", "Event", "Shop", "Rest",
               "Boss", "Unknown", "Treasure")
_ROUTE_PREFERENCES = (
    ("Shop", "Elite", "Event", "Unknown", "Treasure", "Rest", "Combat", "Boss"),
    ("Elite", "Shop", "Unknown", "Event", "Rest", "Combat", "Treasure", "Boss"),
    ("Unknown", "Event", "Shop", "Elite", "Treasure", "Rest", "Combat", "Boss"),
    ("Rest", "Treasure", "Event", "Shop", "Elite", "Unknown", "Combat", "Boss"),
    ("Event", "Shop", "Unknown", "Elite", "Rest", "Treasure", "Combat", "Boss"),
)


def _public_room_name(room_type: Any) -> str:
    if type(room_type) is int and 0 <= room_type < len(_ROOM_NAMES):
        return _ROOM_NAMES[room_type]
    return str(room_type)


def choose(
    actions: list[dict[str, Any]], *, public_state: dict[str, Any] | None = None,
    seed: str = "", step: int = 0, route_variant: int = 0,
) -> dict[str, Any]:
    """Deterministic coverage policy using only the published observation."""
    priorities = {
        kind: n for n, kind in enumerate((
            "start_run", "take_reward_gold", "take_reward_relic",
            "take_reward_extra_gold", "take_reward_card", "take_reward_card_group",
            "take_reward_potion", "leave_reward", "choose_map_node",
            "event_choice", "continue_event", "leave_shop", "rest_heal",
            "rest_upgrade", "rest_train", "use_potion", "play_card", "end_turn",
        ))
    }
    nodes = {}
    if public_state is not None:
        nodes = {
            node["node_id"]: _public_room_name(node["room_type"])
            for node in public_state.get("map", [])
        }
    preferences = _ROUTE_PREFERENCES[route_variant % len(_ROUTE_PREFERENCES)]

    def order(action: dict[str, Any]) -> tuple[int, int, str]:
        kind = action["kind"]
        menu_priority = priorities.get(kind, 99)
        route_priority = 0
        if kind == "choose_map_node":
            payload = json.loads(action["payloadJson"])
            node_id = payload.get("NodeId", payload.get("node_id"))
            room_type = nodes.get(node_id, "")
            route_priority = (
                preferences.index(room_type)
                if room_type in preferences else len(preferences)
            )
        # Stable tie-breakers vary the route/event selections across seeds
        # without using hidden emulator streams or RNG.
        data = f"{seed}:{step}:{action['kind']}:{action['payloadJson']}"
        tie = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return menu_priority, route_priority, tie

    return min(actions, key=order)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--seed", action="append", dest="seeds")
    parser.add_argument("--max-decisions", type=int, default=55)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    seeds = args.seeds or ["h21-public-contract-s1", "h21-public-contract-s2"]
    if args.max_decisions < 1:
        parser.error("--max-decisions must be positive")
    if not args.emulator_dll.is_file():
        parser.error("emulator DLL not found")

    start = time.perf_counter()
    phases: Counter[str] = Counter()
    action_kinds: Counter[str] = Counter()
    gaps: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    decisions = 0
    equivalent_aliases = 0
    canonical_choices = 0
    public = PublicJsonl(args.emulator_dll)
    try:
        hello = public.call("hello")
        if hello.get("aiSchemaId") != EMULATOR_PUBLIC_SCHEMA_V1:
            raise ValueError("Unexpected public observation frame schema")
        for seed in seeds:
            begin = public.call(
                "reset_native_overgrowth", seed=seed, ascension=0,
                compact_card_selection=True,
            )
            handle = begin["stateHandle"]
            for step in range(args.max_decisions):
                view = public.call(
                    "observe", state_handle=handle, policy_id=FAIR_POLICY_ID)
                if view.get("schemaId") != EMULATOR_PUBLIC_SCHEMA_V1:
                    raise ValueError("Observation schema changed within run")
                observation = json.loads(view["payloadJson"])
                menu = public.call("legal_actions", state_handle=handle)["actions"]
                if not menu:
                    break
                decisions += 1
                phases[str(observation.get("phase"))] += 1
                action_kinds.update(item["kind"] for item in menu)
                report = audit_public_observation(observation)
                gaps.update(report.gaps)
                actions = [LegalAction(
                    item["actionId"], item["kind"], item["payloadJson"],
                ) for item in menu]
                try:
                    encoding = encode_h21_macro_decision(
                        observation, actions, public_schema_id=view["schemaId"])
                    alias_audit = audit_h21_action_aliases(
                        observation, actions, encoding)
                    equivalent_aliases += alias_audit.equivalent_aliases
                    canonical_choices += alias_audit.canonical_choices
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({
                        "seed": seed, "step": step,
                        "phase": str(observation.get("phase")),
                        "action_kinds": sorted(set(item["kind"] for item in menu)),
                        "error": type(exc).__name__ + ": " + str(exc),
                    })
                selected = choose(menu)
                result = public.call(
                    "step", state_handle=handle, action_id=selected["actionId"])
                handle = result["child"]
                if result.get("terminal"):
                    break
    finally:
        public.close()
    print(json.dumps({
        "schema": EMULATOR_PUBLIC_SCHEMA_V1,
        "decisions": decisions, "seeds": seeds,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "canonical_choices": canonical_choices,
        "equivalent_action_aliases": equivalent_aliases,
        "phase_counts": dict(phases),
        "legal_action_kinds": dict(sorted(action_kinds.items())),
        "known_gap_occurrences": dict(sorted(gaps.items())),
        "encode_failures": errors[:30], "total_encode_failures": len(errors),
    }, sort_keys=True, indent=2), flush=True)
    if decisions < len(seeds):
        raise AssertionError("Insufficient live public decisions")
    if args.strict and errors:
        raise AssertionError(
            f"{len(errors)} genuine v1 public frames failed H21 grounding")
    return 0


if __name__ == "__main__":
    sys.exit(main())
