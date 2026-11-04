#!/usr/bin/env python3
"""Audit H21 against genuine v1 emulator observations and legal actions.

This is a finite deterministic test probe, never a training session.
No private RunState, RNG stream or hidden deck order is read by the policy.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import pstats
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
from sts2_ai.training.h21_stable_identity import stable_id

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
    if public_state is None:
        # Keep the short 2-seed smoke policy exactly reproducible.
        return min(actions, key=lambda a: (
            priorities.get(a["kind"], 99), a["kind"], a["payloadJson"]))
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
    parser.add_argument(
        "--profile-codec-out", type=Path,
        help="Write cProfile cumulative-time report for the H21 encoder only",
    )
    parser.add_argument(
        "--diverse-route", action="store_true",
        help="Use seed-varied public-map room priorities instead of fixed smoke ordering",
    )
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
    partials: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    step_errors: list[dict[str, Any]] = []
    decisions = 0
    equivalent_aliases = 0
    canonical_choices = 0
    encode_seconds: list[float] = []
    codec_seconds = 0.0
    alias_seconds = 0.0
    max_objects = 0
    max_relations = 0
    max_floor = 0
    chosen_map_room_types: Counter[str] = Counter()
    stable_id.cache_clear()
    profiler = cProfile.Profile() if args.profile_codec_out is not None else None
    public = PublicJsonl(args.emulator_dll)
    try:
        hello = public.call("hello")
        if hello.get("aiSchemaId") != EMULATOR_PUBLIC_SCHEMA_V1:
            raise ValueError("Unexpected public observation frame schema")
        for seed_index, seed in enumerate(seeds):
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
                current_floor = observation.get("floor")
                if type(current_floor) is int:
                    max_floor = max(max_floor, current_floor)
                menu = public.call("legal_actions", state_handle=handle)["actions"]
                if not menu:
                    break
                decisions += 1
                phases[str(observation.get("phase"))] += 1
                action_kinds.update(item["kind"] for item in menu)
                actions = [LegalAction(
                    item["actionId"], item["kind"], item["payloadJson"],
                ) for item in menu]
                encode_start = time.perf_counter()
                try:
                    report = audit_public_observation(observation)
                    gaps.update(report.gaps)
                    partials.update(report.partial)
                    codec_start = time.perf_counter()
                    if profiler is not None:
                        profiler.enable()
                    try:
                        encoding = encode_h21_macro_decision(
                            observation, actions, public_schema_id=view["schemaId"])
                    finally:
                        if profiler is not None:
                            profiler.disable()
                    codec_end = time.perf_counter()
                    codec_seconds += codec_end - codec_start
                    alias_audit = audit_h21_action_aliases(
                        observation, actions, encoding)
                    alias_seconds += time.perf_counter() - codec_end
                    equivalent_aliases += alias_audit.equivalent_aliases
                    canonical_choices += alias_audit.canonical_choices
                    max_objects = max(max_objects, len(encoding.objects))
                    max_relations = max(max_relations, len(encoding.relations))
                except (ValueError, KeyError, TypeError) as exc:
                    errors.append({
                        "seed": seed, "step": step,
                        "phase": str(observation.get("phase")),
                        "action_kinds": sorted(set(item["kind"] for item in menu)),
                        "error": type(exc).__name__ + ": " + str(exc),
                    })
                finally:
                    encode_seconds.append(time.perf_counter() - encode_start)
                if args.diverse_route:
                    selected = choose(
                        menu, public_state=observation, seed=seed,
                        step=step, route_variant=seed_index,
                    )
                else:
                    selected = choose(menu)
                if selected["kind"] == "choose_map_node":
                    selected_payload = json.loads(selected["payloadJson"])
                    selected_node = selected_payload.get(
                        "NodeId", selected_payload.get("node_id"))
                    for node in observation.get("map", []):
                        if node.get("node_id") == selected_node:
                            chosen_map_room_types.update(
                                [_public_room_name(node.get("room_type"))])
                            break
                try:
                    result = public.call(
                        "step", state_handle=handle, action_id=selected["actionId"])
                except RuntimeError as exc:
                    step_errors.append({
                        "seed": seed, "step": step,
                        "action_kind": selected["kind"],
                        "error": str(exc)[:1000],
                    })
                    break
                handle = result["child"]
                if result.get("terminal"):
                    break
    finally:
        public.close()
    if profiler is not None:
        from io import StringIO

        output = StringIO()
        stats = pstats.Stats(profiler, stream=output)
        stats.strip_dirs().sort_stats("cumulative").print_stats(65)
        args.profile_codec_out.parent.mkdir(parents=True, exist_ok=True)
        args.profile_codec_out.write_text(output.getvalue(), encoding="utf-8")
    measured = sorted(encode_seconds)
    p95_index = max(0, (95 * len(measured) + 99) // 100 - 1)
    measured_total = sum(measured)
    cache = stable_id.cache_info()
    print(json.dumps({
        "schema": EMULATOR_PUBLIC_SCHEMA_V1,
        "decisions": decisions, "seeds": seeds,
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "diverse_route": args.diverse_route,
        "max_floor": max_floor,
        "chosen_map_room_types": dict(sorted(chosen_map_room_types.items())),
        "encoding_only_seconds": round(measured_total, 6),
        "codec_only_seconds": round(codec_seconds, 6),
        "alias_check_seconds": round(alias_seconds, 6),
        "stable_hash_cache_hits": cache.hits,
        "stable_hash_cache_misses": cache.misses,
        "stable_hash_cache_size": cache.currsize,
        "encoding_only_decisions_per_second": (
            round(len(measured) / measured_total, 2) if measured_total else 0),
        "encoding_p95_ms": (
            round(measured[p95_index] * 1000, 3) if measured else 0),
        "max_objects": max_objects,
        "max_relations": max_relations,
        "canonical_choices": canonical_choices,
        "equivalent_action_aliases": equivalent_aliases,
        "phase_counts": dict(phases),
        "legal_action_kinds": dict(sorted(action_kinds.items())),
        "known_gap_occurrences": dict(sorted(gaps.items())),
        "known_partial_occurrences": dict(sorted(partials.items())),
        "encode_failures": errors[:30], "total_encode_failures": len(errors),
        "step_failures": step_errors[:30],
        "total_step_failures": len(step_errors),
    }, sort_keys=True, indent=2), flush=True)
    if decisions < len(seeds):
        raise AssertionError("Insufficient live public decisions")
    if args.strict and (errors or step_errors or gaps or partials):
        raise AssertionError(
            f"Live v1 probe: {len(errors)} H21 frame failures, "
            f"{len(step_errors)} emulator step failures, "
            f"{sum(gaps.values())} field gaps and "
            f"{sum(partials.values())} incomplete fields")
    return 0


if __name__ == "__main__":
    sys.exit(main())
