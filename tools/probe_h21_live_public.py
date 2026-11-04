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
from sts2_ai.training.h19_resource_codec import (
    EMULATOR_PUBLIC_SCHEMA_V1,
    EMULATOR_PUBLIC_SCHEMA_V2,
)
from sts2_ai.training.h21_action_alias_audit import audit_h21_action_aliases
from sts2_ai.training.h21_field_coverage import audit_public_observation
from sts2_ai.training.h21_macro_offer_codec import encode_h21_macro_decision
from sts2_ai.training.h21_stable_identity import stable_id

if __package__:
    from .p2_h21_public_corpus import (
        make_public_corpus,
        make_public_record,
        write_public_corpus,
    )
else:
    from p2_h21_public_corpus import (
        make_public_corpus,
        make_public_record,
        write_public_corpus,
    )

FAIR_POLICY_ID = "prototype-fair-v0"
_ENEMY_EFFECT_POWER_IDS = frozenset({
    "proto.power.poison", "proto.power.weak", "proto.power.vulnerable",
})
_DEBUFF_COVERAGE_CARDS = frozenset({
    "proto.silent.neutralize", "proto.silent.deadly_poison",
    "proto.silent.bouncing_flask", "proto.silent.crippling_cloud",
})


def observed_canonical_enemy_effects(observation: dict[str, Any]) -> set[str]:
    """Identify effects positively present in an actual public v2 frame."""
    combat = observation.get("combat")
    if not isinstance(combat, dict):
        return set()
    return {
        power["power_id"]
        for enemy in combat.get("enemies", [])
        for power in enemy.get("powers", [])
        if isinstance(power, dict)
        and power.get("power_id") in _ENEMY_EFFECT_POWER_IDS
        and type(power.get("stacks")) is int and power["stacks"] > 0
    }


def preferred_debuff_card_action(
    actions: list[dict[str, Any]], observation: dict[str, Any],
) -> dict[str, Any] | None:
    """Favor legal debuff cards using only the visible hand and action payloads."""
    combat = observation.get("combat")
    if not isinstance(combat, dict):
        return None
    candidates = {
        card.get("instance_id")
        for card in combat.get("hand", [])
        if isinstance(card, dict)
        and card.get("card_id") in _DEBUFF_COVERAGE_CARDS
    }
    matching = []
    for action in actions:
        if action.get("kind") != "play_card":
            continue
        payload = json.loads(action["payloadJson"])
        if payload.get("CardInstanceId", payload.get("card_instance_id")) in candidates:
            matching.append(action)
    return min(matching, key=lambda action: (
        action["kind"], action["payloadJson"])) if matching else None


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
    continuation_coverage: bool = False,
    prefer_enemy_debuff_cards: bool = False,
) -> dict[str, Any]:
    """Deterministic coverage policy using only the published observation."""
    if prefer_enemy_debuff_cards and public_state is not None:
        preferred = preferred_debuff_card_action(actions, public_state)
        if preferred is not None:
            return preferred
    priorities = {
        kind: n for n, kind in enumerate((
            "start_run", "take_reward_gold", "take_reward_relic",
            "take_reward_extra_gold", "take_reward_card", "take_reward_card_group",
            "take_reward_potion", "leave_reward", "choose_map_node",
            "event_choice", "continue_event", "leave_shop", "rest_heal",
            "rest_upgrade", "rest_train", "use_potion", "play_card", "end_turn",
        ))
    }
    if continuation_coverage:
        # Exercise actual reward/shop/event/rest continuations, rather than
        # simply inspecting their menus before choosing leave_shop.
        # Keep the original smoke/throughput policy untouched.
        preferred = [
            "start_run", "choose_event_deck_card",
            "choose_shop_relic_deck_card", "choose_relic_deck_card",
            "finish_relic_deck_choice", "replace_shop_potion",
            "buy_relic", "buy_potion", "buy_card", "remove_card",
            "replace_reward_extra_potion", "take_reward_extra_potion",
            "replace_reward_potion", "take_reward_potion",
            "take_event_potion", "replace_event_potion", "skip_event_potion",
            "take_reward_extra_relic", "take_reward_relic",
            "take_reward_bundle", "take_reward_card_group",
            "take_reward_card", "take_reward_extra_gold",
            "take_reward_gold", "skip_reward_extra_relic",
            "skip_reward_extra_potion", "skip_reward_card_group",
            "skip_reward_card", "reroll_reward_card",
            "trade_future_potion", "throw_foul_potion_at_merchant",
            "rest_cook", "rest_upgrade", "rest_dig",
            "rest_lift", "rest_clone", "rest_heal",
            "rest_kindle", "rest_hatch", "rest_train",
            "event_choice", "continue_event",
            "leave_reward", "leave_shop", "choose_map_node",
            "select_cards", "use_potion", "play_card", "end_turn",
        ]
        if route_variant % 2:
            # Also exercise optional early completion, while even variants
            # pick a candidate before finishing their deck selection.
            preferred.remove("finish_relic_deck_choice")
            preferred.insert(1, "finish_relic_deck_choice")
        priorities = {kind: n for n, kind in enumerate(preferred)}
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


def missing_executed_action_groups(
    executed: Counter[str], required_groups: list[str],
) -> list[str]:
    """Find required OR-groups with no actions actually executed in the run."""
    missing = []
    for group in required_groups:
        alternatives = [kind.strip() for kind in group.split(",")]
        if not all(alternatives) or len(set(alternatives)) != len(alternatives):
            raise ValueError(f"Invalid required action group: {group!r}")
        if not any(executed.get(kind, 0) for kind in alternatives):
            missing.append(",".join(alternatives))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulator-dll", type=Path, required=True)
    parser.add_argument("--seed", action="append", dest="seeds")
    parser.add_argument("--max-decisions", type=int, default=55)
    parser.add_argument(
        "--capture-public-decisions", type=Path,
        help="P2/H21: save validated public frames and legal menus only, "
             "for source-pinned offline codec/collation benchmarks",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--canonical-enemy-powers", action="store_true",
        help="Opt into the experimental v2 canonical enemy-power public frames",
    )
    parser.add_argument(
        "--require-executed-any", action="append", default=[], metavar="KIND[,KIND...]",
        help=("Require at least one actually executed action from each "
              "comma-separated group"),
    )
    parser.add_argument(
        "--prefer-enemy-debuff-cards", action="store_true",
        help="Prefer legal visible Neutralize/Poison cards in v2 acceptance",
    )
    parser.add_argument(
        "--require-observed-enemy-power", action="append", default=[],
        choices=sorted(_ENEMY_EFFECT_POWER_IDS),
        help="Require a positive instance of this canonical enemy power in a live v2 frame",
    )
    parser.add_argument(
        "--continuation-coverage", action="store_true",
        help="Prefer purchasing/choosing exceptional player continuations",
    )
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
    if (args.capture_public_decisions is not None
            and args.capture_public_decisions.exists()):
        parser.error("Refusing to overwrite H21 public benchmark corpus")
    if args.continuation_coverage and not args.diverse_route:
        parser.error("--continuation-coverage requires --diverse-route")
    if args.require_observed_enemy_power and not args.canonical_enemy_powers:
        parser.error("--require-observed-enemy-power requires --canonical-enemy-powers")

    start = time.perf_counter()
    phases: Counter[str] = Counter()
    action_kinds: Counter[str] = Counter()
    executed_action_kinds: Counter[str] = Counter()
    observed_enemy_effect_frames: Counter[str] = Counter()
    gaps: Counter[str] = Counter()
    partials: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    step_errors: list[dict[str, Any]] = []
    decisions = 0
    public_corpus_records: list[dict[str, Any]] = []
    equivalent_aliases = 0
    canonical_choices = 0
    encode_seconds: list[float] = []
    codec_seconds = 0.0
    alias_seconds = 0.0
    max_objects = 0
    max_relations = 0
    max_floor = 0
    chosen_map_room_types: Counter[str] = Counter()
    seen_schema_ids: Counter[str] = Counter()
    stable_id.cache_clear()
    profiler = cProfile.Profile() if args.profile_codec_out is not None else None
    public = PublicJsonl(args.emulator_dll)
    try:
        hello = public.call("hello")
        if hello.get("aiSchemaId") != EMULATOR_PUBLIC_SCHEMA_V1:
            raise ValueError("Unexpected public observation frame schema")
        if (args.canonical_enemy_powers
                and hello.get("experimentalAiPowerSchemaId")
                != EMULATOR_PUBLIC_SCHEMA_V2):
            raise ValueError("Emulator does not advertise canonical v2 power frames")
        for seed_index, seed in enumerate(seeds):
            reset_args = {"seed": seed, "ascension": 0,
                          "compact_card_selection": True}
            if args.canonical_enemy_powers:
                reset_args["canonical_enemy_effect_powers"] = True
            begin = public.call("reset_native_overgrowth", **reset_args)
            handle = begin["stateHandle"]
            for step in range(args.max_decisions):
                view = public.call(
                    "observe", state_handle=handle, policy_id=FAIR_POLICY_ID)
                observation = json.loads(view["payloadJson"])
                required_schema = (
                    EMULATOR_PUBLIC_SCHEMA_V2 if args.canonical_enemy_powers
                    and isinstance(observation.get("combat"), dict)
                    else EMULATOR_PUBLIC_SCHEMA_V1
                )
                if view.get("schemaId") != required_schema:
                    raise ValueError("Unexpected public frame schema at decision")
                seen_schema_ids[required_schema] += 1
                if required_schema == EMULATOR_PUBLIC_SCHEMA_V2:
                    observed_enemy_effect_frames.update(
                        observed_canonical_enemy_effects(observation))
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
                    if args.capture_public_decisions is not None:
                        public_corpus_records.append(make_public_record(
                            seed, step, view["schemaId"], observation, menu,
                        ))
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
                        continuation_coverage=args.continuation_coverage,
                        prefer_enemy_debuff_cards=args.prefer_enemy_debuff_cards,
                    )
                elif args.prefer_enemy_debuff_cards:
                    selected = choose(
                        menu, public_state=observation, seed=seed, step=step,
                        prefer_enemy_debuff_cards=True,
                    )
                else:
                    selected = choose(menu)
                executed_action_kinds.update([selected["kind"]])
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
    missing_required = missing_executed_action_groups(
        executed_action_kinds, args.require_executed_any,
    )
    print(json.dumps({
        "schema": (EMULATOR_PUBLIC_SCHEMA_V2
                   if args.canonical_enemy_powers else EMULATOR_PUBLIC_SCHEMA_V1),
        "seen_schema_ids": dict(sorted(seen_schema_ids.items())),
        "canonical_enemy_powers": args.canonical_enemy_powers,
        "observed_enemy_power_frames": dict(sorted(observed_enemy_effect_frames.items())),
        "required_observed_enemy_powers": args.require_observed_enemy_power,
        "decisions": decisions, "seeds": seeds,
        "p2_public_corpus_records": len(public_corpus_records),
        "p2_public_corpus_output": (
            str(args.capture_public_decisions)
            if args.capture_public_decisions is not None else None
        ),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "diverse_route": args.diverse_route,
        "continuation_coverage": args.continuation_coverage,
        "executed_action_kinds": dict(sorted(executed_action_kinds.items())),
        "required_executed_action_groups": args.require_executed_any,
        "missing_required_executed_groups": missing_required,
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
    if missing_required:
        raise AssertionError(
            "Unexercised required public continuations: " + repr(missing_required))
    missing_effects = [
        power for power in args.require_observed_enemy_power
        if observed_enemy_effect_frames[power] == 0
    ]
    if missing_effects:
        raise AssertionError(
            "No positive live canonical enemy effect instances: " + repr(missing_effects))
    if (args.canonical_enemy_powers
            and seen_schema_ids[EMULATOR_PUBLIC_SCHEMA_V2] == 0):
        raise AssertionError("Canonical probe did not visit any v2 combat frames")
    if args.strict and (errors or step_errors or gaps or partials):
        raise AssertionError(
            f"Live public probe: {len(errors)} H21 frame failures, "
            f"{len(step_errors)} emulator step failures, "
            f"{sum(gaps.values())} field gaps and "
            f"{sum(partials.values())} incomplete fields")
    if args.capture_public_decisions is not None:
        if len(public_corpus_records) != decisions or errors or step_errors:
            raise AssertionError(
                "P2/H21 corpus requires fully successful encoded public frames "
                "and emulator continuations; no partial capture"
            )
        write_public_corpus(
            args.capture_public_decisions,
            make_public_corpus(
                emulator_dll=args.emulator_dll,
                records=public_corpus_records,
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
