#!/usr/bin/env python3
"""Synthetic HP/intent + legacy playable Nature JSONL server for worker tests.

Only a transport/state-machine imitation; no STS2 gameplay mechanics.
Seed names with `blocked-` deliberately hit an unsupported controlled RNG
site after the first player action. It allows deterministic coverage reports.
"""
from __future__ import annotations

import json
import sys

states: dict[str, dict] = {}
serial = 0


def publish(prefix, phase, choices=(), seed=""):
    global serial
    serial += 1
    handle = f"{prefix}{serial}"
    states[handle] = {
        "prefix": prefix, "phase": phase,
        "choices": list(choices), "seed": seed,
    }
    return frame(handle)


def frame(handle):
    s = states[handle]
    prefix, phase, choices = s["prefix"], s["phase"], s["choices"]
    fp = f"{prefix}:{phase}:{tuple(choices)}:{s['seed']}"
    if prefix == "n":
        at_choice = not choices
        pending = None
        if at_choice:
            pending = {
                "siteId": ("combat.initial_unique_hp" if phase == 0 else
                           "combat.committed_intent.index_draw"),
                "streamId": "run.niche" if phase == 0 else "run.monster_ai",
                "ownerId": "creature:1", "ordinal": 0,
                "operationKind": "bounded_integer",
                "sourceVersion": "v0.111.0", "orderedCandidates": [],
                "boundedCount": 2,
            }
        return {
            "natureHandle": handle, "fingerprint": fp,
            "kind": "initial_hp" if phase == 0 else "intents",
            "complete": pending is None, "pending": pending,
        }
    chance = phase in (0, 2, 3)
    pending = None
    if chance:
        pending = {
            "eventId": f"native-event:{fp}",
            "supportFingerprint": f"support:{phase}",
            "supportCount": 2,
            "siteId": ("controlled.fixture.opening_shuffle.swap_index" if phase == 0
                       else "combat.effect.random_enemy"),
            "streamId": "prototype.combat", "ownerId": "creature:1",
            "ordinal": phase, "operationKind": "bounded_integer",
            "sourceVersion": "v0.111.0", "orderedCandidates": [],
            "boundedCount": 2,
        }
    phase_name = {0: "opening_shuffle", 1: "player", 2: "random_target",
                  3: "random_target", 4: "player", 5: "terminal"}[phase]
    return {
        "schemaId": "sts2-v111-controlled-playable-fixture-v1",
        "controlledHandle": handle, "fingerprint": fp,
        "phase": phase_name, "terminal": phase == 5, "pending": pending,
    }


def serve(req):
    op = req["op"]
    if op == "hello":
        return {
            "wireSchemaId": "prototype-ai-jsonl-v0",
            "fairPolicyId": "prototype-fair-v0",
            "natureBoundarySchemaId": "sts2-v111-nature-isolated-boundary-v1",
            "natureBoundaryScope": "isolated-combat-entry-only",
            "natureBoundaryKinds": ["initial_hp", "intents"],
            "controlledChanceSchemaId": "sts2-v111-controlled-playable-fixture-v1",
            "controlledChanceScope": "isolated-playable-combat-stage-continuations-v1",
            "controlledChanceSupportedSites": [
                "controlled.fixture.opening_shuffle.swap_index",
                "combat.effect.random_enemy",
            ],
        }
    if op == "nature_begin":
        return publish("n", 0, seed=req["seed"])
    if op == "nature_advance":
        parent = states[req["nature_handle"]]
        if not parent["choices"]:
            raise ValueError("Incomplete HP")
        return publish("n", 1, seed=parent["seed"])
    if op == "nature_resolve":
        parent = states[req["nature_handle"]]
        if frame(req["nature_handle"])["fingerprint"] != req["fingerprint"]:
            raise ValueError("Stale Nature state")
        if not 0 <= req["selected_index"] < 2:
            raise ValueError("Illegal Nature index")
        return publish(
            "n", parent["phase"], [*parent["choices"], req["selected_index"]],
            seed=parent["seed"],
        )
    if op == "controlled_begin":
        parent = states[req["nature_handle"]]
        if not frame(req["nature_handle"])["complete"] or parent["phase"] != 1:
            raise ValueError("Incomplete intents")
        return publish("c", 0, seed=req["seed"])
    if op == "controlled_resolve":
        parent = states[req["controlled_handle"]]
        if frame(req["controlled_handle"])["fingerprint"] != req["fingerprint"]:
            raise ValueError("Stale playable pending state")
        if parent["phase"] not in (0, 2, 3):
            raise ValueError("Not a chance phase")
        if not 0 <= req["selected_index"] < 2:
            raise ValueError("Illegal index")
        output = publish(
            "c", parent["phase"] + 1,
            [*parent["choices"], req["selected_index"]], seed=parent["seed"],
        )
        output["parent"] = req["controlled_handle"]
        return output
    if op == "controlled_step":
        parent = states[req["controlled_handle"]]
        if parent["phase"] not in (1, 4) or req["action_id"] != "end_turn":
            raise ValueError("Illegal player action")
        if parent["seed"].startswith("blocked-"):
            raise ValueError(
                "UNSUPPORTED_CONTROLLED_CHANCE "
                "site=combat.unmigrated_test operation=random_index"
            )
        output = publish(
            "c", 2 if parent["phase"] == 1 else 5,
            [*parent["choices"], req["action_id"]], seed=parent["seed"],
        )
        output["parent"] = req["controlled_handle"]
        return output
    if op == "controlled_observe":
        if states[req["controlled_handle"]]["phase"] not in (1, 4):
            raise ValueError("Not a public player boundary")
        return {
            "policyId": "prototype-fair-v0", "schemaId": "prototype-ai-v0",
            "payloadJson": json.dumps({"player_hp": 32, "turn": 1}),
            "observationHash": "fair-observation",
            "actions": [{"actionId": "end_turn", "kind": "end_turn"}],
            "terminal": False,
        }
    if op == "controlled_release_many":
        return {"released": sum(states.pop(h, None) is not None
                                for h in req["controlled_handles"])}
    if op == "nature_release_many":
        return {"released": sum(states.pop(h, None) is not None
                                for h in req["nature_handles"])}
    if op == "close":
        return {}
    raise ValueError(f"Unknown operation: {op}")


for line in sys.stdin:
    if not line.strip():
        continue
    req = None
    try:
        req = json.loads(line)
        response = {"ok": True, "requestId": req.get("request_id"), **serve(req)}
    except Exception as exc:
        response = {
            "ok": False, "requestId": req.get("request_id") if req else None,
            "errorType": type(exc).__name__, "error": str(exc),
        }
    print(json.dumps(response, separators=(",", ":")), flush=True)
    if req and req.get("op") == "close":
        break
