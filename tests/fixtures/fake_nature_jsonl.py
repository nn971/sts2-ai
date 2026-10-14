#!/usr/bin/env python3
"""Explicitly synthetic v111 Nature JSONL fixture service for process tests.

Matches the isolated HP/intent *wire shape*, not the C# combat mechanics.
"""
from __future__ import annotations

import hashlib
import json
import sys

states: dict[str, dict] = {}
counter = 0


def digest(state):
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def publish(state):
    global counter
    counter += 1
    handle = f"n{counter}"
    state = json.loads(json.dumps(state))
    states[handle] = state
    kind, index = state["kind"], state["index"]
    limit = (2 if kind == "initial_hp" else
             0 if state["params"]["native_entry"] == "DeterministicNoDraw" else 1)
    pending = None
    if index < limit:
        pending = {
            "siteId": "combat.initial_unique_hp"
            if kind == "initial_hp" else "combat.committed_intent.index_draw",
            "streamId": "run.niche" if kind == "initial_hp" else "run.monster_ai",
            "ownerId": f"creature:{index + 1}",
            "ordinal": index,
            "sourceVersion": "v0.111.0",
            "operationKind": "choose_index" if kind == "initial_hp" else "bounded_integer",
            "orderedCandidates": [11, 12, 13] if kind == "initial_hp" else [],
            "boundedCount": None if kind == "initial_hp" else 3,
        }
    return {
        "kind": kind, "natureHandle": handle, "fingerprint": digest(state),
        "pending": pending, "complete": pending is None,
    }


def serve(request):
    op = request["op"]
    if op == "hello":
        return {
            "wireSchemaId": "prototype-ai-jsonl-v0",
            "natureBoundarySchemaId": "sts2-v111-nature-isolated-boundary-v1",
            "natureBoundaryScope": "isolated-combat-entry-only",
            "natureBoundaryKinds": ["initial_hp", "intents"],
        }
    if op == "nature_begin":
        if request["act"] not in (2, 3):
            raise ValueError("Unsupported act")
        return publish({
            "kind": request["kind"], "index": 0, "choices": [],
            "params": {k: request[k] for k in (
                "seed", "kind", "act", "encounter_id", "native_entry", "total_floor",
                "ascension", "map_column", "map_row",
            )},
        })
    if op == "nature_resolve":
        state = states[request["nature_handle"]]
        if digest(state) != request["fingerprint"]:
            raise ValueError("Stale fingerprint")
        count = (2 if state["kind"] == "initial_hp" else
                 0 if state["params"]["native_entry"] == "DeterministicNoDraw" else 1)
        if state["index"] >= count:
            raise ValueError("Completed fixture")
        if type(request["selected_index"]) is not int or not 0 <= request["selected_index"] < 3:
            raise ValueError("Illegal selected index")
        return publish({
            **state,
            "index": state["index"] + 1,
            "choices": [*state["choices"], request["selected_index"]],
        })
    if op == "nature_advance":
        state = states[request["nature_handle"]]
        if state["kind"] != "initial_hp" or state["index"] != 2:
            raise ValueError("Incomplete HP")
        return publish({**state, "kind": "intents", "index": 0})
    if op == "nature_fork":
        return publish(states[request["nature_handle"]])
    if op == "nature_save":
        state = states[request["nature_handle"]]
        return {"kind": state["kind"], "snapshot": json.dumps(state, sort_keys=True)}
    if op == "nature_load":
        state = json.loads(request["snapshot"])
        if state["kind"] != request["kind"]:
            raise ValueError("Snapshot type mismatch")
        return publish(state)
    if op == "nature_release_many":
        handles = request["nature_handles"]
        if not isinstance(handles, list) or any(h not in states for h in handles):
            raise ValueError("Unknown Nature handle")
        for handle in handles:
            del states[handle]
        return {"released": len(handles)}
    if op == "close":
        return {}
    raise ValueError(f"Unsupported operation: {op}")


for line in sys.stdin:
    if not line.strip():
        continue
    try:
        req = json.loads(line)
        answer = {
            "requestId": req.get("request_id"), "ok": True, **serve(req),
        }
    except Exception as exc:
        answer = {
            "requestId": req.get("request_id") if "req" in locals() else None,
            "ok": False, "errorType": type(exc).__name__, "error": str(exc),
        }
    print(json.dumps(answer, separators=(",", ":")), flush=True)
    if "req" in locals() and req.get("op") == "close":
        break
