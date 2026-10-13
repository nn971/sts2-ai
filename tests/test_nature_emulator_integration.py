"""AI adapter against the actual versioned JSONL *shape*.

A deterministic fake protocol supplies fast CI coverage. Optional game-local
end-to-end tests use the emulator's own tools/nature_jsonl_smoke.py; no native
engine or submodule is modified by this suite.
"""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator.controlled_chance import (
    ControlledChanceUnavailable,
    IsolatedNatureBridge,
)
from sts2_ai.nature.policies import FirstLegalNature, ScriptedNature, UniformNature


class FakeNatureWire:
    def __init__(self) -> None:
        self.states: dict[str, dict] = {}
        self.next_handle = 0
        self.hello_scope = "isolated-combat-entry-only"

    def _publish(self, state):
        self.next_handle += 1
        handle = f"n{self.next_handle}"
        state = dict(state)
        self.states[handle] = state
        kind, index = state["kind"], state["index"]
        count = 2 if kind == "initial_hp" else 1
        pending = None
        if index < count:
            pending = {
                "siteId": "combat.initial_unique_hp" if kind == "initial_hp"
                          else "combat.committed_intent.index_draw",
                "streamId": "run.niche" if kind == "initial_hp"
                            else "run.monster_ai",
                "ownerId": f"creature:{index + 1}",
                "ordinal": index,
                "sourceVersion": "v0.111.0",
                "operationKind": "choose_index" if kind == "initial_hp"
                                 else "bounded_integer",
                "orderedCandidates": [9, 12, 15] if kind == "initial_hp" else [],
                "boundedCount": None if kind == "initial_hp" else 3,
            }
        return {
            "kind": kind, "natureHandle": handle,
            "fingerprint": f"{kind}:{index}:{state['selections']}",
            "pending": pending, "complete": pending is None,
        }

    def call(self, operation, **kw):
        if operation == "hello":
            return {
                "wireSchemaId": "prototype-ai-jsonl-v0",
                "natureBoundarySchemaId": "sts2-v111-nature-isolated-boundary-v1",
                "natureBoundaryScope": self.hello_scope,
                "natureBoundaryKinds": ["initial_hp", "intents"],
            }
        if operation == "nature_begin":
            return self._publish({
                "kind": kw["kind"], "index": 0, "selections": [],
            })
        if operation == "nature_resolve":
            state = self.states[kw["nature_handle"]]
            index = state["index"]
            expected = f"{state['kind']}:{index}:{state['selections']}"
            if kw["fingerprint"] != expected:
                raise ValueError("stale fingerprint")
            return self._publish({
                **state, "index": index + 1,
                "selections": [*state["selections"], kw["selected_index"]],
            })
        if operation == "nature_advance":
            state = self.states[kw["nature_handle"]]
            assert state["kind"] == "initial_hp" and state["index"] == 2
            return self._publish({
                "kind": "intents", "index": 0,
                "selections": state["selections"],
            })
        if operation == "nature_fork":
            return self._publish(self.states[kw["nature_handle"]])
        if operation == "nature_save":
            return {
                "kind": self.states[kw["nature_handle"]]["kind"],
                "snapshot": json.dumps(self.states[kw["nature_handle"]]),
            }
        if operation == "nature_load":
            state = json.loads(kw["snapshot"])
            assert state["kind"] == kw["kind"]
            return self._publish(state)
        if operation == "nature_release_many":
            return {"released": sum(
                self.states.pop(h, None) is not None for h in kw["nature_handles"]
            )}
        raise AssertionError(operation)


PARAMS = {
    "seed": "simulated-entry",
    "kind": "initial_hp", "act": 2,
    "encounter_id": "proto.native.hive.encounter.mytes_normal",
    "native_entry": "MytesNormal", "total_floor": 21,
}


def test_isolated_nature_adapter_scripted_replay_and_immutable_fork():
    wire = FakeNatureWire()
    bridge = IsolatedNatureBridge(wire, emulator_revision="pinned-git-sha")
    hp = bridge.begin(**PARAMS)
    fork = bridge.fork(hp)
    saved = bridge.save(hp)
    loaded = bridge.load(saved, kind="initial_hp")
    assert loaded.fingerprint == hp.fingerprint == fork.fingerprint
    child = bridge.resolve(hp, 1)
    assert child.fingerprint != hp.fingerprint
    assert bridge.release(fork, loaded) == 2
    with pytest.raises(ValueError, match="Illegal"):
        bridge.resolve(hp, 4)

    result = bridge.run_fixture(UniformNature(seed=29), **PARAMS)
    assert result.final.kind == "intents" and result.final.complete
    assert len(result.transcript.entries) == 3
    choices = result.transcript.replay_script()
    scripted = ScriptedNature.from_selections(choices)
    replay = bridge.run_fixture(scripted, **PARAMS)
    scripted.assert_complete()
    assert replay.final.fingerprint == result.final.fingerprint
    assert replay.transcript.replay_script() == choices


def test_bridge_rejects_unsupported_capability_and_gameplay_scope():
    wire = FakeNatureWire()
    wire.hello_scope = "all-playable-combat"
    with pytest.raises(ControlledChanceUnavailable, match="scope changed"):
        IsolatedNatureBridge(wire, emulator_revision="sha")
    wire.hello_scope = "isolated-combat-entry-only"
    bridge = IsolatedNatureBridge(wire, emulator_revision="sha")
    with pytest.raises(ControlledChanceUnavailable, match="Unsupported isolated"):
        bridge.begin(**{**PARAMS, "act": 1})
    with pytest.raises(ControlledChanceUnavailable, match="Only completed HP"):
        bridge.advance(bridge.begin(**PARAMS))
    with pytest.raises(ValueError, match="max_choices"):
        bridge.run_fixture(FirstLegalNature(), **PARAMS, max_choices=0)
