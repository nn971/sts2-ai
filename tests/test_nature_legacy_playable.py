"""Regression tests against the real #143 legacy controlled_* JSONL shape.

The wire is deterministic and synthetic; live emulator qualification is a
separate integration gate on the exact emulator revision.
"""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from sts2_ai.emulator.controlled_chance import ControlledChanceUnavailable
from sts2_ai.emulator.legacy_playable_chance import LegacyPlayableBridge
from sts2_ai.nature.playable_trajectory import (
    collect_legacy_pilot,
    replay_legacy_pilot,
)
from sts2_ai.nature.policies import FirstLegalNature


class FakePlayableWire:
    def __init__(self):
        self._counter = 0
        self.handles = {}
        self.bad_hello = False
        self.mutate_outcome = False
        self.unsupported = False
        self.released = []

    def _make(self, stage, chosen=(), transcript=None):
        self._counter += 1
        handle = f"h{self._counter}"
        # Fingerprints deliberately independent of generated handle IDs.
        fingerprint = f"state-{stage}-{'-'.join(map(str, chosen))}"
        state = {
            "stage": stage, "chosen": tuple(chosen),
            "fingerprint": fingerprint, "transcript": list(transcript or ()),
        }
        self.handles[handle] = state
        return handle, state

    @staticmethod
    def _pending(state):
        stage = state["stage"]
        if stage not in (0, 1, 3):
            return None
        return {
            "eventId": f"event-{stage}",
            "siteId": f"combat.fixture.{stage}",
            "streamId": "run.monster_ai",
            "ownerId": "creature:1",
            "ordinal": stage, "operationKind": "choose_index",
            "orderedCandidates": [1, 2], "boundedCount": None,
            "sourceVersion": "v0.111.0",
            "supportFingerprint": f"support-{stage}",
            "probabilityStatus": "UNVERIFIED",
            "referenceLawId": None,
        }

    def _frame(self, handle):
        s = self.handles[handle]
        return {
            "schemaId": "sts2-v111-controlled-playable-fixture-v1",
            "controlledHandle": handle, "fingerprint": s["fingerprint"],
            "phase": "opening_shuffle" if s["stage"] < 2 else (
                "next_turn_intent" if s["stage"] == 3 else "player"),
            "terminal": s["stage"] == 4,
            "pending": self._pending(s), "transcript": deepcopy(s["transcript"]),
        }

    def call(self, operation, **payload):
        if operation == "hello":
            return {
                "wireSchemaId": "prototype-ai-jsonl-v0",
                "fairPolicyId": "prototype-fair-v0",
                "controlledChanceSchemaId": (
                    "wrong" if self.bad_hello else
                    "sts2-v111-controlled-playable-fixture-v1"
                ),
                "controlledChanceScope":
                    "isolated-playable-combat-stage-continuations-v1",
                "controlledChanceSupportedSites": [
                    "controlled.fixture.opening_shuffle.swap_index",
                    "combat.next_turn_intent.index_draw",
                ],
            }
        if operation == "controlled_begin":
            assert payload == {"nature_handle": "completed-intents", "seed": "seed42"}
            h, _ = self._make(0)
            return self._frame(h)
        if operation == "controlled_release_many":
            self.released.extend(payload["controlled_handles"])
            for h in payload["controlled_handles"]:
                self.handles.pop(h, None)
            return {"released": len(payload["controlled_handles"])}
        if operation == "controlled_load":
            obj = json.loads(payload["snapshot"])
            h, _ = self._make(
                obj["stage"], obj["chosen"], obj["transcript"],
            )
            return self._frame(h)
        if operation == "controlled_save":
            s = self.handles[payload["controlled_handle"]]
            return {"schemaId": "sts2-v111-controlled-playable-fixture-v1",
                    "snapshot": json.dumps(s)}
        if operation == "controlled_fork":
            parent = payload["controlled_handle"]
            s = self.handles[parent]
            h, _ = self._make(s["stage"], s["chosen"], s["transcript"])
            return {"parent": parent, "controlledHandle": h,
                    "fingerprint": s["fingerprint"]}
        if operation == "controlled_trace":
            s = self.handles[payload["controlled_handle"]]
            return {
                "schemaId": "sts2-v111-nature-event-metadata-v1",
                "fingerprint": s["fingerprint"],
                "transcript": deepcopy(s["transcript"]),
            }
        if operation == "controlled_observe":
            s = self.handles[payload["controlled_handle"]]
            assert s["stage"] == 2
            assert payload["policy_id"] == "prototype-fair-v0"
            return {
                "policyId": "prototype-fair-v0", "terminal": False,
                "payloadJson": json.dumps({"hp": 12}),
                "observationHash": "fair-hash",
                "actions": [{"actionId": "action-1", "kind": "end_turn",
                             "payloadJson": "{}"}],
            }
        if operation == "controlled_legal_actions":
            return {"actions": [{"actionId": "action-1"}]}
        if operation == "controlled_resolve":
            parent = self.handles[payload["controlled_handle"]]
            assert parent["fingerprint"] == payload["fingerprint"]
            stage = parent["stage"]
            assert stage in (0, 1, 3)
            chosen = (*parent["chosen"], payload["selected_index"])
            transcript = parent["transcript"] + [{
                "eventId": f"event-{stage}",
                "selectedIndex": (
                    1 - payload["selected_index"] if self.mutate_outcome
                    else payload["selected_index"]
                ),
            }]
            h, _ = self._make(stage + 1, chosen, transcript)
            return {"parent": payload["controlled_handle"], **self._frame(h)}
        if operation == "controlled_step":
            if self.unsupported:
                raise ControlledChanceUnavailable(
                    "UNSUPPORTED_CONTROLLED_CHANCE: combat.card_draw")
            parent = self.handles[payload["controlled_handle"]]
            assert parent["stage"] == 2
            assert payload["action_id"] == "action-1"
            h, _ = self._make(3, parent["chosen"], parent["transcript"])
            return {"parent": payload["controlled_handle"], **self._frame(h)}
        raise AssertionError(f"Unexpected wire operation {operation}")


def make():
    wire = FakePlayableWire()
    return wire, LegacyPlayableBridge(wire, emulator_revision="sha-123")


def test_real_legacy_capabilities_and_fairness_gating():
    wire, bridge = make()
    frame = bridge.begin(completed_intents_handle="completed-intents", seed="seed42")
    assert frame.pending is not None
    assert frame.pending.probability_status == "UNVERIFIED"
    with pytest.raises(ValueError, match="fair decision"):
        bridge.observe(frame)
    with pytest.raises(ValueError, match="requires resolution"):
        bridge.step(frame, "action-1")
    with pytest.raises(ValueError, match="Illegal"):
        bridge.resolve(frame, 7)
    child = bridge.resolve(frame, 0)
    child = bridge.resolve(child, 1)
    fair = bridge.observe(child)
    assert fair.observation == {"hp": 12}
    with pytest.raises(ValueError, match="illegal action"):
        bridge.step(child, "forged", decision=fair)
    assert len(wire.handles) == 3
    bridge.release(frame, child)
    assert len(wire.released) == 2


def test_joint_player_chance_pilot_and_exact_replay():
    wire, bridge = make()
    pilot = collect_legacy_pilot(
        bridge, completed_intents_handle="completed-intents",
        seed="seed42", nature_policy=FirstLegalNature(),
        player_choose=lambda _obs, actions: actions[0],
    )
    assert pilot.status == "TERMINAL"
    assert [d.kind for d in pilot.decisions] == [
        "nature", "nature", "player", "nature",
    ]
    assert pilot.nature_policy_id
    assert not wire.handles
    assert replay_legacy_pilot(
        bridge, completed_intents_handle="completed-intents", pilot=pilot,
    ) == pilot
    assert not wire.handles


def test_censoring_and_unsupported_mechanics_are_explicit():
    wire, bridge = make()
    pilot = collect_legacy_pilot(
        bridge, completed_intents_handle="completed-intents",
        seed="seed42", nature_policy=FirstLegalNature(),
        player_choose=lambda _obs, actions: actions[0], max_decisions=2,
    )
    assert pilot.status == "CENSORED_BOUNDARY_LIMIT"
    replay_legacy_pilot(
        bridge, completed_intents_handle="completed-intents", pilot=pilot,
    )
    wire.unsupported = True
    with pytest.raises(ControlledChanceUnavailable, match="combat.card_draw"):
        collect_legacy_pilot(
            bridge, completed_intents_handle="completed-intents",
            seed="seed42", nature_policy=FirstLegalNature(),
            player_choose=lambda _obs, actions: actions[0],
        )
    assert not wire.handles


def test_bad_wire_law_and_outcome_drift_fail_closed():
    wire = FakePlayableWire()
    wire.bad_hello = True
    with pytest.raises(ControlledChanceUnavailable, match="unavailable"):
        LegacyPlayableBridge(wire, emulator_revision="sha")
    wire, bridge = make()
    wire.mutate_outcome = True
    frame = bridge.begin(completed_intents_handle="completed-intents", seed="seed42")
    with pytest.raises(ControlledChanceUnavailable, match="outcome replay drift"):
        bridge.resolve(frame, 0)


def test_fork_save_restore_keeps_the_same_mechanics_fingerprint():
    _wire, bridge = make()
    frame = bridge.begin(completed_intents_handle="completed-intents", seed="seed42")
    snapshot = bridge.save(frame)
    restored = bridge.load(snapshot)
    assert restored.fingerprint == frame.fingerprint
    branch = bridge.fork(frame)
    assert branch.fingerprint == frame.fingerprint
    bridge.release(frame, restored, branch)
