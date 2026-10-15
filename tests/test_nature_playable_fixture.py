"""Version-gated playable legacy fixture with joint player/Nature replay."""
from __future__ import annotations

import json

import pytest

from sts2_ai.emulator.controlled_chance import (
    ControlledChanceUnavailable, IsolatedNatureBridge,
)
from sts2_ai.emulator.playable_chance import (
    PLAYABLE_SCHEMA, PlayableBoundary, PlayableNatureBridge,
)
from sts2_ai.nature.playable_trajectory import play_controlled_fixture
from sts2_ai.nature.policies import FirstLegalNature, ScriptedNature


class Wire:
    def __init__(self):
        self.states = {}
        self.serial = 0
        self.actions = []
        self.scoped = True
        self.changed_parent = False

    def put(self, prefix, phase, choices=()):
        self.serial += 1
        handle = f"{prefix}{self.serial}"
        state = {"prefix": prefix, "phase": phase, "choices": tuple(choices)}
        self.states[handle] = state
        return self.frame(handle)

    def frame(self, handle):
        s = self.states[handle]
        prefix, phase, choices = s["prefix"], s["phase"], s["choices"]
        fingerprint = f"{prefix}:{phase}:{choices}"
        if prefix == "n":
            kind = "initial_hp" if phase == 0 else "intents"
            at_choice = len(choices) == 0
            pending = None if not at_choice else {
                "siteId": "combat.initial_unique_hp" if phase == 0 else "combat.committed_intent.index_draw",
                "streamId": "run.niche" if phase == 0 else "run.monster_ai",
                "ownerId": "creature:1", "ordinal": 0, "operationKind": "bounded_integer",
                "sourceVersion": "v0.111.0", "orderedCandidates": [], "boundedCount": 2,
            }
            return {"kind": kind, "natureHandle": handle, "fingerprint": fingerprint,
                    "complete": not at_choice, "pending": pending}
        at_chance = phase in (0, 2, 3)
        pending = None if not at_chance else {
            "eventId": f"native-event:{fingerprint}",
            "supportFingerprint": f"support:{phase}",
            "supportCount": 2,
            "siteId": ("controlled.fixture.opening_shuffle.swap_index" if phase == 0
                       else "combat.effect.random_enemy"),
            "streamId": "prototype.combat", "ownerId": "creature:1",
            "ordinal": phase, "operationKind": "bounded_integer",
            "sourceVersion": "v0.111.0", "orderedCandidates": [], "boundedCount": 2,
        }
        names = {0: "opening_shuffle", 1: "player", 2: "random_target",
                 3: "random_target", 4: "player"}
        return {"schemaId": PLAYABLE_SCHEMA, "controlledHandle": handle,
                "fingerprint": fingerprint, "pending": pending,
                "phase": names.get(phase, "player"), "terminal": phase == 5}

    def call(self, op, **kw):
        self.actions.append((op, kw))
        if op == "hello":
            return {
                "wireSchemaId": "prototype-ai-jsonl-v0",
                "fairPolicyId": "prototype-fair-v0",
                "natureBoundarySchemaId": "sts2-v111-nature-isolated-boundary-v1",
                "natureBoundaryScope": "isolated-combat-entry-only",
                "natureBoundaryKinds": ["initial_hp", "intents"],
                "controlledChanceSchemaId": PLAYABLE_SCHEMA if self.scoped else None,
                "controlledChanceScope": "isolated-playable-combat-stage-continuations-v1",
                "controlledChanceSupportedSites": [
                    "controlled.fixture.opening_shuffle.swap_index",
                    "combat.effect.random_enemy",
                ],
            }
        if op == "nature_begin":
            return self.put("n", 0)
        if op == "nature_advance":
            return self.put("n", 1)
        if op == "nature_resolve":
            s = self.states[kw["nature_handle"]]
            assert kw["fingerprint"] == self.frame(kw["nature_handle"])["fingerprint"]
            return self.put("n", s["phase"], (*s["choices"], kw["selected_index"]))
        if op == "controlled_begin":
            assert self.frame(kw["nature_handle"])["complete"]
            return self.put("c", 0)
        if op == "controlled_resolve":
            s = self.states[kw["controlled_handle"]]
            assert kw["fingerprint"] == self.frame(kw["controlled_handle"])["fingerprint"]
            out = self.put("c", s["phase"] + 1, (*s["choices"], kw["selected_index"]))
            out["parent"] = kw["controlled_handle"] if not self.changed_parent else "forged"
            return out
        if op == "controlled_step":
            s = self.states[kw["controlled_handle"]]
            assert s["phase"] in (1, 4)
            out = self.put("c", 2 if s["phase"] == 1 else 5, (*s["choices"], kw["action_id"]))
            out["parent"] = kw["controlled_handle"]
            return out
        if op == "controlled_observe":
            assert self.states[kw["controlled_handle"]]["phase"] in (1, 4)
            return {"policyId": "prototype-fair-v0", "schemaId": "prototype-ai-v0",
                    "payloadJson": json.dumps({"hp": 42, "turn": 1}),
                    "observationHash": "fair-hash", "actions": [
                        {"actionId": "end_turn", "kind": "end_turn", "payloadJson": "{}"},
                    ], "terminal": False}
        if op == "controlled_fork":
            source = self.frame(kw["controlled_handle"])
            out = self.put("c", self.states[kw["controlled_handle"]]["phase"],
                           self.states[kw["controlled_handle"]]["choices"])
            return {"controlledHandle": out["controlledHandle"],
                    "fingerprint": source["fingerprint"]}
        if op == "controlled_save":
            return {"schemaId": PLAYABLE_SCHEMA,
                    "snapshot": json.dumps(self.states[kw["controlled_handle"]])}
        if op == "controlled_load":
            s = json.loads(kw["snapshot"])
            out = self.put("c", s["phase"], s["choices"])
            del out["pending"]
            return out
        if op == "controlled_trace":
            out = self.frame(kw["controlled_handle"])
            return {"pending": out["pending"]}
        if op == "controlled_release_many":
            return {"released": sum(self.states.pop(h, None) is not None
                                    for h in kw["controlled_handles"])}
        if op == "nature_release_many":
            return {"released": sum(self.states.pop(h, None) is not None
                                    for h in kw["nature_handles"])}
        raise AssertionError(op)


FIXTURE = dict(seed="fake-seed", act=2, encounter_id="mytes_normal",
               native_entry="MytesNormal", total_floor=21)


def bridges(wire):
    return (IsolatedNatureBridge(wire, emulator_revision="pinned-emulator-sha"),
            PlayableNatureBridge(wire, emulator_revision="pinned-emulator-sha"))


def test_complete_combat_joint_script_is_exact_and_releases_handles():
    wire = Wire()
    isolated, playable = bridges(wire)
    choose = lambda frame: frame.action_ids[0]
    first = play_controlled_fixture(isolated, playable, FirstLegalNature(),
                                    player_choose=choose, **FIXTURE)
    assert [d.kind for d in first.decisions] == [
        "nature", "nature", "nature", "player", "nature", "nature", "player"
    ]
    assert wire.states == {}
    scripted = ScriptedNature.from_selections(
        (d.identity, int(d.selection)) for d in first.decisions if d.kind == "nature"
    )
    replay = play_controlled_fixture(isolated, playable, scripted,
                                     player_choose=choose, **FIXTURE)
    scripted.assert_complete()
    assert replay.replay_id == first.replay_id
    assert wire.states == {}


def test_capability_phase_support_and_player_fairness_fail_closed():
    wire = Wire()
    wire.scoped = False
    with pytest.raises(ControlledChanceUnavailable, match="capability"):
        PlayableNatureBridge(wire, emulator_revision="sha")
    wire.scoped = True
    isolated, playable = bridges(wire)
    incomplete = isolated.begin(kind="initial_hp", **FIXTURE)
    with pytest.raises(ValueError, match="completed"):
        playable.begin(incomplete, seed=FIXTURE["seed"])
    with pytest.raises(ControlledChanceUnavailable, match="support count"):
        PlayableBoundary.decode({**Wire().put("c", 0), "pending": {
            **Wire().put("c", 0)["pending"], "supportCount": 99,
        }})
    with pytest.raises(ValueError, match="Player observations"):
        playable.observe(PlayableBoundary.decode(wire.put("c", 0)))


def test_parent_mismatch_rejected_and_budget_is_censored():
    wire = Wire()
    isolated, playable = bridges(wire)
    wire.changed_parent = True
    with pytest.raises(ControlledChanceUnavailable, match="parent mismatch"):
        play_controlled_fixture(isolated, playable, FirstLegalNature(),
                                player_choose=lambda f: f.action_ids[0], **FIXTURE)
    assert wire.states == {}
    wire.changed_parent = False
    with pytest.raises(RuntimeError, match="censored"):
        play_controlled_fixture(isolated, playable, FirstLegalNature(),
                                player_choose=lambda f: f.action_ids[0],
                                max_boundaries=2, **FIXTURE)
    assert wire.states == {}


def test_save_load_fork_at_unresolved_chance():
    wire = Wire()
    _, playable = bridges(wire)
    start = PlayableBoundary.decode(wire.put("c", 0))
    fork = playable.fork(start)
    assert fork.fingerprint == start.fingerprint
    restored = playable.load(playable.save(start))
    assert restored.fingerprint == start.fingerprint
    assert restored.event == start.event
    assert playable.resolve(restored, 1).phase == "player"
