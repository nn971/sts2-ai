"""Synthetic semantic Nature wire contract tests (no live emulator capability yet).

The fake service models *only* source-shaped boundary metadata and event
ordering. It is not used as the gameplay or deck-state implementation.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from sts2_ai.emulator.semantic_chance import (
    PROPOSED_WIRE_SCHEMA,
    SemanticCapabilityUnavailable,
    SemanticNatureBridge,
)
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.semantic import SemanticEvent, SemanticProposal
from sts2_ai.nature.semantic_policies import (
    ScriptedSemanticNature,
    SourceWeightedIntentNature,
    UniformSemanticNature,
)
from sts2_ai.nature.semantic_trajectory import collect_semantic_trajectory


def pending(*, operation="draw_card", candidates=("card:a", "card:b"),
            weights=None, status="UNVERIFIED", epoch="shuffle-1"):
    return {
        "eventId": "evt-" + operation,
        "boundaryFingerprint": "boundary-0",
        "siteId": "combat.deck.reveal" if operation != "enemy_intent"
                  else "combat.enemy.commit",
        "ownerId": "draw-pile" if operation != "enemy_intent" else "creature:1",
        "ordinal": 0, "operation": operation, "candidateIds": list(candidates),
        "supportFingerprint": canonical_sha256(candidates),
        "randomizationId": epoch, "constraintFingerprint": "constraints-1",
        "sourceVersion": "v0.111.0", "representation": "nature-deferred-v1",
        "probabilityStatus": status,
        "rawWeights": list(weights) if weights is not None else None,
        "weightTotal": sum(weights) if weights is not None else None,
    }


def test_semantic_contract_is_separate_from_primitive_swaps():
    e = SemanticEvent.decode(pending())
    assert e.support_count == 2 and e.check_index(1) == "card:b"
    assert e.replay_identity != SemanticEvent.decode({
        **pending(), "constraintFingerprint": "new-public-peek",
    }).replay_identity
    with pytest.raises(ValueError, match="legacy swap tickets"):
        SemanticEvent.decode({**pending(), "representation": "legacy-fisher-yates"})
    with pytest.raises(NotImplementedError, match="Unsupported"):
        SemanticEvent.decode({**pending(), "operation": "shuffle_swap"})
    with pytest.raises(ValueError, match="unique stable"):
        SemanticEvent.decode(pending(candidates=("card:a", "card:a")))
    with pytest.raises(ValueError, match="identity/index mismatch"):
        SemanticProposal(0, "card:b", "q").validate(e)
    with pytest.raises(ValueError, match="Incomplete"):
        data = pending()
        data.pop("constraintFingerprint")
        SemanticEvent.decode(data)


def test_intent_weights_q_are_not_calibrated_reference_p():
    e = SemanticEvent.decode(
        pending(operation="enemy_intent",
                candidates=("attack", "defend", "not-eligible"),
                weights=(1, 3, 0), epoch="intent-commit"),
    )
    assert e.normalized_intent_weights == pytest.approx((0.25, 0.75, 0.0))
    assert e.reference_law_id is None and e.probability_status == "UNVERIFIED"
    with pytest.raises(ValueError, match="ineligible"):
        e.check_index(2)
    with pytest.raises(ValueError, match="weights"):
        SemanticEvent.decode({
            **pending(operation="enemy_intent", candidates=("a", "b"),
                      weights=(1, 1)), "weightTotal": 3,
        })
    with pytest.raises(ValueError, match="reference law"):
        SemanticEvent.decode({
            **pending(operation="enemy_intent", candidates=("a", "b"),
                      weights=(1, 1)), "referenceLawId": "unverified-native",
        })
    with pytest.raises(ValueError, match="source-law"):
        SemanticEvent.decode({
            **pending(operation="enemy_intent", candidates=("a", "b"),
                      weights=(1, 1)), "probabilityStatus": "AUDITED_NATIVE",
        })
    policy = SourceWeightedIntentNature(seed=19)
    out = policy.propose(e)
    assert out.selected_id != "not-eligible"
    assert math.exp(out.log_q) == pytest.approx(
        e.normalized_intent_weights[out.selected_index],
    )
    other = UniformSemanticNature(seed=19).propose(e)
    assert math.exp(other.log_q) == pytest.approx(0.5)
    assert other.selected_id != "not-eligible"
    with pytest.raises(ValueError, match="does not describe enemy intent"):
        policy.propose(SemanticEvent.decode(pending()))


def test_schedule_independent_semantic_outcomes():
    events = [
        SemanticEvent.decode(pending()),
        SemanticEvent.decode(pending(operation="enemy_intent",
                                     candidates=("a", "b"), weights=(2, 1))),
    ]
    for event in events:
        policy = UniformSemanticNature(73)
        expected = policy.propose(event)
        with ThreadPoolExecutor(max_workers=15) as pool:
            actual = tuple(pool.map(lambda _, p=policy, e=event: p.propose(e), range(150)))
        assert actual == (expected,) * 150


class FakeSemanticWire:
    """A synthetic sequence: peek, known-card draw, unknown draw, intent."""

    def __init__(self, *, legacy=False):
        self.legacy = legacy
        self.states = {}
        self.counter = 0
        self.actions_seen = []

    def _make(self, state):
        self.counter += 1
        handle = f"s{self.counter}"
        state = json.loads(json.dumps(state))
        self.states[handle] = state
        stage = state["stage"]
        fingerprint = canonical_sha256(state)
        result = {
            "semanticChanceSchemaId": PROPOSED_WIRE_SCHEMA,
            "semanticHandle": handle, "fingerprint": fingerprint,
            "phase": "chance" if stage in (0, 1, 3) else
                     "player" if stage in (2, 4) else "terminal",
            "pending": None,
        }
        if stage in (0, 1, 3):
            cards = ["card:a", "card:b", "card:c"]
            candidates = (
                cards if stage == 0 else
                [c for c in cards if c != state["chosen"][0]] if stage == 1 else
                ["attack", "defend", "forbidden"]
            )
            operation = ("reveal_card" if stage == 0 else
                         "draw_card" if stage == 1 else "enemy_intent")
            raw = pending(
                operation=operation, candidates=tuple(candidates),
                weights=(1, 3, 0) if stage == 3 else None,
                epoch="shuffle-epoch-1" if stage != 3 else "intent-turn-2",
            )
            raw["eventId"] = f"event-{stage}-" + canonical_sha256(state)
            raw["boundaryFingerprint"] = fingerprint
            raw["ordinal"] = stage
            raw["constraintFingerprint"] = canonical_sha256(
                [state["chosen"], "known-top-reused-without-new-event"]
            )
            result["pending"] = raw
        elif stage in (2, 4):
            result["fairPolicyId"] = "prototype-fair-v0"
            result["observation"] = {
                "hand_count": 2, "known_top_reused": True,
                "enemy_intent": state["chosen"][-1] if stage == 4 else "previous",
            }
            result["legalActions"] = ["end_turn"]
        else:
            result["terminal"] = "synthetic_completed"
        return result

    def call(self, operation, **payload):
        if operation == "hello":
            return {
                "wireSchemaId": "prototype-ai-jsonl-v0",
                "semanticChanceSchemaId": PROPOSED_WIRE_SCHEMA if not self.legacy else None,
                "semanticChanceRepresentationId": "nature-deferred-v1",
                "semanticChanceOperations": [
                    "draw_card", "reveal_card", "enemy_intent",
                    "select_distinct", "generate_offer",
                ],
                "fairPolicyId": "prototype-fair-v0",
            }
        if operation == "semantic_begin":
            return self._make({
                "seed": payload["seed"], "fixture": payload["fixture_id"],
                "stage": 0, "chosen": [],
            })
        if operation == "semantic_resolve":
            state = self.states[payload["semantic_handle"]]
            expected = self._make_reconstructed_pending(state)
            if (payload["fingerprint"] != canonical_sha256(state)
                    or payload["event_id"] != expected["eventId"]
                    or payload["support_fingerprint"] != expected["supportFingerprint"]
                    or payload["constraint_fingerprint"] != expected["constraintFingerprint"]
                    or expected["candidateIds"][payload["selected_index"]]
                    != payload["selected_id"]):
                raise ValueError("Stale semantic pending event")
            return self._make({
                **state, "stage": state["stage"] + 1,
                "chosen": [*state["chosen"], payload["selected_id"]],
            })
        if operation == "semantic_step":
            state = self.states[payload["semantic_handle"]]
            if state["stage"] not in (2, 4) or payload["action_id"] != "end_turn":
                raise ValueError("Illegal player step")
            if payload["fingerprint"] != canonical_sha256(state):
                raise ValueError("Stale player state")
            self.actions_seen.append(payload["action_id"])
            return self._make({**state, "stage": state["stage"] + 1})
        if operation == "semantic_fork":
            return self._make(self.states[payload["semantic_handle"]])
        if operation == "semantic_save":
            return {
                "semanticChanceSchemaId": PROPOSED_WIRE_SCHEMA,
                "snapshot": json.dumps(self.states[payload["semantic_handle"]]),
            }
        if operation == "semantic_load":
            return self._make(json.loads(payload["snapshot"]))
        if operation == "semantic_release_many":
            return {"released": sum(
                self.states.pop(h, None) is not None for h in payload["semantic_handles"]
            )}
        raise AssertionError(operation)

    def _make_reconstructed_pending(self, state):
        # Temporary frame only for validation, removing its handle.
        frame = self._make(state)
        return frame["pending"]


def test_new_capability_required_and_legacy_fails_closed():
    with pytest.raises(SemanticCapabilityUnavailable, match="unavailable"):
        SemanticNatureBridge(FakeSemanticWire(legacy=True), emulator_revision="rev-test")


def test_semantic_bridge_peek_known_top_draw_and_visible_intent_replay():
    wire = FakeSemanticWire()
    bridge = SemanticNatureBridge(wire, emulator_revision="fake-semantic-v1")
    first = bridge.begin(fixture_id="toy", seed="49")
    assert first.phase == "chance" and first.observation is None
    forked = bridge.fork(first)
    assert forked.fingerprint == first.fingerprint
    snapshot = bridge.save(first)
    restored = bridge.load(snapshot)
    assert restored.event is not None
    assert restored.event.replay_identity == first.event.replay_identity
    policy = UniformSemanticNature(seed=17)
    recorded = collect_semantic_trajectory(
        bridge, policy, fixture_id="toy", seed="49",
        player_choose=lambda obs, menu: menu[0],
    )
    # No separate draw event for the top card after it was revealed.
    assert tuple(d.kind for d in recorded.decisions) == (
        "nature", "nature", "player", "nature", "player",
    )
    assert recorded.terminal_outcome == "synthetic_completed"
    assert len(recorded.nature_script()) == 3
    scripted = ScriptedSemanticNature.from_decisions(recorded.nature_script())
    replay = collect_semantic_trajectory(
        bridge, scripted, fixture_id="toy", seed="49",
        player_choose=lambda obs, menu: menu[0],
    )
    scripted.assert_complete()
    assert recorded.mechanical_signature() == replay.mechanical_signature()
    assert recorded.terminal_fingerprint == replay.terminal_fingerprint
    assert bridge.release(first, forked, restored) == 3


def test_semantic_player_fairness_and_stale_pending_guards():
    wire = FakeSemanticWire()
    bridge = SemanticNatureBridge(wire, emulator_revision="fake")
    frame = bridge.begin(fixture_id="toy", seed="49")
    with pytest.raises(ValueError, match="Illegal semantic player action"):
        bridge.step(frame, "end_turn")
    event = frame.event
    assert event is not None
    with pytest.raises(ValueError, match="identity/index mismatch"):
        bridge.resolve(frame, SemanticProposal(0, "incorrect-card", "script"))
    selected = SemanticProposal(0, event.candidate_ids[0], "script")
    child = bridge.resolve(frame, selected)
    assert child.phase == "chance"
    with pytest.raises(ValueError, match="Stale semantic pending"):
        bridge.resolve(
            replace(frame, fingerprint="forged-stale-fingerprint"),
            SemanticProposal(1, event.candidate_ids[1], "script"),
        )
