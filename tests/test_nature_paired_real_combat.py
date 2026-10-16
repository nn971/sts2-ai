"""Source-matched Nature/ordinary comparator contract and replay oracle tests.

Synthetic state machines exercise only the connector protocol. .NET mechanics
are checked separately by cross-repository integration CI.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from sts2_ai.nature.paired_real_combat import (
    NATIVE_PROVIDER,
    SOURCE_ROUTE_PROVIDER,
    PairedCombatError,
    PairedSourceFixture,
    check_comparator_capabilities,
    paired_record,
    play_paired_source_combat,
)
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def weighted_intent():
    return {
        "kind": "enemy_intent", "siteId": "combat.committed_intent.index_draw",
        "ownerId": "creature:1", "ordinal": 0,
        "semanticIntentSupportSchemaId": "sts2-v111-semantic-weighted-intent-v1",
        "semanticCandidates": [
            {"intentId": "intent:light", "moveId": "LIGHT", "rawWeight": 3},
            {"intentId": "intent:heavy", "moveId": "HEAVY", "rawWeight": 1},
        ],
        "totalRawWeight": 4, "weightsVerified": False,
        "semanticSupportFingerprint": "intent-sup-v1",
        "supportFingerprint": "bounded-tickets-v1",
        "probabilityStatus": "UNVERIFIED",
    }


def draw_card():
    return {
        "kind": "draw_card", "eventId": "draw-0001",
        "siteId": "combat.deck.draw_card", "ownerId": "player",
        "ordinal": 1, "shuffleEpoch": 2,
        "supportFingerprint": "deck-support",
        "constraintFingerprint": "private-position-constraint",
        "probabilityStatus": "UNVERIFIED",
        "candidates": [
            {"index": 0, "cardInstanceId": 3,
             "cardModelId": "proto.silent.strike"},
            {"index": 1, "cardInstanceId": 4,
             "cardModelId": "proto.silent.defend"},
        ],
    }


@dataclass(frozen=True)
class FakeState:
    session: FakeSession
    arm: str
    stage: int

    @property
    def handle(self):
        return f"{self.arm}-{self.stage}"

    @property
    def fingerprint(self):
        return f"{self.arm}-source-state-{self.stage}"

    @property
    def boundary(self):
        if self.arm == "ordinary":
            return "player" if self.stage == 0 else "terminal"
        return ("nature", "player", "nature", "terminal")[self.stage]

    @property
    def terminal(self):
        return self.boundary == "terminal"

    @property
    def outcome(self):
        return "victory" if self.terminal else None

    @property
    def exit_hp(self):
        return (44 if self.arm == "nature" else 52) if self.terminal else None

    @property
    def pending(self):
        if self.arm == "nature" and self.stage == 0:
            return weighted_intent()
        if self.arm == "nature" and self.stage == 2:
            return draw_card()
        return None

    def status(self):
        return {
            "sourceOpeningFingerprint": self.session.opening,
            "sourceRouteProvider": SOURCE_ROUTE_PROVIDER,
            "sourceRouteProbabilityStatus": "UNVERIFIED",
            "controllerMode": (
                "nature-two-site" if self.arm == "nature"
                else "engine-owned-native-v111-policy-baseline"
            ),
            "nativePolicyProvider": (
                NATIVE_PROVIDER if self.arm == "ordinary" else None
            ),
            "chanceSites": (
                ["enemy_intent", "draw_card"] if self.arm == "nature" else []
            ),
            "nativePolicyDecisionCount": (
                32 if self.arm == "ordinary" else 0
            ),
            "nativePolicyShuffleCount": (
                12 if self.arm == "ordinary" else 0
            ),
            "probabilityStatus": "UNVERIFIED",
        }

    def observe(self):
        if self.boundary != "player":
            raise ValueError("Not public boundary")
        return {
            "combat": {
                "hand": [{"instance_id": 5, "card_id": "proto.silent.strike"}],
                "enemies": [{"id": "creature:1", "hp": 9,
                             "move_id": (
                                 "LIGHT" if self.arm == "nature" else "HEAVY"
                             ), "intent_damage": 3}],
            },
            "player": {"hp": 52},
        }

    def actions(self):
        return [{"kind": "end_turn", "actionId": "finish", "payloadJson": "{}"}]

    def resolve_intent(self, choice):
        assert self.arm == "nature" and self.stage == 0
        assert choice in {"intent:light", "intent:heavy"}
        return FakeState(self.session, self.arm, 1)

    def step(self, action):
        assert action == "finish"
        if self.arm == "nature":
            assert self.stage == 1
            return FakeState(self.session, self.arm, 2)
        assert self.stage == 0
        return FakeState(self.session, self.arm, 1)

    def resolve_draw_card(self, instance_id):
        assert self.arm == "nature" and self.stage == 2
        assert instance_id in (3, 4)
        return FakeState(self.session, self.arm, 3)

    def save_snapshot(self):
        return f"{self.arm}:{self.stage}"


class FakeSession:
    capabilities = {
        "wireSchemaId": "prototype-ai-jsonl-v0",
        "semanticChanceSchemaId": "sts2-v111-semantic-playable-p0-v1",
        "semanticPlayable": True,
        "semanticControlledSites": ["enemy_intent", "draw_card"],
        "semanticIntentSupportSchemaId": "sts2-v111-semantic-weighted-intent-v1",
        "semanticNativePolicyComparatorId": NATIVE_PROVIDER,
        "semanticSeededRouteEntryProviderId": SOURCE_ROUTE_PROVIDER,
        "fairPolicyId": "prototype-fair-v0",
        "semanticProbabilityStatus": "UNVERIFIED",
    }

    def __init__(self):
        self.opening = "identical-pre-controller-opening"
        self.released = 0
        self.started = []

    def begin_seeded_semantic_route(self, seed, *, region, native_policy_baseline):
        assert seed == "toy-seed" and region == "Overgrowth"
        arm = "ordinary" if native_policy_baseline else "nature"
        self.started.append(arm)
        return FakeState(self, arm, 0)

    def load_semantic_snapshot(self, snapshot):
        arm, stage = snapshot.split(":")
        return FakeState(self, arm, int(stage))

    def release_semantic_many(self, states):
        self.released += len(states)
        return len(states)


def fixture():
    return PairedSourceFixture("toy-source", "toy-seed", "Overgrowth")


def test_same_source_intent_mask_terminal_and_full_dual_replay():
    session = FakeSession()
    nature, ordinary = play_paired_source_combat(
        session, fixture(), nature=TwoSiteNature(seed=17),
    )
    assert session.started == ["nature", "ordinary", "nature", "ordinary"]
    assert nature.opening_fingerprint == ordinary.opening_fingerprint
    assert nature.first_public_fingerprint == ordinary.first_public_fingerprint
    assert [x.phase for x in nature.decisions] == [
        "nature:enemy_intent", "player", "nature:draw_card",
    ]
    assert [x.phase for x in ordinary.decisions] == ["player"]
    assert nature.outcome == ordinary.outcome == "victory"
    assert nature.exit_hp == 44 and ordinary.exit_hp == 52
    assert len(nature.nature_choices) == 2
    assert ordinary.nature_choices == ()
    assert ordinary.native_shuffle_count == 12
    assert nature.native_shuffle_count == 0
    result = paired_record(fixture(), nature, ordinary)
    assert result["same_opening"] and result["same_initial_public_state"]
    assert result["nature_minus_ordinary_return"] == pytest.approx(-0.08)
    assert result["probability_status"] == "UNVERIFIED"
    assert session.released == 14


def test_capability_failure_and_source_mismatch_fail_closed():
    bad = FakeSession()
    bad.capabilities = {
        **bad.capabilities,
        "semanticNativePolicyComparatorId": "missing",
    }
    with pytest.raises(PairedCombatError, match="lacks paired"):
        check_comparator_capabilities(bad)
    with pytest.raises(ValueError, match="source-routed"):
        PairedSourceFixture("f", "s", "Hive")
    fake = FakeSession()
    fake.opening = ""
    with pytest.raises(PairedCombatError, match="pre-controller"):
        play_paired_source_combat(fake, fixture(), nature=TwoSiteNature(seed=4))
    assert fake.released >= 1


def test_boundary_censoring_is_not_reclassified_as_defeat():
    fake = FakeSession()
    with pytest.raises(PairedCombatError, match="CENSORED"):
        play_paired_source_combat(
            fake, fixture(), nature=TwoSiteNature(seed=8), max_boundaries=1,
        )
