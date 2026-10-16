"""Synthetic contract tests for the exact v111 P0 JSONL/Python wire shape.

Live .NET integration is separately tested by nature-real-combat CI.
These fakes deliberately do not model game outcomes or native probability.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from sts2_ai.emulator.semantic_p0_combat import (
    CombatFixture, P0SemanticInterfaceError, _pending_event,
    check_capabilities, play_real_two_site,
)
from sts2_ai.nature.two_site_opponent import TwoSiteNature


@pytest.fixture
def fixture() -> CombatFixture:
    return CombatFixture(
        fixture_id="synthetic-contract-only", seed="test-p0", act=1,
        encounter_id="proto.encounter.shrinker_beetle_weak",
        native_entry="ShrinkerBeetleWeak", total_floor=2,
    )


def intent():
    return {
        "kind": "enemy_intent",
        "siteId": "combat.committed_intent.index_draw",
        "streamId": "run.monster_ai",
        "ownerId": "creature:1",
        "ordinal": 0, "operationKind": "bounded_integer",
        "supportCount": 4,
        "semanticIntentSupportSchemaId": "sts2-v111-semantic-weighted-intent-v1",
        "semanticCandidates": [
            {"intentId": "intent:light", "moveId": "LIGHT", "moveIndex": 0,
             "rawWeight": 1, "ticketStart": 0},
            {"intentId": "intent:heavy", "moveId": "HEAVY", "moveIndex": 1,
             "rawWeight": 3, "ticketStart": 1},
        ],
        "rawEligibleWeights": [1, 3],
        "totalRawWeight": 4,
        "semanticSupportFingerprint": "semantic-support-1",
        "supportFingerprint": "bounded-request-1",
        "weightsVerified": False,
        "probabilityStatus": "UNVERIFIED",
    }


def draw():
    return {
        "kind": "draw_card",
        "schemaId": "sts2-v111-semantic-draw-v1",
        "eventId": "draw-evt-1",
        "siteId": "combat.deck.draw_card", "streamId": "run.shuffle",
        "ownerId": "player",
        "ordinal": 1, "operationKind": "choose_index",
        "shuffleEpoch": 2,
        "supportFingerprint": "draw-pool-2",
        "constraintFingerprint": "hidden-top-pool-2",
        "probabilityStatus": "UNVERIFIED",
        "candidates": [
            {"index": 0, "cardInstanceId": 11, "cardModelId": "proto.silent.strike"},
            {"index": 1, "cardInstanceId": 17, "cardModelId": "proto.silent.defend"},
        ],
    }


@dataclass(frozen=True)
class FakeState:
    session: FakeSession
    stage: int

    @property
    def fingerprint(self):
        return f"fixture-p0-stage-{self.stage}"

    @property
    def handle(self):
        return f"m{self.stage}"

    @property
    def boundary(self):
        return ("player", "nature", "nature", "terminal")[self.stage]

    @property
    def terminal(self):
        return self.stage == 3

    @property
    def outcome(self):
        return "victory" if self.terminal else None

    @property
    def exit_hp(self):
        return 44 if self.terminal else None

    @property
    def pending(self):
        return intent() if self.stage == 1 else draw() if self.stage == 2 else None

    def observe(self):
        assert self.boundary == "player"
        return {"combat": {"hand": [{"instance_id": 1,
                                    "card_id": "proto.silent.strike"}]}}

    def actions(self):
        assert self.boundary == "player"
        return [{"actionId": "e", "kind": "end_turn"}]

    def step(self, action_id):
        assert self.stage == 0 and action_id == "e"
        return FakeState(self.session, 1)

    def resolve_intent(self, chosen):
        assert self.stage == 1 and chosen in ("intent:light", "intent:heavy")
        return FakeState(self.session, 2)

    def resolve(self, index):
        assert self.stage == 2 and index in (0, 1)
        return FakeState(self.session, 3)

    def save_snapshot(self):
        return str(self.stage)

    def status(self):
        return {"otherChanceProvider": (
            "engine-owned-native-v111-cost-stream-unverified-v1"
        )}


class FakeSession:
    capabilities = {
        "wireSchemaId": "prototype-ai-jsonl-v0",
        "semanticChanceSchemaId": "sts2-v111-semantic-playable-p0-v1",
        "semanticPlayable": True,
        "semanticControlledSites": ["enemy_intent", "draw_card"],
        "semanticIntentSupportSchemaId": "sts2-v111-semantic-weighted-intent-v1",
        "semanticSeededOpeningProvider": "engine-owned-seeded-source-shaped-unverified-v1",
        "semanticSeededOpeningDrawHooksId":
            "engine-owned-seeded-source-shaped-draw-hooks-unverified-v1",
        "fairPolicyId": "prototype-fair-v0",
        "semanticProbabilityStatus": "UNVERIFIED",
        "semanticOtherChanceProviders": [
            "engine-owned-deterministic-unverified-v1",
            "engine-owned-native-v111-cost-stream-unverified-v1",
        ],
    }

    def begin_seeded_semantic_combat(self, seed, **kwargs):
        assert seed == "test-p0"
        assert kwargs["other_chance_provider_id"].startswith("engine-owned-native")
        self.kwargs = kwargs
        return FakeState(self, 0)

    def load_semantic_snapshot(self, snapshot):
        return FakeState(self, int(snapshot))

    def release_semantic_many(self, states):
        self.released = len(states)
        return len(states)


def test_actual_p0_protocol_deferred_draw_and_weighted_intents(fixture):
    session = FakeSession()
    episode = play_real_two_site(
        session, fixture, emulator_revision="real-protocol-fake",
        nature=TwoSiteNature(seed=29),
    )
    assert episode.trace.outcome == "victory"
    assert episode.trace.exit_hp == 44
    assert session.released == 5
    assert episode.player_return == pytest.approx(1.44)
    assert episode.trace.probability_status == "UNVERIFIED"
    assert [x.phase for x in episode.trace.decisions] == [
        "player", "nature:enemy_intent", "nature:draw_card",
    ]
    assert len(episode.nature_episode.choices) == 2
    assert episode.nature_episode.choices[0].baseline == pytest.approx((0.25, 0.75))
    replay = play_real_two_site(
        session, fixture, emulator_revision="real-protocol-fake",
        scripted_choices=episode.trace.replay_choices(),
    )
    assert episode.trace.digest == replay.trace.digest


def test_unverified_native_law_and_guarded_support(fixture):
    intent_event, labels, ids = _pending_event(intent(), "f")
    assert labels == ("LIGHT", "HEAVY")
    assert ids == ("intent:light", "intent:heavy")
    assert intent_event.normalized_intent_weights == pytest.approx((0.25, 0.75))
    draw_event, models, card_ids = _pending_event(draw(), "f")
    assert models == ("proto.silent.strike", "proto.silent.defend")
    assert card_ids == ("11", "17")
    assert draw_event.randomization_id == "deck:2"
    with pytest.raises(P0SemanticInterfaceError, match="law status"):
        _pending_event({**intent(), "probabilityStatus": "AUDITED_NATIVE"}, "f")
    with pytest.raises(P0SemanticInterfaceError, match="Outside"):
        _pending_event({"kind": "shuffle_swap", "probabilityStatus": "UNVERIFIED"}, "f")
    broken = FakeSession()
    broken.capabilities = {
        **broken.capabilities, "semanticChanceSchemaId": "legacy-fisher-yates",
    }
    with pytest.raises(P0SemanticInterfaceError, match="handshake"):
        check_capabilities(broken, fixture)


def test_scripted_wrong_card_or_incomplete_trajectory_rejected(fixture):
    session = FakeSession()
    with pytest.raises(P0SemanticInterfaceError, match="support/constraint divergence"):
        play_real_two_site(
            session, fixture, emulator_revision="fake",
            scripted_choices=(
                ("player", "forged", "e"),
            ),
        )
    with pytest.raises(P0SemanticInterfaceError, match="censored"):
        play_real_two_site(
            session, fixture, emulator_revision="fake",
            nature=TwoSiteNature(seed=7), max_boundaries=1,
        )
