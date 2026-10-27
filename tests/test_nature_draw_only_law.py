"""Strict independent declared-law parsing for single-combat draw-only Nature."""
from __future__ import annotations

import copy
import math

import pytest

from sts2_ai.nature.draw_only_episode import _score, check_draw_only_capabilities
from sts2_ai.nature.draw_only_p0 import (
    CHANCE_LAW_SCHEMA,
    DeclaredLaw,
    DrawOnlyContractError,
    DrawOnlyRouter,
    SeededLawSampler,
)
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def pending(kind="draw_card", *, weights=(1, 1)):
    ids = ["card_instance:7", "card_instance:11"] if kind == "draw_card" else [
        "intent:a", "intent:b",
    ]
    result = {
        "kind": kind, "siteId": "combat.draw_pile.draw_card"
        if kind == "draw_card" else "combat.committed_intent.index_draw",
        "ownerId": "owner", "ordinal": 0, "operationKind": "choose_index",
        "eventId": "event",
        "probabilityStatus": "UNVERIFIED",
        "supportFingerprint": "pending-support",
    }
    if kind == "draw_card":
        result.update(
            shuffleEpoch=0, constraintFingerprint="constraint",
            candidates=[
                {"cardInstanceId": 7, "cardModelId": "proto.silent.strike", "index": 0},
                {"cardInstanceId": 11, "cardModelId": "proto.silent.defend", "index": 1},
            ],
        )
        condition = "constraint"
    else:
        result.update(
            semanticIntentSupportSchemaId="sts2-v111-semantic-weighted-intent-v1",
            weightsVerified=False, semanticSupportFingerprint="intent-support",
            totalRawWeight=sum(weights),
            semanticCandidates=[
                {"intentId": "intent:a", "moveId": "move:a", "rawWeight": weights[0]},
                {"intentId": "intent:b", "moveId": "move:b", "rawWeight": weights[1]},
            ],
        )
        condition = "intent-condition"
    result["chanceLaw"] = {
        "schemaId": CHANCE_LAW_SCHEMA, "lawId": "law", "eventId": "event",
        "siteId": result["siteId"], "operationKind": "choose_index",
        "ownerId": result["ownerId"], "ordinal": result["ordinal"],
        "supportFingerprint": "law-support",
        "conditioningPolicyId": "test-policy",
        "conditioningFingerprint": condition,
        "probabilityStatus": "UNVERIFIED",
        "probabilityProvenance": "source-shaped-unverified",
        "totalWeight": sum(weights),
        "outcomes": [
            {"outcomeId": id_, "weight": weight}
            for id_, weight in zip(ids, weights, strict=True)
        ],
    }
    return result


def event_number(raw, ordinal):
    modified = copy.deepcopy(raw)
    modified["eventId"] = f"event-{ordinal}"
    modified["ordinal"] = ordinal
    modified["chanceLaw"]["eventId"] = modified["eventId"]
    modified["chanceLaw"]["ordinal"] = ordinal
    modified["chanceLaw"]["lawId"] = f"law-{ordinal}"
    return modified


def test_intents_sample_only_declared_weighted_law_and_do_not_update_nature():
    raw = pending("enemy_intent", weights=(1, 3))
    law = DeclaredLaw.from_pending(raw)
    assert law.probabilities == (0.25, 0.75)
    nature = TwoSiteNature(seed=5)
    router = DrawOnlyRouter(nature, SeededLawSampler(22))
    choices = [
        router.choose(event_number(raw, i), fingerprint="parent",
                      episode_key="one", ordinal=i)
        for i in range(400)
    ]
    assert all(row.operation == "enemy_intent" for row in choices)
    assert all(row.controller == "engine-law-reference" for row in choices)
    assert all(row.nature_choice is None for row in choices)
    assert router.reference.calls == 400
    assert sum(row.selected_outcome_id == "intent:b" for row in choices) in range(260, 340)
    assert all(row.log_p == pytest.approx(row.log_q) for row in choices)
    assert nature.revision == 0
    again = DrawOnlyRouter(nature, SeededLawSampler(22))
    assert choices == [
        again.choose(event_number(raw, i), fingerprint="parent",
                     episode_key="one", ordinal=i)
        for i in range(400)
    ]


def test_exact_reference_draws_use_declared_law_and_ignore_learned_weights():
    raw = pending("draw_card", weights=(3, 1))
    law = DeclaredLaw.from_pending(raw)
    assert law.probabilities == (0.75, 0.25)
    nature = TwoSiteNature(seed=18)
    nature.weights["draw_card|context|proto.silent.strike"] = 100.0
    router = DrawOnlyRouter(nature, SeededLawSampler(5))
    choices = [
        router.choose(event_number(raw, i), fingerprint="parent",
                      episode_key="check", ordinal=i, draw_mode="reference")
        for i in range(400)
    ]
    assert router.reference.calls == 400
    assert all(ch.controller == "engine-law-reference" and
               ch.nature_choice is None and ch.log_q == ch.log_p
               for ch in choices)
    assert sum(ch.selected_index == 0 for ch in choices) in range(260, 340)
    with pytest.raises(DrawOnlyContractError, match="Nonuniform physical draw"):
        router.choose(raw, fingerprint="parent", episode_key="check",
                      ordinal=0, draw_mode="learned")




def test_source_verified_native_intent_grid_support_and_stale_pending_status():
    # The current emulator keeps a conservative UNVERIFIED event metadata
    # status, while ChanceLaw is authoritative and SOURCE_VERIFIED.
    raw = pending("enemy_intent", weights=(1, 3))
    raw["chanceLaw"]["probabilityStatus"] = "SOURCE_VERIFIED"
    raw["chanceLaw"]["conditioningPolicyId"] = "source-filtered-intent-weights-v1"
    raw["chanceLaw"]["probabilityProvenance"] = "source-nextdouble-grid"
    raw["chanceLaw"]["totalWeight"] = 1 << 53
    raw["chanceLaw"]["outcomes"][0]["weight"] = 1 << 51
    raw["chanceLaw"]["outcomes"][1]["weight"] = 3 << 51
    law = DeclaredLaw.from_pending(raw)
    assert law.probabilities == (0.25, 0.75)
    assert law.total_weight == 1 << 53
    router = DrawOnlyRouter(TwoSiteNature(seed=1), SeededLawSampler(3))
    choice = router.choose(raw, fingerprint="parent",
                           episode_key="normal", ordinal=0)
    assert choice.controller == "engine-law-reference"
    assert choice.nature_choice is None

    corrupt = copy.deepcopy(raw)
    corrupt["chanceLaw"]["outcomes"][0]["weight"] += 1
    corrupt["chanceLaw"]["outcomes"][1]["weight"] -= 1
    with pytest.raises(DrawOnlyContractError, match="grid disagrees"):
        DeclaredLaw.from_pending(corrupt)

    wrong_raw = copy.deepcopy(raw)
    wrong_raw["semanticCandidates"][0]["rawWeight"] = 2
    with pytest.raises(DrawOnlyContractError):
        DeclaredLaw.from_pending(wrong_raw)

    draw = pending("draw_card")
    draw["chanceLaw"]["probabilityStatus"] = "SOURCE_VERIFIED"
    with pytest.raises(DrawOnlyContractError, match="provenance"):
        DeclaredLaw.from_pending(draw)

def test_reference_intent_randomness_unaffected_by_inserted_draw_samples():
    intent = DeclaredLaw.from_pending(event_number(
        pending("enemy_intent", weights=(1, 3)), 7
    ))
    draw = DeclaredLaw.from_pending(event_number(pending("draw_card"), 5))
    plain = SeededLawSampler(99)
    mixed = SeededLawSampler(99)
    expected = plain.sample(intent)
    mixed.sample(draw)
    assert mixed.sample(intent) == expected
    assert mixed.sample(intent) == expected  # Stable after replay/fork
    assert plain.calls == 1
    assert mixed.calls == 3


def test_draw_choices_are_physical_instances_not_models_or_new_shuffle():
    raw = pending("draw_card")
    law = DeclaredLaw.from_pending(raw)
    assert law.outcome_ids == ("card_instance:7", "card_instance:11")
    router = DrawOnlyRouter(TwoSiteNature(seed=9), SeededLawSampler(12))
    choice = router.choose(raw, fingerprint="parent", episode_key="one", ordinal=2)
    assert choice.operation == "draw_card"
    assert choice.selected_outcome_id in law.outcome_ids
    assert choice.controller == "learned-draw"
    assert choice.nature_choice is not None
    assert choice.nature_choice.operation == "draw_card"
    assert math.isfinite(choice.log_q)
    assert router.reference.calls == 0
    assert choice == router.choose(
        raw, fingerprint="parent", episode_key="one", ordinal=2
    )


@pytest.mark.parametrize("modification", [
    lambda x: x.pop("chanceLaw"),
    lambda x: x["chanceLaw"].update(schemaId="old-schema"),
    lambda x: x["chanceLaw"]["outcomes"][0].update(outcomeId="card_instance:99"),
    lambda x: x["chanceLaw"]["outcomes"][0].update(weight=-1),
    lambda x: x["chanceLaw"].update(totalWeight=3),
    lambda x: x["chanceLaw"].update(eventId="forged"),
    lambda x: x["chanceLaw"].update(conditioningFingerprint="forged"),
    lambda x: x["chanceLaw"].update(probabilityStatus="VERIFIED"),
    lambda x: x["chanceLaw"]["outcomes"][0].update(outcomeId="card_instance:11"),
])
def test_mismatched_draw_law_or_provenance_rejected(modification):
    altered = copy.deepcopy(pending())
    modification(altered)
    with pytest.raises(DrawOnlyContractError):
        DeclaredLaw.from_pending(altered)


@pytest.mark.parametrize("modification", [
    lambda x: x["chanceLaw"]["outcomes"][0].update(outcomeId="not-legal"),
    lambda x: x["chanceLaw"]["outcomes"][0].update(weight=2),
])
def test_intent_chance_law_and_eligibility_disagreement_rejected(modification):
    altered = pending("enemy_intent", weights=(1, 3))
    modification(altered)
    with pytest.raises(DrawOnlyContractError):
        DeclaredLaw.from_pending(altered)


def test_unknown_chance_sites_fail_closed():
    raw = pending()
    raw["kind"] = "random_target"
    with pytest.raises(DrawOnlyContractError, match="Unsupported"):
        DeclaredLaw.from_pending(raw)


def test_versioned_emulator_capability_gate_and_reward_semantics():
    caps = {
        "semanticChanceSchemaId": "sts2-v111-semantic-playable-p0-v1",
        "semanticPlayable": True,
        "semanticIntentSupportSchemaId": "sts2-v111-semantic-weighted-intent-v1",
        "semanticControlledSites": ["enemy_intent", "draw_card"],
        "referenceNatureSessionSchemaId": "sts2-v111-reference-nature-session-v1",
        "semanticSeededRouteEntryProviderId":
            "engine-owned-v111-native-weak-route-entry-unverified-v1",
        "stochasticCapabilities": {"chanceLawSchemaId": CHANCE_LAW_SCHEMA},
    }
    check_draw_only_capabilities(caps)
    wrong = copy.deepcopy(caps)
    wrong["stochasticCapabilities"]["chanceLawSchemaId"] = "old"
    with pytest.raises(DrawOnlyContractError):
        check_draw_only_capabilities(wrong)
    assert _score("defeat", 58, 66, 0) == 0
    assert _score("victory", 58, 66, 48) == pytest.approx(1 - 10 / 66)
