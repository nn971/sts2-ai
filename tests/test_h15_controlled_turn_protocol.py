"""Live H15 end-turn bridge is explicitly unverified and versioned."""
from __future__ import annotations

import pytest

from sts2_ai.emulator.jsonl_backend import JsonlBridgeError, JsonlEmulatorBackend

MODEL = "h15-native-structure-prototype-rng-candidate-unverified-v1"
SCHEMA = "sts2-h15-controlled-prototype-combat-endturn-experimental-v1"


def response(**overrides):
    raw = {
        "combatExitH9Reward": None,
        "schemaId": SCHEMA,
        "fairContinuationCapability": "unavailable",
        "candidateModelId": MODEL,
        "h15Handle": "h15-1",
        "version": 0,
        "boundaryKind": "player",
        "combatExit": False,
        "combatOutcome": None,
        "chanceLaw": None,
        "legalActions": [{
            "actionId": "legal-end-turn",
            "kind": "end_turn",
        }],
    }
    raw.update(overrides)
    return raw


def backend(answer):
    obj = object.__new__(JsonlEmulatorBackend)
    calls = []
    def request(op, **kwargs):
        calls.append((op, kwargs))
        return answer
    obj._request = request
    return obj, calls


def test_begin_from_real_state_requires_explicit_experimental_provenance():
    bridge, calls = backend(response())
    result = bridge.h15_experimental_begin("s123", independent_seed="opaque-1")
    assert result["h15Handle"] == "h15-1"
    assert result["fairContinuationCapability"] == "unavailable"
    assert calls == [(
        "h15_experimental_begin",
        {"state_handle": "s123", "independent_seed": "opaque-1"},
    )]


def test_end_turn_supplies_exact_handle_and_version():
    bridge, calls = backend(response(version=1, boundaryKind="unsupported",
                                     legalActions=None))
    result = bridge.h15_experimental_end_turn("h15-1", 0, "legal-end-turn")
    assert result["boundaryKind"] == "unsupported"
    assert calls == [(
        "h15_experimental_end_turn",
        {"h15_handle": "h15-1", "version": 0,
         "action_id": "legal-end-turn"},
    )]


def test_choose_passes_declared_unverified_law_unchanged():
    law = {
        "probabilityStatus": "UNVERIFIED",
        "lawId": "law-1",
        "outcomes": [{"outcomeId": "card_instance:5", "weight": 1}],
    }
    bridge, calls = backend(response(
        version=2, boundaryKind="draw_card", chanceLaw=law,
        legalActions=None,
    ))
    bridge.h15_experimental_choose(
        "h15-1", 1, law, "card_instance:5",
    )
    assert calls == [(
        "h15_experimental_choose",
        {"h15_handle": "h15-1", "version": 1,
         "chance_law": law, "outcome_id": "card_instance:5"},
    )]


@pytest.mark.parametrize("override", [
    {"fairContinuationCapability": "available"},
    {"candidateModelId": "native-fair-root-v1"},
    {"schemaId": "unsupported"},
    {"boundaryKind": "future"},
    {"version": -1},
    {"h15Handle": "s1"},
    {"combatExit": None},
    {"chanceLaw": {"probabilityStatus": "SOURCE_VERIFIED",
                   "outcomes": [{"outcomeId": "0", "weight": 1}]}},
])
def test_experimental_bridge_refuses_fair_or_malformed_response(override):
    raw = response(boundaryKind="draw_card",
                   chanceLaw={
                       "probabilityStatus": "UNVERIFIED",
                       "outcomes": [{"outcomeId": "0", "weight": 1}],
                   })
    raw.update(override)
    obj, _ = backend(raw)
    with pytest.raises(JsonlBridgeError):
        obj.h15_experimental_begin("s1", independent_seed="independent")


def test_unverified_choice_is_required_client_side():
    obj, calls = backend(response())
    with pytest.raises(ValueError, match="unverified"):
        obj.h15_experimental_choose(
            "h15-1", 0, {"probabilityStatus": "SOURCE_VERIFIED"},
            "choice_index:0",
        )
    assert not calls


@pytest.mark.parametrize("outcome", ["victory", "defeat"])
def test_combat_exit_retains_explicit_result(outcome):
    bridge, _ = backend(response(
        boundaryKind="combat_exit", combatExit=True,
        combatOutcome=outcome, legalActions=None,
        combatExitH9Reward={
            "hp": 20 if outcome == "victory" else 0,
            "maxHp": 70,
            "defeatTotalNonMinionEnemyHp": 30 if outcome == "defeat" else None,
            "defeatRemainingNonMinionEnemyHp": 12 if outcome == "defeat" else None,
        },
    ))
    result = bridge.h15_experimental_status("h15-1")
    assert result["combatOutcome"] == outcome


def test_experimental_boundary_refuses_unannounced_combat_result():
    bridge, _ = backend(response(
        boundaryKind="combat_exit", combatExit=True, legalActions=None,
    ))
    with pytest.raises(JsonlBridgeError, match="provenance"):
        bridge.h15_experimental_status("h15-1")


def test_real_player_action_is_versioned_and_keeps_experimental_provenance():
    bridge, calls = backend(response(version=1))
    result = bridge.h15_experimental_player_action(
        "h15-1", 0, "play-card-action",
    )
    assert result["fairContinuationCapability"] == "unavailable"
    assert calls == [(
        "h15_experimental_player_action",
        {"h15_handle": "h15-1", "version": 0,
         "action_id": "play-card-action"},
    )]


@pytest.mark.parametrize("version,action", [(-1, "play"), (0, ""), (True, "play")])
def test_player_action_invalid_request_never_reaches_bridge(version, action):
    bridge, calls = backend(response())
    with pytest.raises(ValueError, match="player-action"):
        bridge.h15_experimental_player_action(
            "h15-1", version, action,
        )
    assert not calls


def test_experimental_bridge_rejects_terminal_reward_leak_into_combat():
    obj, _ = backend(response(
        combatExitH9Reward={"hp":20,"maxHp":70},
    ))
    with pytest.raises(JsonlBridgeError, match="leaked terminal"):
        obj.h15_experimental_status("h15-1")


def test_experimental_bridge_rejects_missing_exit_hp_accounting():
    obj, _ = backend(response(
        boundaryKind="combat_exit", combatExit=True,
        combatOutcome="defeat", legalActions=None,
    ))
    with pytest.raises(JsonlBridgeError, match="H9 terminal"):
        obj.h15_experimental_status("h15-1")


def test_reference_choice_invokes_atomic_emulator_sampler_with_no_python_law():
    selection = {
        "lawId": "h15-law", "eventId": "h15-event",
        "outcomeId": "card_instance:12", "outcomeIndex": 1,
        "providerId": "reference-nature-splitmix64-v1",
    }
    bridge, calls = backend(response(
        version=3, boundaryKind="player",
        referenceNatureSelection=selection,
        referenceNatureSampleCount=1,
    ))
    received = bridge.h15_experimental_reference_choose("h15-1", 2)
    assert received["referenceNatureSelection"] == selection
    assert calls == [(
        "h15_experimental_reference_choose",
        {"h15_handle": "h15-1", "version": 2},
    )]


def test_reference_choice_rejects_missing_receipt_and_invalid_version():
    bridge, calls = backend(response(version=3))
    with pytest.raises(JsonlBridgeError, match="choice receipt"):
        bridge.h15_experimental_reference_choose("h15-1", 2)
    assert len(calls) == 1

    bridge, calls = backend(response())
    with pytest.raises(ValueError, match="ReferenceNature"):
        bridge.h15_experimental_reference_choose("h15-1", -1)
    assert not calls
