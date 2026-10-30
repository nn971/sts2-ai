"""Diagnostic native candidate bridge must never claim fair search access."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sts2_ai.emulator.jsonl_backend import JsonlBridgeError, JsonlEmulatorBackend


def diagnostic_backend(result):
    backend = object.__new__(JsonlEmulatorBackend)
    requests = []
    def respond(op, **kwargs):
        requests.append((op, kwargs))
        return result
    backend._request = respond
    return backend, requests


def valid_response():
    return {
        "schemaId": "h15-native-combat-entry-independent-candidate-unverified-v1",
        "fairContinuationCapability": "unavailable",
        "stateHandleReturned": False,
        "observationHash": "source-visible-frame-hash",
        "legalActionCount": 3,
        "unknownDrawPositions": 7,
    }


def test_probe_returns_diagnostic_only_and_passes_independent_seed():
    backend, calls = diagnostic_backend(valid_response())
    result = backend.probe_h15_native_combat_candidate(
        "s47", independent_seed="independent-candidate-seed",
    )
    assert calls == [("h15_native_combat_candidate_probe", {
        "state_handle": "s47",
        "independent_seed": "independent-candidate-seed",
    })]
    assert result["fair_continuation_capability"] == "unavailable"
    assert result["unknown_draw_positions"] == 7
    assert "state_handle" not in result
    assert result["observation_hash"] == "source-visible-frame-hash"


@pytest.mark.parametrize("override", [
    {"fairContinuationCapability": "history-conditioned-fair-v1"},
    {"stateHandleReturned": True},
    {"schemaId": "native-fair-root-v1"},
    {"unknownDrawPositions": -1},
    {"legalActionCount": 0},
])
def test_probe_rejects_upgraded_or_malformed_capability_claims(override):
    backend, _ = diagnostic_backend({**valid_response(), **override})
    with pytest.raises(JsonlBridgeError, match="non-fair"):
        backend.probe_h15_native_combat_candidate(
            "s47", independent_seed="independent",
        )


def test_probe_rejects_empty_seed_without_call():
    backend, calls = diagnostic_backend(valid_response())
    with pytest.raises(ValueError, match="independent seed"):
        backend.probe_h15_native_combat_candidate(
            "s47", independent_seed="",
        )
    assert calls == []
