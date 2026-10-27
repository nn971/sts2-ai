"""Isolated Nature protocol, policy and trace invariants."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from sts2_ai.nature.policies import (
    FirstLegalNature,
    LastLegalNature,
    ScriptedNature,
    UniformNature,
    enumerate_finite_choices,
)
from sts2_ai.nature.protocol import ChanceEvent, NatureProposal
from sts2_ai.nature.trace import ChanceTraceEntry, ChanceTranscript


def pending(*, operation="choose_index", ordinal=0, bounded=None):
    return {
        "siteId": "combat.initial_unique_hp",
        "streamId": "run.niche",
        "ownerId": "creature:1",
        "ordinal": ordinal,
        "sourceVersion": "v0.111.0",
        "operationKind": operation,
        "orderedCandidates": [8, 12, 12] if operation == "choose_index" else [],
        "boundedCount": bounded,
    }


def event(*, operation="choose_index", ordinal=0, bounded=None):
    return ChanceEvent.from_isolated_wire(
        pending(operation=operation, ordinal=ordinal, bounded=bounded),
        fingerprint="parent-state-hash",
    )


def test_event_support_identity_and_duplicate_candidate_values():
    e = event()
    assert e.support_count == 3
    e.check_index(2)
    assert e.candidates[1] == e.candidates[2]
    assert e.event_id != event(ordinal=1).event_id
    with pytest.raises(ValueError, match="Illegal"):
        e.check_index(3)
    with pytest.raises(ValueError, match="Illegal"):
        e.check_index(True)
    with pytest.raises(NotImplementedError, match="Unsupported Nature operation"):
        event(operation="shuffle")


def test_compact_bounded_event_and_invalid_forged_metadata():
    e = event(operation="bounded_integer", bounded=100000000)
    assert e.support_count == 100000000
    assert LastLegalNature().propose(e).selected_index == e.support_count - 1
    with pytest.raises(ValueError, match="Invalid bounded-integer"):
        event(operation="bounded_integer", bounded=0)
    raw = pending()
    raw.pop("ownerId")
    with pytest.raises(ValueError, match="Incomplete"):
        ChanceEvent.from_isolated_wire(raw, fingerprint="parent")


def test_uniform_nature_worker_schedule_deterministic():
    e = event()
    policy = UniformNature(seed=37)
    expected = policy.propose(e)
    assert expected.log_q == pytest.approx(-__import__("math").log(3))
    with ThreadPoolExecutor(max_workers=15) as pool:
        result = list(pool.map(lambda _: policy.propose(e), range(150)))
    assert result == [expected] * 150
    assert FirstLegalNature().propose(e).selected_index == 0


def test_scripted_replay_strict_event_order_and_atomic_failure():
    e = event()
    subsequent = event(ordinal=1)
    replay = ScriptedNature.from_selections([
        (e.event_id, 1), (subsequent.event_id, 2),
    ])
    with pytest.raises(ValueError, match="diverged"):
        replay.propose(subsequent)
    assert replay.propose(e).selected_index == 1
    with pytest.raises(ValueError, match="Unconsumed"):
        replay.assert_complete()
    assert replay.propose(subsequent).selected_index == 2
    replay.assert_complete()
    with pytest.raises(ValueError, match="exhausted"):
        replay.propose(e)


def test_bounded_enumeration_and_transcript_roundtrip():
    e = event()
    assert tuple(x.selected_index for x in enumerate_finite_choices(e)) == (0, 1, 2)
    with pytest.raises(ValueError, match="exceeds"):
        enumerate_finite_choices(e, limit=2)
    outcome = NatureProposal(1, "scripted")
    original = ChanceTranscript(
        emulator_revision="rev-123", fixture_id="fixture", nature_policy_id="scripted",
        entries=(ChanceTraceEntry(e, outcome),),
    )
    decoded = ChanceTranscript.from_json(original.to_json())
    assert decoded == original
    assert decoded.fingerprint == original.fingerprint
    assert decoded.replay_script() == ((e.event_id, 1),)
    with pytest.raises(ValueError, match="Duplicated"):
        ChanceTranscript(
            emulator_revision="rev-123", fixture_id="fixture",
            nature_policy_id="scripted",
            entries=(ChanceTraceEntry(e, outcome), ChanceTraceEntry(e, outcome)),
        )
    with pytest.raises(ValueError, match="log proposal"):
        NatureProposal(0, "q", 0.1).validate(e)
