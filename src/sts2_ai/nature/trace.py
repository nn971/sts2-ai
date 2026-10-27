"""Immutable chance-only transcripts for the initial isolated Nature bridge.

These records are intentionally distinct from player-visible histories. A
future playable protocol must interleave player actions and chance decisions
without leaking simulator-only state into policy features.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from .protocol import ChanceEvent, NatureProposal, canonical_sha256

TRACE_SCHEMA = "sts2-ai-nature-isolated-trace-v1"


@dataclass(frozen=True, slots=True)
class ChanceTraceEntry:
    event: ChanceEvent
    outcome: NatureProposal

    def __post_init__(self) -> None:
        self.outcome.validate(self.event)


@dataclass(frozen=True, slots=True)
class ChanceTranscript:
    emulator_revision: str
    fixture_id: str
    nature_policy_id: str
    entries: tuple[ChanceTraceEntry, ...]
    schema: str = TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TRACE_SCHEMA:
            raise ValueError("Unsupported chance transcript schema")
        if not self.emulator_revision or not self.fixture_id or not self.nature_policy_id:
            raise ValueError("Incomplete Nature transcript provenance")
        ids = [entry.event.event_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicated pending event identity in one Nature transcript")
        if any(e.outcome.proposal_id != self.nature_policy_id for e in self.entries):
            raise ValueError("Nature proposal identity changed within transcript")

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(asdict(self))

    def replay_script(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (row.event.event_id, row.outcome.selected_index)
            for row in self.entries
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> ChanceTranscript:
        raw = json.loads(payload)
        if not isinstance(raw, dict) or raw.get("schema") != TRACE_SCHEMA:
            raise ValueError("Unrecognized Nature transcript")
        entries = []
        for row in raw["entries"]:
            data = dict(row["event"])
            data["candidates"] = tuple(data["candidates"])
            event = ChanceEvent(**data)
            outcome = NatureProposal(**row["outcome"])
            entries.append(ChanceTraceEntry(event, outcome))
        return cls(
            emulator_revision=raw["emulator_revision"],
            fixture_id=raw["fixture_id"],
            nature_policy_id=raw["nature_policy_id"],
            entries=tuple(entries),
        )


def record_entry(event: ChanceEvent, proposal: NatureProposal) -> ChanceTraceEntry:
    if proposal.log_q is not None and not math.isfinite(proposal.log_q):
        raise ValueError("Nonfinite Nature probability")
    return ChanceTraceEntry(event, proposal)
