"""Typed, explicitly *experimental* Nature chance-event contracts.

No source-game probability is inferred from a legal support set. Player
observations are deliberately absent from this module: chance metadata is
available only to Nature, never implicitly to the player policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

NATURE_EVENT_SCHEMA = "sts2-ai-nature-event-v1"
ISOLATED_EMULATOR_SCHEMA = "sts2-v111-nature-isolated-boundary-v1"
SUPPORTED_OPERATIONS = frozenset({"choose_index", "bounded_integer"})


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ChanceEvent:
    """A single finite, legal source-index decision at a pending boundary."""

    boundary_fingerprint: str
    site_id: str
    stream_id: str
    owner_id: str
    ordinal: int
    operation: str
    source_version: str
    candidates: tuple[int, ...] = ()
    bounded_count: int | None = None
    probability_status: str = "UNVERIFIED"
    schema: str = NATURE_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NATURE_EVENT_SCHEMA or not self.boundary_fingerprint:
            raise ValueError("Missing or unsupported Nature event identity")
        if not self.site_id or not self.stream_id or not self.owner_id:
            raise ValueError("Nature event requires site/stream/owner")
        if self.source_version != "v0.111.0":
            raise ValueError("Unsupported Nature game source version")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("Invalid Nature event ordinal")
        if self.operation not in SUPPORTED_OPERATIONS:
            raise NotImplementedError(f"Unsupported Nature operation: {self.operation}")
        if self.probability_status != "UNVERIFIED":
            raise ValueError("Unverified emulator events cannot advertise calibrated p")
        if self.operation == "choose_index":
            if (not self.candidates or self.bounded_count is not None or
                    any(type(x) is not int for x in self.candidates)):
                raise ValueError("Invalid ordered candidate support")
        elif self.candidates or type(self.bounded_count) is not int or self.bounded_count <= 0:
            raise ValueError("Invalid bounded-integer support")

    @property
    def support_count(self) -> int:
        return len(self.candidates) if self.operation == "choose_index" else self.bounded_count  # type: ignore[return-value]

    @property
    def event_id(self) -> str:
        return canonical_sha256([
            self.schema, self.boundary_fingerprint, self.site_id,
            self.stream_id, self.owner_id, self.ordinal, self.operation,
            self.source_version, self.candidates, self.bounded_count,
        ])

    def check_index(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < self.support_count:
            raise ValueError(f"Illegal Nature index {index!r} for {self.event_id}")

    @classmethod
    def from_isolated_wire(
        cls, pending: dict, *, fingerprint: str,
    ) -> ChanceEvent:
        """Decode actual v111 C# JSON camelCase; do not invent probabilities."""
        required = ("siteId", "streamId", "ownerId", "ordinal",
                    "operationKind", "sourceVersion", "orderedCandidates")
        if any(key not in pending for key in required):
            raise ValueError("Incomplete emulator pending-chance request")
        candidates = pending["orderedCandidates"]
        if not isinstance(candidates, list):
            raise ValueError("Pending candidates must be a list")
        return cls(
            boundary_fingerprint=fingerprint,
            site_id=pending["siteId"], stream_id=pending["streamId"],
            owner_id=pending["ownerId"], ordinal=pending["ordinal"],
            operation=pending["operationKind"],
            source_version=pending["sourceVersion"],
            candidates=tuple(candidates),
            bounded_count=pending.get("boundedCount"),
        )


@dataclass(frozen=True, slots=True)
class NatureProposal:
    """One legal Nature outcome; log_q may be absent for scripted intervention."""

    selected_index: int
    proposal_id: str
    log_q: float | None = None

    def validate(self, event: ChanceEvent) -> None:
        event.check_index(self.selected_index)
        if not self.proposal_id:
            raise ValueError("Missing immutable Nature policy identity")
        if self.log_q is not None and (
            not math.isfinite(self.log_q) or self.log_q > 1e-12
        ):
            raise ValueError("Invalid Nature log proposal probability")


class NaturePolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def propose(self, event: ChanceEvent) -> NatureProposal: ...
