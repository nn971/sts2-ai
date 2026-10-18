"""Strict *proposed* semantic Nature event contract.

The v111 semantic deck provider-state kernel exists in sts2-emulator, but has
no gameplay/JSONL interface yet. These types specify an opt-in future wire
contract and can be exercised with synthetic adapters today. They are kept
separate from the legacy primitive-index ChanceEvent schema.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .protocol import canonical_sha256

SEMANTIC_SCHEMA = "sts2-ai-semantic-event-v1"
PROPOSED_WIRE_SCHEMA = "sts2-v111-semantic-chance-v1"
SEMANTIC_REPRESENTATION = "nature-deferred-v1"
SEMANTIC_OPERATIONS = frozenset({
    "draw_card", "reveal_card", "enemy_intent",
    "select_distinct", "generate_offer",
})
DECK_OPERATIONS = frozenset({"draw_card", "reveal_card"})
CHANCE_LAW_STATUSES = frozenset({"UNVERIFIED", "AUDITED_NATIVE"})


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    """A finite semantic outcome supplied by the authoritative emulator.

    `event_id`, `support_fingerprint` and `constraint_fingerprint` are
    emulator-authored and independently checked on replay, not synthesized
    by the AI from hidden state. Candidate IDs refer to distinct physical
    card instances for draw/reveal, and to stable intents for enemy_intent.
    """

    event_id: str
    boundary_fingerprint: str
    site_id: str
    owner_id: str
    ordinal: int
    operation: str
    candidate_ids: tuple[str, ...]
    support_fingerprint: str
    randomization_id: str
    constraint_fingerprint: str
    source_version: str = "v0.111.0"
    representation: str = SEMANTIC_REPRESENTATION
    probability_status: str = "UNVERIFIED"
    reference_law_id: str | None = None
    raw_weights: tuple[float, ...] | None = None
    weight_total: float | None = None
    schema: str = SEMANTIC_SCHEMA
    candidate_features: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("event_id", self.event_id),
            ("boundary_fingerprint", self.boundary_fingerprint),
            ("site_id", self.site_id),
            ("owner_id", self.owner_id),
            ("support_fingerprint", self.support_fingerprint),
            ("randomization_id", self.randomization_id),
            ("constraint_fingerprint", self.constraint_fingerprint),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"Missing semantic Nature {label}")
        if self.schema != SEMANTIC_SCHEMA or self.source_version != "v0.111.0":
            raise ValueError("Unsupported semantic game/event schema")
        if self.representation != SEMANTIC_REPRESENTATION:
            raise ValueError("Semantic Nature cannot consume legacy swap tickets")
        if self.operation not in SEMANTIC_OPERATIONS:
            raise NotImplementedError(f"Unsupported semantic operation: {self.operation}")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("Invalid semantic event ordinal")
        if (not self.candidate_ids
                or any(not isinstance(x, str) or not x for x in self.candidate_ids)
                or len(set(self.candidate_ids)) != len(self.candidate_ids)):
            raise ValueError("Semantic candidates must be unique stable identities")
        if self.candidate_features is not None and (
            len(self.candidate_features) != len(self.candidate_ids)
            or any(not isinstance(x, str) or not x
                   for x in self.candidate_features)
        ):
            raise ValueError("Semantic candidate features must match the support")
        if self.probability_status not in CHANCE_LAW_STATUSES:
            raise ValueError("Unsupported semantic probability status")
        if self.probability_status == "UNVERIFIED" and self.reference_law_id is not None:
            raise ValueError("Unverified semantic weights cannot claim a reference law")
        if self.probability_status == "AUDITED_NATIVE" and not self.reference_law_id:
            raise ValueError("Audited chance weights require a source-law identity")
        if self.operation == "enemy_intent":
            if self.raw_weights is None or self.weight_total is None:
                raise ValueError("Enemy intent requires current source eligibility weights")
            if (len(self.raw_weights) != len(self.candidate_ids)
                    or any(type(w) not in (float, int) or
                           not math.isfinite(w) or w < 0 for w in self.raw_weights)
                    or not math.isfinite(self.weight_total)
                    or self.weight_total <= 0
                    or not math.isclose(
                        math.fsum(self.raw_weights), self.weight_total,
                        abs_tol=1e-9, rel_tol=1e-9,
                    )):
                raise ValueError("Invalid enemy-intent weights/normalization")
            if not any(w > 0 for w in self.raw_weights):
                raise ValueError("No eligible positive-weight intent")
        elif self.raw_weights is not None or self.weight_total is not None:
            raise ValueError("Unspecified weights on non-intent semantic event")

    @property
    def support_count(self) -> int:
        return len(self.candidate_ids)

    def check_index(self, index: int) -> str:
        if type(index) is not int or not 0 <= index < self.support_count:
            raise ValueError("Illegal semantic Nature outcome index")
        if self.operation == "enemy_intent":
            assert self.raw_weights is not None
            if self.raw_weights[index] <= 0:
                raise ValueError("Zero-weight enemy intent is ineligible")
        return self.candidate_ids[index]

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> SemanticEvent:
        """Decode proposed camelCase semantics; reject primitive RNG fallbacks."""
        required = (
            "eventId", "boundaryFingerprint", "siteId", "ownerId",
            "ordinal", "operation", "candidateIds", "supportFingerprint",
            "randomizationId", "constraintFingerprint",
            "sourceVersion", "representation", "probabilityStatus",
        )
        if any(k not in data for k in required):
            raise ValueError("Incomplete semantic Nature event")
        candidates = data["candidateIds"]
        weights = data.get("rawWeights")
        if not isinstance(candidates, list) or (weights is not None and
                                                not isinstance(weights, list)):
            raise ValueError("Malformed semantic support or weights")
        return cls(
            event_id=data["eventId"],
            boundary_fingerprint=data["boundaryFingerprint"],
            site_id=data["siteId"], owner_id=data["ownerId"],
            ordinal=data["ordinal"], operation=data["operation"],
            candidate_ids=tuple(candidates),
            support_fingerprint=data["supportFingerprint"],
            randomization_id=data["randomizationId"],
            constraint_fingerprint=data["constraintFingerprint"],
            source_version=data["sourceVersion"],
            representation=data["representation"],
            probability_status=data["probabilityStatus"],
            reference_law_id=data.get("referenceLawId"),
            raw_weights=tuple(weights) if weights is not None else None,
            weight_total=data.get("weightTotal"),
            candidate_features=(
                tuple(data["candidateFeatures"])
                if isinstance(data.get("candidateFeatures"), list) else None
            ),
        )

    @property
    def replay_identity(self) -> str:
        """Hash the complete public-to-Nature support contract, not game state."""
        identity: list[object] = [
            self.schema, self.event_id, self.boundary_fingerprint,
            self.site_id, self.owner_id, self.ordinal, self.operation,
            self.candidate_ids, self.support_fingerprint,
            self.randomization_id, self.constraint_fingerprint,
            self.representation, self.source_version,
            self.raw_weights, self.weight_total,
            self.probability_status, self.reference_law_id,
        ]
        if self.candidate_features is not None:
            identity.append(("candidate_features", self.candidate_features))
        return canonical_sha256(identity)

    @property
    def normalized_intent_weights(self) -> tuple[float, ...]:
        """Proposal inputs only until the independently audited N5 law gate."""
        if self.operation != "enemy_intent":
            raise ValueError("Event does not describe enemy intent")
        assert self.raw_weights is not None
        assert self.weight_total is not None
        return tuple(w / self.weight_total for w in self.raw_weights)


@dataclass(frozen=True, slots=True)
class SemanticProposal:
    selected_index: int
    selected_id: str
    proposal_id: str
    log_q: float | None = None

    def validate(self, event: SemanticEvent) -> None:
        if self.selected_id != event.check_index(self.selected_index):
            raise ValueError("Semantic Nature selected identity/index mismatch")
        if not self.proposal_id:
            raise ValueError("Missing semantic Nature policy identity")
        if self.log_q is not None and (
            not isinstance(self.log_q, (int, float)) or
            not math.isfinite(self.log_q) or self.log_q > 1e-12
        ):
            raise ValueError("Invalid semantic proposal log-q")
