"""Draw-only Nature policy router for the emulator's typed Chance Law.

Nature selects *physical unresolved card instances* only. Every pending
enemy intent is sampled from the emulator-authored pre-outcome law, using a
separate seeded reference RNG; the emulator executes all consequences.
This is a structural emulator-law experiment, not native-game calibration.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping

from sts2_ai.emulator.semantic_p0_combat import _pending_event
from sts2_ai.nature.two_site_opponent import PilotChoice, TwoSiteNature

CHANCE_LAW_SCHEMA = "sts2-v111-chance-law-v1"
DRAW_ONLY_SCHEMA = "sts2-ai-nature-draw-only-law-router-v1"
REFERENCE_STATUS = "UNVERIFIED"


class DrawOnlyContractError(ValueError):
    """Reject incompatible chance-law metadata or unsupported boundaries."""


@dataclass(frozen=True)
class DeclaredLaw:
    law_id: str
    event_id: str
    site_id: str
    support_fingerprint: str
    conditioning_policy_id: str
    conditioning_fingerprint: str
    operation: str
    outcome_ids: tuple[str, ...]
    weights: tuple[int, ...]
    total_weight: int
    provenance: str

    @property
    def probabilities(self) -> tuple[float, ...]:
        return tuple(weight / self.total_weight for weight in self.weights)

    @staticmethod
    def _valid_identifier(value: object) -> bool:
        return isinstance(value, str) and bool(value)

    @classmethod
    def from_pending(cls, pending: Mapping[str, Any]) -> DeclaredLaw:
        kind = pending.get("kind")
        if kind not in ("enemy_intent", "draw_card"):
            raise DrawOnlyContractError(f"Unsupported draw-only boundary: {kind!r}")
        law = pending.get("chanceLaw")
        if not isinstance(law, dict) or law.get("schemaId") != CHANCE_LAW_SCHEMA:
            raise DrawOnlyContractError("Emulator Chance Law schema absent or changed")
        if (law.get("probabilityStatus") != REFERENCE_STATUS
                or pending.get("probabilityStatus") != REFERENCE_STATUS):
            raise DrawOnlyContractError("Unexpected native probability provenance")
        if law.get("siteId") != pending.get("siteId"):
            raise DrawOnlyContractError("Chance Law site differs from pending request")
        event_id = pending.get("eventId")
        if (not cls._valid_identifier(event_id)
                or law.get("eventId") != event_id):
            raise DrawOnlyContractError("Chance Law event identity mismatch")
        for name in ("lawId", "supportFingerprint", "conditioningPolicyId",
                     "conditioningFingerprint", "probabilityProvenance"):
            if not cls._valid_identifier(law.get(name)):
                raise DrawOnlyContractError(f"Missing Chance Law field {name}")
        outcomes = law.get("outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise DrawOnlyContractError("Chance Law has no ordered outcomes")
        ids: list[str] = []
        weights: list[int] = []
        for row in outcomes:
            if (not isinstance(row, dict)
                    or not cls._valid_identifier(row.get("outcomeId"))
                    or type(row.get("weight")) is not int or row["weight"] <= 0):
                raise DrawOnlyContractError("Malformed Chance Law outcome/weight")
            ids.append(row["outcomeId"])
            weights.append(row["weight"])
        if len(set(ids)) != len(ids):
            raise DrawOnlyContractError("Duplicate physical/intent outcome identity")
        total = law.get("totalWeight")
        if type(total) is not int or total != sum(weights):
            raise DrawOnlyContractError("Chance Law total weight mismatch")
        if kind == "enemy_intent":
            candidates = pending.get("semanticCandidates")
            if (not isinstance(candidates, list) or not candidates
                    or [row.get("intentId") for row in candidates] != ids):
                raise DrawOnlyContractError("Law intents differ from legal move IDs")
            if (law.get("operationKind") != pending.get("operationKind")
                    or any(row.get("rawWeight") != weight
                           for row, weight in zip(candidates, weights, strict=True))):
                raise DrawOnlyContractError("Law weights differ from eligible intents")
        else:
            candidates = pending.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise DrawOnlyContractError("Missing physical card-instance support")
            try:
                expected = [
                    f"card_instance:{row['cardInstanceId']}"
                    for row in candidates
                    if type(row["cardInstanceId"]) is int and row["cardInstanceId"] >= 0
                ]
            except (KeyError, TypeError) as exc:
                raise DrawOnlyContractError("Malformed physical-instance candidates") from exc
            if (expected != ids or len(expected) != len(candidates)
                    or law.get("conditioningFingerprint") !=
                    pending.get("constraintFingerprint")
                    or law.get("operationKind") != pending.get("operationKind")):
                raise DrawOnlyContractError("Law violates physical-deck constraints")
        return cls(
            law_id=law["lawId"], event_id=event_id,
            site_id=law["siteId"], support_fingerprint=law["supportFingerprint"],
            conditioning_policy_id=law["conditioningPolicyId"],
            conditioning_fingerprint=law["conditioningFingerprint"],
            operation=kind, outcome_ids=tuple(ids),
            weights=tuple(weights), total_weight=total,
            provenance=law["probabilityProvenance"],
        )


@dataclass(frozen=True)
class RoutedChoice:
    operation: str
    law_id: str
    event_id: str
    conditioning_fingerprint: str
    selected_outcome_id: str
    selected_index: int
    log_p: float
    log_q: float
    controller: str
    nature_choice: PilotChoice | None = None

    def __post_init__(self) -> None:
        if (self.controller not in ("engine-law-reference", "learned-draw")
                or (self.operation == "draw_card") !=
                (self.controller == "learned-draw")
                or (self.controller == "learned-draw") !=
                (self.nature_choice is not None)):
            raise DrawOnlyContractError("Wrong Nature/reference site routing")
        if not (math.isfinite(self.log_p) and math.isfinite(self.log_q)):
            raise DrawOnlyContractError("Nonfinite proposal/reference likelihood")


class SeededLawSampler:
    """Independent and replayable declared-law sampler for non-draw sites."""

    def __init__(self, seed: int):
        if type(seed) is not int:
            raise ValueError("Reference RNG seed must be an integer")
        self.seed = seed
        self._random = random.Random(seed)
        self.calls = 0

    def sample(self, law: DeclaredLaw) -> int:
        if law.total_weight <= 0:
            raise DrawOnlyContractError("Empty reference distribution")
        ticket = self._random.randrange(law.total_weight)
        self.calls += 1
        for index, weight in enumerate(law.weights):
            if ticket < weight:
                return index
            ticket -= weight
        raise DrawOnlyContractError("Failed to sample declared support")


@dataclass
class DrawOnlyRouter:
    nature: TwoSiteNature
    reference: SeededLawSampler

    def choose(
        self, pending: Mapping[str, Any], *, fingerprint: str,
        episode_key: str, ordinal: int,
    ) -> RoutedChoice:
        law = DeclaredLaw.from_pending(pending)
        if law.operation == "enemy_intent":
            chosen = self.reference.sample(law)
            q = law.probabilities[chosen]
            return RoutedChoice(
                operation=law.operation, law_id=law.law_id,
                event_id=law.event_id,
                conditioning_fingerprint=law.conditioning_fingerprint,
                selected_outcome_id=law.outcome_ids[chosen],
                selected_index=chosen,
                log_p=math.log(q), log_q=math.log(q),
                controller="engine-law-reference",
            )
        event, features, ids = _pending_event(pending, fingerprint)
        if tuple(f"card_instance:{value}" for value in ids) != law.outcome_ids:
            raise DrawOnlyContractError("AI event adapter disagrees with Chance Law")
        proposal, choice = self.nature.propose(
            event, context_key=f"{law.site_id}|{event.owner_id}",
            candidate_features=features,
            episode_key=f"{episode_key}:{ordinal}",
        )
        if (proposal.selected_id != ids[proposal.selected_index]
                or law.outcome_ids[proposal.selected_index] not in law.outcome_ids):
            raise DrawOnlyContractError("Draw Nature proposed an illegal instance")
        q = math.exp(proposal.log_q)
        if not 0 < q <= 1:
            raise DrawOnlyContractError("Invalid learned Nature proposal")
        return RoutedChoice(
            operation=law.operation, law_id=law.law_id,
            event_id=law.event_id,
            conditioning_fingerprint=law.conditioning_fingerprint,
            selected_outcome_id=law.outcome_ids[proposal.selected_index],
            selected_index=proposal.selected_index,
            log_p=math.log(law.probabilities[proposal.selected_index]),
            log_q=proposal.log_q, controller="learned-draw",
            nature_choice=choice,
        )
