"""Draw-only Nature policy router for the emulator's typed Chance Law.

Nature selects *physical unresolved card instances* only. Every pending
enemy intent is sampled from the emulator-authored pre-outcome law, using a
separate seeded reference RNG; the emulator executes all consequences.
This is a structural emulator-law experiment, not native-game calibration.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.semantic_p0_combat import _pending_event
from sts2_ai.nature.contextual_draw import ContextualDrawNature
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import PilotChoice, TwoSiteNature

CHANCE_LAW_SCHEMA = "sts2-v111-chance-law-v1"
DRAW_ONLY_SCHEMA = "sts2-ai-nature-draw-only-law-router-v1"
REFERENCE_STATUS = "UNVERIFIED"
SOURCE_VERIFIED_STATUS = "SOURCE_VERIFIED"
NATIVE_DOUBLE_GRID = 1 << 53


def _native_intent_grid_weights(raw_weights: tuple[int, ...]) -> tuple[int, ...]:
    """Mirror C# NativeNextIntBoundaryGridIndex with exact integer math.

    The emulator's SOURCE_VERIFIED enemy-intent law assigns 53-bit Native
    NextDouble grid masses, not legacy raw-ticket counts.
    """
    if (not raw_weights or any(type(w) is not int or w <= 0
                               for w in raw_weights)):
        raise DrawOnlyContractError("Invalid source intent raw ticket weights")
    bound = sum(raw_weights)

    def boundary_index(boundary: int) -> int:
        if boundary == 0:
            return 0
        exponent = boundary.bit_length() - 1
        power_of_two = boundary & (boundary - 1) == 0
        shift = 53 - exponent + int(power_of_two)
        scale = 1 << shift
        numerator = (boundary * scale - 1) * NATIVE_DOUBLE_GRID
        denominator = scale * bound
        return (numerator + denominator - 1) // denominator

    results = []
    start = 0
    previous = 0
    for count in raw_weights:
        start += count
        stop = boundary_index(start)
        results.append(stop - previous)
        previous = stop
    if previous != NATIVE_DOUBLE_GRID or any(w <= 0 for w in results):
        raise DrawOnlyContractError("Source intent grid does not cover Native RNG")
    return tuple(results)


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
        status = law.get("probabilityStatus")
        metadata_status = pending.get("probabilityStatus")
        if not (
            status == REFERENCE_STATUS and metadata_status == REFERENCE_STATUS
            or (kind == "enemy_intent"
                and status == SOURCE_VERIFIED_STATUS
                and metadata_status in (REFERENCE_STATUS, SOURCE_VERIFIED_STATUS))
        ):
            raise DrawOnlyContractError(
                "Unexpected native probability provenance: "
                f"kind={kind!r} site={pending.get('siteId')!r} "
                f"pending={metadata_status!r} law={status!r}"
            )
        if (law.get("ownerId") != pending.get("ownerId")
                or law.get("ordinal") != pending.get("ordinal")
                or type(law.get("ordinal")) is not int
                or law["ordinal"] < 0):
            raise DrawOnlyContractError("Chance Law owner/ordinal differs from pending event")
        if law.get("operationKind") != pending.get("operationKind"):
            raise DrawOnlyContractError("Chance Law operation identity changed")
        if law.get("siteId") != pending.get("siteId"):
            raise DrawOnlyContractError("Chance Law site differs from pending request")
        event_id = pending.get("eventId")
        if (not isinstance(event_id, str) or not event_id
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
            if law.get("operationKind") != pending.get("operationKind"):
                raise DrawOnlyContractError("Law operation differs from eligible intents")
            raw_weights = tuple(row.get("rawWeight") for row in candidates)
            if (any(type(w) is not int or w <= 0 for w in raw_weights)
                    or pending.get("totalRawWeight") != sum(raw_weights)):
                raise DrawOnlyContractError("Malformed weighted intent support")
            if status == SOURCE_VERIFIED_STATUS:
                if (law.get("conditioningPolicyId") !=
                        "source-filtered-intent-weights-v1"
                        or tuple(weights) != _native_intent_grid_weights(
                            raw_weights
                        )):
                    raise DrawOnlyContractError(
                        "Source-verified Native intent grid disagrees with support"
                    )
            elif tuple(weights) != raw_weights:
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
                or (self.controller == "learned-draw") !=
                (self.operation == "draw_card" and self.nature_choice is not None)
                or (self.controller == "engine-law-reference"
                    and self.nature_choice is not None)):
            raise DrawOnlyContractError("Wrong Nature/reference site routing")
        if not (math.isfinite(self.log_p) and math.isfinite(self.log_q)):
            raise DrawOnlyContractError("Nonfinite proposal/reference likelihood")


class SeededLawSampler:
    """Independent and replayable declared-law sampler for non-draw sites."""

    def __init__(self, seed: int):
        if type(seed) is not int:
            raise ValueError("Reference RNG seed must be an integer")
        self.seed = seed
        self.calls = 0

    def sample(self, law: DeclaredLaw) -> int:
        """Event-keyed RNG: resolving a draw cannot shift future intent RNG.

        Stable within the same supported event, independent of process
        scheduling, replay order and unrelated stochastic site consumption.
        This couples identical event IDs across arms where possible, without
        claiming native RNG cursor equivalence.
        """
        if law.total_weight <= 0:
            raise DrawOnlyContractError("Empty reference distribution")
        event_key = canonical_sha256([
            DRAW_ONLY_SCHEMA, self.seed, law.site_id, law.event_id,
            law.conditioning_fingerprint, law.outcome_ids, law.weights,
        ])
        ticket = random.Random(int(event_key, 16)).randrange(law.total_weight)
        self.calls += 1
        for index, weight in enumerate(law.weights):
            if ticket < weight:
                return index
            ticket -= weight
        raise DrawOnlyContractError("Failed to sample declared support")


@dataclass
class DrawOnlyRouter:
    nature: TwoSiteNature | ContextualDrawNature
    reference: SeededLawSampler

    def choose(
        self, pending: Mapping[str, Any], *, fingerprint: str,
        episode_key: str, ordinal: int,
        draw_mode: str = "learned",
        public_observation: dict[str, Any] | None = None,
        score_reference_q: bool = False,
    ) -> RoutedChoice:
        if draw_mode not in ("learned", "reference"):
            raise DrawOnlyContractError("Unsupported draw routing mode")
        law = DeclaredLaw.from_pending(pending)
        if law.operation == "enemy_intent" or draw_mode == "reference":
            chosen = self.reference.sample(law)
            q = law.probabilities[chosen]
            # H3 reference-target multiple IS needs the frozen learned
            # Nature probability along *reference* trajectories as well.
            # This is an observation-only query: the chosen transition still
            # follows p; it never adds a Nature learning decision.
            log_behavior_q = math.log(q)
            if score_reference_q and law.operation == "draw_card":
                event, features, ids = _pending_event(pending, fingerprint)
                if tuple(f"card_instance:{value}" for value in ids) != law.outcome_ids:
                    raise DrawOnlyContractError("Counterfactual q support differs from p")
                if any(not math.isclose(prob, 1 / len(ids), abs_tol=1e-12)
                       for prob in law.probabilities):
                    raise DrawOnlyContractError("Counterfactual q requires uniform draw base")
                if not hasattr(self.nature, "propose"):
                    raise DrawOnlyContractError("Nature policy cannot score reference draws")
                _, scored = self.nature.propose(
                    event, context_key=f"{law.site_id}|{event.owner_id}",
                    candidate_features=features,
                    episode_key=f"{episode_key}:{ordinal}",
                    public_observation=public_observation,
                )
                if (len(scored.probabilities) != len(ids)
                        or not 0 < scored.probabilities[chosen] <= 1):
                    raise DrawOnlyContractError("Malformed frozen Nature reference q")
                log_behavior_q = math.log(scored.probabilities[chosen])
            return RoutedChoice(
                operation=law.operation, law_id=law.law_id,
                event_id=law.event_id,
                conditioning_fingerprint=law.conditioning_fingerprint,
                selected_outcome_id=law.outcome_ids[chosen],
                selected_index=chosen,
                log_p=math.log(q), log_q=log_behavior_q,
                controller="engine-law-reference",
            )
        event, features, ids = _pending_event(pending, fingerprint)
        if tuple(f"card_instance:{value}" for value in ids) != law.outcome_ids:
            raise DrawOnlyContractError("AI event adapter disagrees with Chance Law")
        # The pilot's exploratory base is uniform physical-card support.
        # If the emulator declares a different conditional law, upgrade q's
        # base distribution first; otherwise the KL/readout would be false.
        if any(not math.isclose(prob, 1 / len(ids), abs_tol=1e-12)
               for prob in law.probabilities):
            raise DrawOnlyContractError(
                "Nonuniform physical draw law needs an explicit updated Nature base"
            )
        proposal, choice = self.nature.propose(
            event, context_key=f"{law.site_id}|{event.owner_id}",
            candidate_features=features,
            episode_key=f"{episode_key}:{ordinal}",
            public_observation=public_observation,
        )
        if (proposal.selected_id != ids[proposal.selected_index]
                or law.outcome_ids[proposal.selected_index] not in law.outcome_ids):
            raise DrawOnlyContractError("Draw Nature proposed an illegal instance")
        log_q = proposal.log_q
        if log_q is None or not math.isfinite(log_q):
            raise DrawOnlyContractError("Invalid learned Nature log probability")
        q = math.exp(log_q)
        if not 0 < q <= 1:
            raise DrawOnlyContractError("Invalid learned Nature proposal")
        return RoutedChoice(
            operation=law.operation, law_id=law.law_id,
            event_id=law.event_id,
            conditioning_fingerprint=law.conditioning_fingerprint,
            selected_outcome_id=law.outcome_ids[proposal.selected_index],
            selected_index=proposal.selected_index,
            log_p=math.log(law.probabilities[proposal.selected_index]),
            log_q=log_q, controller="learned-draw",
            nature_choice=choice,
        )
