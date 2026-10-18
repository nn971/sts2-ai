"""Opt-in semantic Nature adapters.

``SemanticNatureBridge`` preserves the original proposed-wire contract for
synthetic compatibility. ``PlayableSemanticCombatBridge`` consumes the
emulator's current seeded semantic combat API. Both leave deck constraints
and game mechanics with the emulator.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sts2_ai.nature.semantic import (
    PROPOSED_WIRE_SCHEMA,
    SemanticEvent,
    SemanticProposal,
)

FAIR_POLICY_ID = "prototype-fair-v0"
WIRE_SCHEMA = "prototype-ai-jsonl-v0"


class SemanticCapabilityUnavailable(RuntimeError):
    """The emulator does not yet expose the new semantic chance protocol."""


class SemanticWire(Protocol):
    def call(self, operation: str, **payload: Any) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SemanticBoundary:
    """Player, Nature or terminal boundary; private chance view is isolated."""

    handle: str
    phase: str
    fingerprint: str
    event: SemanticEvent | None
    observation: dict[str, Any] | None
    legal_actions: tuple[str, ...]
    terminal: str | None
    combat: Any | None = None
    exit_hp: int | None = None

    @classmethod
    def decode(cls, result: Mapping[str, Any]) -> SemanticBoundary:
        if result.get("semanticChanceSchemaId") != PROPOSED_WIRE_SCHEMA:
            raise SemanticCapabilityUnavailable("Semantic frame schema mismatch")
        phase = result.get("phase")
        if phase not in ("chance", "player", "terminal"):
            raise SemanticCapabilityUnavailable("Unexpected semantic boundary phase")
        if not result.get("semanticHandle") or not result.get("fingerprint"):
            raise SemanticCapabilityUnavailable("Incomplete semantic state provenance")
        pending = result.get("pending")
        obs = result.get("observation")
        actions = result.get("legalActions")
        terminal = result.get("terminal")
        if phase == "chance":
            if pending is None or obs is not None or actions is not None or terminal is not None:
                raise SemanticCapabilityUnavailable("Chance state leaked public player fields")
            event = SemanticEvent.decode(pending)
            if event.boundary_fingerprint != result["fingerprint"]:
                raise SemanticCapabilityUnavailable("Pending semantic state fingerprint drift")
            return cls(result["semanticHandle"], phase, result["fingerprint"],
                       event, None, (), None)
        if pending is not None:
            raise SemanticCapabilityUnavailable("Pending Nature event at player/terminal")
        if phase == "player":
            if (result.get("fairPolicyId") != FAIR_POLICY_ID or
                    not isinstance(obs, dict) or
                    not isinstance(actions, list) or
                    not actions or any(not isinstance(a, str) or not a for a in actions)
                    or len(set(actions)) != len(actions) or terminal is not None):
                raise SemanticCapabilityUnavailable("Player boundary fairness/menu mismatch")
            return cls(result["semanticHandle"], phase, result["fingerprint"],
                       None, dict(obs), tuple(actions), None)
        if obs is not None or actions is not None or not isinstance(terminal, str):
            raise SemanticCapabilityUnavailable("Invalid terminal semantic boundary")
        return cls(result["semanticHandle"], phase, result["fingerprint"],
                   None, None, (), terminal)


class SemanticNatureBridge:
    """No native RNG reimplementation, and no legacy swap-ticket reinterpretation."""

    def __init__(
        self, wire: SemanticWire, *, emulator_revision: str,
        required_operations: frozenset[str] = frozenset({
            "draw_card", "reveal_card", "enemy_intent",
        }),
    ):
        if not emulator_revision:
            raise ValueError("Explicit emulator SHA is required for Nature provenance")
        hello = wire.call("hello")
        if hello.get("wireSchemaId") != WIRE_SCHEMA:
            raise SemanticCapabilityUnavailable("Unexpected emulator wire schema")
        if hello.get("semanticChanceSchemaId") != PROPOSED_WIRE_SCHEMA:
            raise SemanticCapabilityUnavailable(
                "Semantic gameplay capability unavailable; pure provider kernel "
                "or legacy controlled shuffle is insufficient"
            )
        if hello.get("semanticChanceRepresentationId") != "nature-deferred-v1":
            raise SemanticCapabilityUnavailable("No deferred-card Nature representation")
        advertised = hello.get("semanticChanceOperations")
        supported = {
            "draw_card", "reveal_card", "enemy_intent",
            "select_distinct", "generate_offer",
        }
        if (not isinstance(advertised, list) or not advertised
                or len(set(advertised)) != len(advertised)
                or not set(advertised) <= supported
                or not required_operations <= set(advertised)):
            raise SemanticCapabilityUnavailable("Semantic operation contract mismatch")
        self.operations = frozenset(advertised)
        if hello.get("fairPolicyId") != FAIR_POLICY_ID:
            raise SemanticCapabilityUnavailable("Player policy fairness contract changed")
        self.wire = wire
        self.emulator_revision = emulator_revision

    def _decode(self, response: Mapping[str, Any]) -> SemanticBoundary:
        frame = SemanticBoundary.decode(response)
        if frame.event is not None and frame.event.operation not in self.operations:
            raise SemanticCapabilityUnavailable("Unexpected unadvertised semantic event")
        return frame

    def begin(self, *, fixture_id: str, seed: str) -> SemanticBoundary:
        if not fixture_id or not seed:
            raise ValueError("Semantic fixture provenance missing")
        return self._decode(
            self.wire.call("semantic_begin", fixture_id=fixture_id, seed=seed)
        )

    def resolve(
        self, frame: SemanticBoundary, proposal: SemanticProposal,
    ) -> SemanticBoundary:
        if frame.phase != "chance" or frame.event is None:
            raise ValueError("Nature may only resolve a pending semantic chance")
        proposal.validate(frame.event)
        result = self.wire.call(
            "semantic_resolve", semantic_handle=frame.handle,
            fingerprint=frame.fingerprint, event_id=frame.event.event_id,
            support_fingerprint=frame.event.support_fingerprint,
            constraint_fingerprint=frame.event.constraint_fingerprint,
            selected_index=proposal.selected_index,
            selected_id=proposal.selected_id,
        )
        return self._decode(result)

    def step(self, frame: SemanticBoundary, action_id: str) -> SemanticBoundary:
        if frame.phase != "player" or action_id not in frame.legal_actions:
            raise ValueError("Illegal semantic player action or wrong decision phase")
        return self._decode(
            self.wire.call("semantic_step", semantic_handle=frame.handle,
                           fingerprint=frame.fingerprint, action_id=action_id)
        )

    def fork(self, frame: SemanticBoundary) -> SemanticBoundary:
        child = self._decode(
            self.wire.call("semantic_fork", semantic_handle=frame.handle)
        )
        if child.fingerprint != frame.fingerprint or child.phase != frame.phase:
            raise SemanticCapabilityUnavailable("Semantic fork modified gameplay state")
        if (child.event is not None and frame.event is not None
                and child.event.replay_identity != frame.event.replay_identity):
            raise SemanticCapabilityUnavailable("Semantic fork modified support/constraints")
        return child

    def save(self, frame: SemanticBoundary) -> str:
        result = self.wire.call("semantic_save", semantic_handle=frame.handle)
        if result.get("semanticChanceSchemaId") != PROPOSED_WIRE_SCHEMA:
            raise SemanticCapabilityUnavailable("Semantic snapshot schema mismatch")
        snapshot = result.get("snapshot")
        if not isinstance(snapshot, str) or not snapshot:
            raise SemanticCapabilityUnavailable("Missing semantic snapshot")
        return snapshot

    def load(self, snapshot: str) -> SemanticBoundary:
        if not snapshot:
            raise ValueError("Semantic snapshot required")
        return self._decode(
            self.wire.call("semantic_load", snapshot=snapshot)
        )

    def release(self, *frames: SemanticBoundary) -> int:
        result = self.wire.call(
            "semantic_release_many", semantic_handles=[x.handle for x in frames]
        )
        return result["released"]


class PlayableSemanticCombatBridge:
    """Adapter for the emulator's current, seeded-source-route combat API.

    It leaves the existing proposed-wire bridge intact for old synthetic
    contract tests. The real adapter uses the bound ``Session`` API, obtains
    Nature support through its versioned Nature-only view, and gives player
    policies only ``SemanticCombat.observe()`` plus legal action IDs.
    """

    CHANCE_SCHEMA = "sts2-v111-semantic-playable-p0-v1"
    NATURE_VIEW_SCHEMA = "sts2-v111-semantic-nature-choice-view-v2"
    ROUTE_PROVIDER = "engine-owned-v111-native-weak-route-entry-unverified-v1"

    def __init__(
        self,
        session: Any,
        *,
        emulator_revision: str,
        region: str = "Overgrowth",
        ascension: int = 0,
        neow_choice_id: str | None = None,
        first_map_node_id: str | None = None,
    ) -> None:
        if not emulator_revision:
            raise ValueError("Explicit emulator SHA is required for Nature provenance")
        if not region or type(ascension) is not int or ascension < 0:
            raise ValueError("Invalid source-route configuration")
        capabilities = session.capabilities
        required_sites = {"enemy_intent", "draw_card"}
        advertised_sites = capabilities.get("semanticControlledSites")
        view_operations = capabilities.get("semanticNatureChoiceViewOperations")
        view_session_kinds = capabilities.get("semanticNatureChoiceViewSessionKinds")
        if (
            capabilities.get("wireSchemaId") != WIRE_SCHEMA
            or capabilities.get("semanticChanceSchemaId") != self.CHANCE_SCHEMA
            or capabilities.get("semanticPlayable") is not True
            or capabilities.get("semanticSeededRouteEntryProviderId")
            != self.ROUTE_PROVIDER
            or capabilities.get("fairPolicyId") != FAIR_POLICY_ID
            or not isinstance(advertised_sites, list)
            or not required_sites <= set(advertised_sites)
            or capabilities.get("semanticNatureChoiceViewSchemaId")
            != self.NATURE_VIEW_SCHEMA
            or not isinstance(view_operations, list)
            or "nature_choice_observe" not in view_operations
            or not isinstance(view_session_kinds, list)
            or "semantic_combat" not in view_session_kinds
        ):
            raise SemanticCapabilityUnavailable(
                "Emulator lacks the current seeded semantic-combat contract"
            )

        self.session = session
        self.emulator_revision = emulator_revision
        self.chance_schema_id = self.CHANCE_SCHEMA
        self.region = region
        self.ascension = ascension
        self.neow_choice_id = neow_choice_id
        self.first_map_node_id = first_map_node_id
        self.fixture_id = "source-route:" + json.dumps(
            {
                "region": region,
                "ascension": ascension,
                "neowChoiceId": neow_choice_id,
                "firstMapNodeId": first_map_node_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def begin(self, *, fixture_id: str, seed: str) -> SemanticBoundary:
        if fixture_id != self.fixture_id:
            raise ValueError(
                "fixture_id must identify this bridge's configured source route"
            )
        if not isinstance(seed, str) or not seed:
            raise ValueError("Semantic source-route seed is required")
        state = self.session.begin_seeded_semantic_route(
            seed,
            region=self.region,
            ascension=self.ascension,
            neow_choice_id=self.neow_choice_id,
            first_map_node_id=self.first_map_node_id,
        )
        return self._decode(state)

    def _decode_event(
        self, frame: Any, view: Mapping[str, Any],
    ) -> SemanticEvent:
        if (
            view.get("schemaId") != self.NATURE_VIEW_SCHEMA
            or not isinstance(view.get("eventMetadata"), dict)
            or not isinstance(view.get("chanceLaw"), dict)
        ):
            raise SemanticCapabilityUnavailable("Malformed Nature-choice view")
        metadata = view["eventMetadata"]
        law = view["chanceLaw"]
        operation = view.get("eventKind")
        event_id = view.get("eventId")
        site_id = view.get("siteId")
        owner_id = law.get("ownerId")
        ordinal = law.get("ordinal")
        law_outcomes = law.get("outcomes")
        if (
            not isinstance(operation, str)
            or operation not in {"enemy_intent", "draw_card"}
            or metadata.get("kind") != operation
            or not isinstance(event_id, str)
            or not isinstance(site_id, str)
            or not isinstance(owner_id, str)
            or type(ordinal) is not int
            or metadata.get("eventId") != event_id
            or metadata.get("siteId") != site_id
            or metadata.get("ownerId") != owner_id
            or metadata.get("ordinal") != ordinal
            or law.get("eventId") != event_id
            or law.get("siteId") != site_id
            or not isinstance(law_outcomes, list)
            or law.get("probabilityStatus") not in {
                "UNVERIFIED", "SOURCE_VERIFIED"
            }
        ):
            raise SemanticCapabilityUnavailable(
                "Unsupported or unverified semantic combat event"
            )

        candidate_features: tuple[str, ...]
        raw_weights: tuple[float, ...] | None = None
        weight_total: float | None = None
        if operation == "enemy_intent":
            source_candidates = metadata.get("semanticCandidates")
            if (
                not isinstance(source_candidates, list)
                or not source_candidates
                or any(not isinstance(candidate, dict)
                       for candidate in source_candidates)
            ):
                raise SemanticCapabilityUnavailable("Intent support is missing")
            candidates = tuple(source_candidates)
            if any(
                not isinstance(candidate.get("intentId"), str)
                or not candidate.get("intentId")
                or not isinstance(candidate.get("moveId"), str)
                or not candidate.get("moveId")
                for candidate in candidates
            ):
                raise SemanticCapabilityUnavailable("Intent candidates are malformed")
            candidate_ids = tuple(
                candidate["intentId"] for candidate in candidates
            )
            candidate_features = tuple(
                candidate["moveId"] for candidate in candidates
            )
            weights_value = metadata.get("rawEligibleWeights")
            total_value = metadata.get("totalRawWeight")
            if (
                not isinstance(weights_value, list)
                or type(total_value) is not int
                or total_value <= 0
                or len(weights_value) != len(candidate_ids)
                or any(type(value) is not int or value <= 0
                       for value in weights_value)
            ):
                raise SemanticCapabilityUnavailable("Intent weights are missing")
            raw_weights = tuple(float(value) for value in weights_value)
            weight_total = float(total_value)
            semantic_event = metadata.get("semanticEvent")
            if not isinstance(semantic_event, dict):
                raise SemanticCapabilityUnavailable("Intent event identity is missing")
            constraint_fingerprint = semantic_event.get("supportFingerprint")
            randomization_id = event_id
            expected_law_ids = candidate_ids
        else:
            source_candidates = metadata.get("candidates")
            if (
                not isinstance(source_candidates, list)
                or not source_candidates
                or any(not isinstance(candidate, dict)
                       for candidate in source_candidates)
            ):
                raise SemanticCapabilityUnavailable("Draw support is missing")
            candidates = tuple(source_candidates)
            instance_ids = tuple(
                candidate.get("cardInstanceId") for candidate in candidates
            )
            if any(type(instance_id) is not int for instance_id in instance_ids):
                raise SemanticCapabilityUnavailable("Draw instance IDs are malformed")
            candidate_ids = tuple(str(instance_id) for instance_id in instance_ids)
            if any(
                not isinstance(candidate.get("cardModelId"), str)
                or not candidate.get("cardModelId")
                for candidate in candidates
            ):
                raise SemanticCapabilityUnavailable("Draw model labels are malformed")
            candidate_features = tuple(
                candidate["cardModelId"] for candidate in candidates
            )
            epoch = metadata.get("shuffleEpoch")
            if type(epoch) is not int or epoch < 0:
                raise SemanticCapabilityUnavailable("Draw shuffle epoch is missing")
            constraint_fingerprint = metadata.get("constraintFingerprint")
            randomization_id = f"{site_id}:{epoch}"
            expected_law_ids = tuple(
                f"card_instance:{instance_id}" for instance_id in candidate_ids
            )

        support_fingerprint = view.get("eventSupportFingerprint")
        boundary_fingerprint = frame.fingerprint
        law_ids = tuple(
            outcome.get("outcomeId")
            for outcome in law_outcomes
            if isinstance(outcome, dict)
        )
        if (
            not isinstance(support_fingerprint, str)
            or not support_fingerprint
            or metadata.get("supportFingerprint") != support_fingerprint
            or not isinstance(constraint_fingerprint, str)
            or not constraint_fingerprint
            or len(candidate_features) != len(candidate_ids)
            or len(law_ids) != len(law_outcomes)
            or law_ids != expected_law_ids
        ):
            raise SemanticCapabilityUnavailable("Incomplete semantic support metadata")

        return SemanticEvent(
            event_id=event_id,
            boundary_fingerprint=boundary_fingerprint,
            site_id=site_id,
            owner_id=owner_id,
            ordinal=ordinal,
            operation=operation,
            candidate_ids=candidate_ids,
            support_fingerprint=support_fingerprint,
            randomization_id=randomization_id,
            constraint_fingerprint=constraint_fingerprint,
            # Emulator source-law status does not establish Native game
            # parity, so the training event remains conservatively unverified.
            probability_status="UNVERIFIED",
            raw_weights=raw_weights,
            weight_total=weight_total,
            candidate_features=candidate_features,
        )

    def _decode(self, state: Any) -> SemanticBoundary:
        try:
            return self._decode_validated(state)
        except Exception:
            if isinstance(getattr(state, "handle", None), str):
                try:
                    self.session.release_semantic_many([state])
                except Exception:
                    pass
            raise

    def _decode_validated(self, state: Any) -> SemanticBoundary:
        if not state.handle or not state.fingerprint:
            raise SemanticCapabilityUnavailable("Incomplete semantic combat state")
        if state.boundary == "nature":
            view = self.session.observe_nature_choice(
                "semantic_combat", state.handle
            )
            if view.get("fingerprint") != state.fingerprint:
                raise SemanticCapabilityUnavailable("Nature view state fingerprint drift")
            event_view = view.get("natureChoice")
            if not isinstance(event_view, dict):
                raise SemanticCapabilityUnavailable(
                    "Nature boundary has no supported Nature-choice view"
                )
            event = self._decode_event(state, event_view)
            return SemanticBoundary(
                state.handle, "chance", state.fingerprint, event,
                None, (), None, state,
            )
        if state.boundary in ("player", "player_choice"):
            observation = state.observe()
            action_rows = state.actions()
            try:
                actions = tuple(row["actionId"] for row in action_rows)
            except (TypeError, KeyError) as error:
                raise SemanticCapabilityUnavailable(
                    "Malformed semantic player action menu"
                ) from error
            if (
                not isinstance(observation, dict)
                or not actions
                or any(not isinstance(action, str) or not action for action in actions)
                or len(set(actions)) != len(actions)
            ):
                raise SemanticCapabilityUnavailable(
                    "Semantic player boundary violates the fair action contract"
                )
            return SemanticBoundary(
                state.handle, "player", state.fingerprint, None,
                observation, actions, None, state,
            )
        if state.boundary == "terminal":
            if (
                state.terminal is not True
                or not isinstance(state.outcome, str)
                or not state.outcome
            ):
                raise SemanticCapabilityUnavailable(
                    "Terminal combat outcome is missing or unclassified"
                )
            return SemanticBoundary(
                state.handle, "terminal", state.fingerprint, None,
                None, (), state.outcome, state, state.exit_hp,
            )
        raise SemanticCapabilityUnavailable(
            f"Unknown semantic combat boundary: {state.boundary!r}"
        )

    def resolve(
        self, frame: SemanticBoundary, proposal: SemanticProposal,
    ) -> SemanticBoundary:
        if frame.phase != "chance" or frame.event is None or frame.combat is None:
            raise ValueError("Nature may only resolve a pending semantic chance")
        proposal.validate(frame.event)
        if frame.event.operation == "enemy_intent":
            child = frame.combat.resolve_intent(proposal.selected_id)
        elif frame.event.operation == "draw_card":
            child = frame.combat.resolve_draw_card(int(proposal.selected_id))
        else:
            raise SemanticCapabilityUnavailable("Unsupported semantic Nature operation")
        return self._decode(child)

    def step(self, frame: SemanticBoundary, action_id: str) -> SemanticBoundary:
        if (
            frame.phase != "player"
            or frame.combat is None
            or action_id not in frame.legal_actions
        ):
            raise ValueError("Illegal semantic player action or wrong decision phase")
        return self._decode(frame.combat.step(action_id))

    def release(self, *frames: SemanticBoundary) -> int:
        states = [frame.combat for frame in frames if frame.combat is not None]
        if len(states) != len(frames):
            raise ValueError("Cannot release a boundary without its semantic combat")
        return self.session.release_semantic_many(states)
