"""Opt-in adapter for a *future* playable semantic Nature wire capability.

As of emulator PR #156, the semantic deck is a pure C# provider-state
kernel. Its gameplay/JSONL interface has NOT been implemented. This adapter
negotiates a deliberately separate proposed schema, refuses the current
legacy playable/shuffle interface, and runs against a synthetic wire in CI.
The emulator remains authoritative for deck constraints and game mechanics.
"""
from __future__ import annotations

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
        released = result.get("released")
        if type(released) is not int or released < 0:
            raise SemanticCapabilityUnavailable("Malformed semantic release count")
        return released
