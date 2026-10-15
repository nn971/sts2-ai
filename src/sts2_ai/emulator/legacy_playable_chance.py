"""Opt-in bridge to the *existing* v111 legacy playable controlled fixture.

The emulator owns all mechanics, hidden shuffle order and event validation.
This bridge never converts legacy swap tickets into deferred semantic draws.
Unlike the proposed semantic_chance bridge, this protocol is implemented by
sts2-emulator's isolated RNG/Inspector integration branch (issue #143).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sts2_ai.nature.protocol import ChanceEvent

from .controlled_chance import ControlledChanceUnavailable, NatureWire

SCHEMA = "sts2-v111-controlled-playable-fixture-v1"
SCOPE = "isolated-playable-combat-stage-continuations-v1"
FAIR_POLICY = "prototype-fair-v0"
WIRE_SCHEMA = "prototype-ai-jsonl-v0"
KNOWN_PHASES = frozenset({
    "opening_shuffle", "next_turn_intent", "draw_recycle",
    "random_target", "generated_card", "draw_insertion",
    "generation_pool_shuffle", "random_card_cost", "drawn_card_cost",
    "player",
})


@dataclass(frozen=True, slots=True)
class LegacyChance:
    """Primitive ticket with emulator-authored identity and support hash."""

    event: ChanceEvent
    event_id: str
    support_fingerprint: str
    probability_status: str

    @classmethod
    def decode(cls, pending: Mapping[str, Any], fingerprint: str) -> LegacyChance:
        if (not isinstance(pending.get("eventId"), str) or
                not pending["eventId"] or
                not isinstance(pending.get("supportFingerprint"), str) or
                not pending["supportFingerprint"]):
            raise ControlledChanceUnavailable("Missing authoritative chance identity")
        if (pending.get("probabilityStatus") != "UNVERIFIED" or
                pending.get("referenceLawId") is not None):
            raise ControlledChanceUnavailable("Unexpected chance reference-law claim")
        return cls(
            event=ChanceEvent.from_isolated_wire(dict(pending), fingerprint=fingerprint),
            event_id=pending["eventId"],
            support_fingerprint=pending["supportFingerprint"],
            probability_status="UNVERIFIED",
        )


@dataclass(frozen=True, slots=True)
class LegacyFrame:
    handle: str
    fingerprint: str
    phase: str
    pending: LegacyChance | None
    terminal: bool
    transcript: tuple[dict[str, Any], ...]

    @property
    def at_player(self) -> bool:
        return self.phase == "player" and not self.terminal

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> LegacyFrame:
        if raw.get("schemaId") != SCHEMA:
            raise ControlledChanceUnavailable("Legacy controlled schema mismatch")
        handle, fingerprint, phase = (
            raw.get("controlledHandle"), raw.get("fingerprint"), raw.get("phase")
        )
        if (not isinstance(handle, str) or not handle or
                not isinstance(fingerprint, str) or not fingerprint or
                phase not in KNOWN_PHASES or type(raw.get("terminal")) is not bool):
            raise ControlledChanceUnavailable("Incomplete controlled boundary")
        terminal = raw["terminal"]
        pending_raw = raw.get("pending")
        if (terminal or phase == "player") and pending_raw is not None:
            raise ControlledChanceUnavailable("Chance pending at player/terminal boundary")
        if not terminal and phase != "player" and pending_raw is None:
            raise ControlledChanceUnavailable("Missing pending controlled chance")
        if pending_raw is not None and not isinstance(pending_raw, dict):
            raise ControlledChanceUnavailable("Malformed controlled chance request")
        previous = raw.get("transcript")
        if previous is not None and (
            not isinstance(previous, list) or
            any(not isinstance(x, dict) for x in previous)
        ):
            raise ControlledChanceUnavailable("Malformed controlled transcript")
        pending = (
            LegacyChance.decode(pending_raw, fingerprint)
            if pending_raw is not None else None
        )
        return cls(handle, fingerprint, phase, pending, terminal, tuple(previous or ()))


@dataclass(frozen=True, slots=True)
class FairDecision:
    observation: dict[str, Any]
    observation_hash: str
    action_ids: tuple[str, ...]


class LegacyPlayableBridge:
    """Strict consumer of the real, limited controlled_* JSONL commands."""

    def __init__(self, wire: NatureWire, *, emulator_revision: str):
        if not emulator_revision:
            raise ValueError("Pin the exact emulator revision")
        hello = wire.call("hello")
        if (hello.get("wireSchemaId") != WIRE_SCHEMA or
                hello.get("fairPolicyId") != FAIR_POLICY or
                hello.get("controlledChanceSchemaId") != SCHEMA or
                hello.get("controlledChanceScope") != SCOPE):
            raise ControlledChanceUnavailable("Playable controlled fixture unavailable")
        sites = hello.get("controlledChanceSupportedSites")
        if (not isinstance(sites, list) or not sites or
                any(not isinstance(s, str) or not s for s in sites)):
            raise ControlledChanceUnavailable("Invalid controlled site capabilities")
        self.sites = frozenset(sites)
        self.wire = wire
        self.emulator_revision = emulator_revision

    def _read(self, raw: Mapping[str, Any], *, parent: LegacyFrame | None = None,
              chosen_index: int | None = None) -> LegacyFrame:
        child = LegacyFrame.decode(raw)
        if parent is not None:
            if raw.get("parent") != parent.handle:
                raise ControlledChanceUnavailable("Controlled continuation parent mismatch")
            before, after = self.trace(parent), self.trace(child)
            if len(after) != len(before) + (chosen_index is not None):
                raise ControlledChanceUnavailable("Controlled chance transcript length drift")
            if after[:len(before)] != before:
                raise ControlledChanceUnavailable("Controlled chance transcript prefix changed")
            if chosen_index is not None:
                assert parent.pending is not None
                last = after[-1]
                if (last.get("eventId") != parent.pending.event_id or
                        last.get("selectedIndex") != chosen_index):
                    raise ControlledChanceUnavailable("Controlled chance outcome replay drift")
        return child

    def begin(self, *, completed_intents_handle: str, seed: str) -> LegacyFrame:
        if not completed_intents_handle or not seed:
            raise ValueError("Completed intent provenance and seed are required")
        return self._read(self.wire.call(
            "controlled_begin", nature_handle=completed_intents_handle, seed=seed,
        ))

    def trace(self, frame: LegacyFrame) -> tuple[dict[str, Any], ...]:
        raw = self.wire.call("controlled_trace", controlled_handle=frame.handle)
        if raw.get("schemaId") != "sts2-v111-nature-event-metadata-v1":
            raise ControlledChanceUnavailable("Controlled trace metadata schema mismatch")
        if raw.get("fingerprint") != frame.fingerprint:
            raise ControlledChanceUnavailable("Controlled state fingerprint changed")
        entries = raw.get("transcript")
        if not isinstance(entries, list) or any(not isinstance(x, dict) for x in entries):
            raise ControlledChanceUnavailable("Malformed controlled replay transcript")
        return tuple(entries)

    def resolve(self, frame: LegacyFrame, index: int) -> LegacyFrame:
        if frame.pending is None:
            raise ValueError("No pending controlled Nature event")
        frame.pending.event.check_index(index)
        return self._read(self.wire.call(
            "controlled_resolve", controlled_handle=frame.handle,
            fingerprint=frame.fingerprint, selected_index=index,
        ), parent=frame, chosen_index=index)

    def observe(self, frame: LegacyFrame) -> FairDecision:
        if not frame.at_player:
            raise ValueError("Player observation only at a fair decision boundary")
        raw = self.wire.call(
            "controlled_observe", controlled_handle=frame.handle,
            policy_id=FAIR_POLICY,
        )
        if raw.get("policyId") != FAIR_POLICY or raw.get("terminal") is not False:
            raise ControlledChanceUnavailable("Controlled fair observation mismatch")
        payload = raw.get("payloadJson")
        try:
            observation = json.loads(payload) if isinstance(payload, str) else None
        except (ValueError, TypeError) as exc:
            raise ControlledChanceUnavailable("Invalid player observation JSON") from exc
        actions = raw.get("actions")
        if (not isinstance(observation, dict) or
                not isinstance(raw.get("observationHash"), str) or
                not raw["observationHash"] or not isinstance(actions, list)):
            raise ControlledChanceUnavailable("Malformed fair player observation")
        ids = tuple(
            a.get("actionId") if isinstance(a, dict) else None for a in actions
        )
        if not ids or any(not isinstance(a, str) or not a for a in ids):
            raise ControlledChanceUnavailable("Invalid player action IDs")
        if len(set(ids)) != len(ids):
            raise ControlledChanceUnavailable("Duplicate legal player action ID")
        legal = self.wire.call(
            "controlled_legal_actions", controlled_handle=frame.handle,
        ).get("actions")
        if not isinstance(legal, list) or [a.get("actionId") for a in legal] != list(ids):
            raise ControlledChanceUnavailable("Player action menu drift")
        return FairDecision(observation, raw["observationHash"], ids)

    def step(self, frame: LegacyFrame, action_id: str,
             *, decision: FairDecision | None = None) -> LegacyFrame:
        if not frame.at_player:
            raise ValueError("Nature boundary requires resolution before player action")
        available = decision if decision is not None else self.observe(frame)
        if action_id not in available.action_ids:
            raise ValueError("Player selected an illegal action ID")
        return self._read(self.wire.call(
            "controlled_step", controlled_handle=frame.handle, action_id=action_id,
        ), parent=frame)

    def fork(self, frame: LegacyFrame) -> LegacyFrame:
        raw = self.wire.call("controlled_fork", controlled_handle=frame.handle)
        if (raw.get("parent") != frame.handle or
                raw.get("fingerprint") != frame.fingerprint or
                not isinstance(raw.get("controlledHandle"), str)):
            raise ControlledChanceUnavailable("Fork provenance mismatch")
        forked = LegacyFrame(
            handle=raw["controlledHandle"], fingerprint=frame.fingerprint,
            phase=frame.phase, pending=frame.pending, terminal=frame.terminal,
            transcript=frame.transcript,
        )
        if self.trace(forked) != self.trace(frame):
            raise ControlledChanceUnavailable("Fork chance transcript divergence")
        return forked

    def save_handle(self, handle: str) -> str:
        raw = self.wire.call("controlled_save", controlled_handle=handle)
        if raw.get("schemaId") != SCHEMA or not isinstance(raw.get("snapshot"), str):
            raise ControlledChanceUnavailable("Invalid controlled snapshot")
        return raw["snapshot"]

    def save(self, frame: LegacyFrame) -> str:
        return self.save_handle(frame.handle)

    def load(self, snapshot: str) -> LegacyFrame:
        return self._read(self.wire.call("controlled_load", snapshot=snapshot))

    def release(self, *frames: LegacyFrame) -> None:
        handles = list(dict.fromkeys(f.handle for f in frames))
        if handles:
            self.wire.call("controlled_release_many", controlled_handles=handles)
