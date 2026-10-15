"""Strict opt-in client for the legacy v111 *playable fixture* chance protocol.

This is an indexed-ticket bridge (including Fisher--Yates swaps), not the
new deferred-card semantic Nature interface. No player view contains pending
Nature metadata; no existing emulator StateHandle is reused.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.controlled_chance import (
    WIRE_SCHEMA,
    ControlledChanceUnavailable,
    IsolatedBoundary,
    NatureWire,
)
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID, OBSERVATION_SCHEMA_ID
from sts2_ai.nature.protocol import ChanceEvent

PLAYABLE_SCHEMA = "sts2-v111-controlled-playable-fixture-v1"
PLAYABLE_SCOPE = "isolated-playable-combat-stage-continuations-v1"


@dataclass(frozen=True, slots=True)
class PlayableBoundary:
    handle: str
    fingerprint: str
    phase: str  # chance | player | terminal
    event: ChanceEvent | None
    native_event_id: str | None
    support_fingerprint: str | None
    site_phase: str

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> PlayableBoundary:
        if raw.get("schemaId") != PLAYABLE_SCHEMA:
            raise ControlledChanceUnavailable("Playable fixture schema mismatch")
        handle, fingerprint = raw.get("controlledHandle"), raw.get("fingerprint")
        if (not isinstance(handle, str) or not handle or
                not isinstance(fingerprint, str) or not fingerprint):
            raise ControlledChanceUnavailable("Incomplete playable handle/fingerprint")
        if type(raw.get("terminal")) is not bool:
            raise ControlledChanceUnavailable("Missing playable terminal flag")
        pending = raw.get("pending")
        site_phase = raw.get("phase")
        if not isinstance(site_phase, str) or not site_phase:
            raise ControlledChanceUnavailable("Missing playable phase")
        if raw["terminal"]:
            if pending is not None:
                raise ControlledChanceUnavailable("Terminal has pending chance")
            phase = "terminal"
        elif pending is None:
            if site_phase != "player":
                raise ControlledChanceUnavailable("Non-player phase without pending chance")
            phase = "player"
        else:
            if not isinstance(pending, dict) or site_phase == "player":
                raise ControlledChanceUnavailable("Invalid playable pending event")
            phase = "chance"
        event = None
        native_id = None
        support = None
        if pending is not None:
            native_id, support = pending.get("eventId"), pending.get("supportFingerprint")
            if not all(isinstance(x, str) and x for x in (native_id, support)):
                raise ControlledChanceUnavailable("Missing native event/support identity")
            event = ChanceEvent.from_isolated_wire(pending, fingerprint=fingerprint)
            if pending.get("supportCount") != event.support_count:
                raise ControlledChanceUnavailable("Inconsistent playable support count")
        return cls(handle, fingerprint, phase, event, native_id, support, site_phase)


@dataclass(frozen=True, slots=True)
class FairPlayerFrame:
    observation: dict[str, Any]
    observation_hash: str
    action_ids: tuple[str, ...]
    terminal: bool


class PlayableNatureBridge:
    """Scoped adapter: begins only from completed isolated HP -> intents."""

    def __init__(self, wire: NatureWire, *, emulator_revision: str):
        if not emulator_revision:
            raise ValueError("Emulator SHA required")
        hello = wire.call("hello")
        if (hello.get("wireSchemaId") != WIRE_SCHEMA or
                hello.get("controlledChanceSchemaId") != PLAYABLE_SCHEMA or
                hello.get("controlledChanceScope") != PLAYABLE_SCOPE or
                hello.get("fairPolicyId") != FAIR_POLICY_ID):
            raise ControlledChanceUnavailable("Playable controlled fixture capability unavailable")
        sites = hello.get("controlledChanceSupportedSites")
        if not isinstance(sites, list) or not sites or any(
            not isinstance(site, str) or not site for site in sites
        ) or len(set(sites)) != len(sites):
            raise ControlledChanceUnavailable("Invalid playable chance site advertisement")
        self.sites = frozenset(sites)
        self.wire = wire
        self.emulator_revision = emulator_revision

    def _decode(self, raw: Mapping[str, Any]) -> PlayableBoundary:
        frame = PlayableBoundary.decode(raw)
        if frame.event and frame.event.site_id not in self.sites:
            raise ControlledChanceUnavailable(
                f"Unadvertised playable chance site: {frame.event.site_id}"
            )
        return frame

    def _child(self, raw: Mapping[str, Any], parent: str) -> PlayableBoundary:
        try:
            if raw.get("parent") != parent:
                raise ControlledChanceUnavailable("Playable child parent mismatch")
            return self._decode(raw)
        except Exception:
            # Server has already allocated the child. A rejected response must
            # not leave an unreachable continuation resident in its process.
            handle = raw.get("controlledHandle")
            if isinstance(handle, str) and handle and handle != parent:
                self.wire.call("controlled_release_many", controlled_handles=[handle])
            raise

    def begin(self, intents: IsolatedBoundary, *, seed: str) -> PlayableBoundary:
        if intents.kind != "intents" or not intents.complete or not seed:
            raise ValueError("Playable fixture needs completed HP and intents with seed")
        return self._decode(self.wire.call(
            "controlled_begin", nature_handle=intents.handle, seed=seed,
        ))

    def resolve(self, frame: PlayableBoundary, index: int) -> PlayableBoundary:
        if frame.phase != "chance" or frame.event is None:
            raise ValueError("Nature choice requires a pending event")
        frame.event.check_index(index)
        raw = self.wire.call(
            "controlled_resolve", controlled_handle=frame.handle,
            fingerprint=frame.fingerprint, selected_index=index,
        )
        return self._child(raw, frame.handle)

    def observe(self, frame: PlayableBoundary) -> FairPlayerFrame:
        if frame.phase != "player":
            raise ValueError("Player observations require a player boundary")
        raw = self.wire.call(
            "controlled_observe", controlled_handle=frame.handle,
            policy_id=FAIR_POLICY_ID,
        )
        if raw.get("policyId") != FAIR_POLICY_ID or raw.get("schemaId") != OBSERVATION_SCHEMA_ID:
            raise ControlledChanceUnavailable("Playable player observation contract changed")
        payload = raw.get("payloadJson")
        if not isinstance(payload, str):
            raise ControlledChanceUnavailable("Missing public observation JSON")
        try:
            observation = json.loads(payload)
        except (TypeError, ValueError) as exc:
            raise ControlledChanceUnavailable("Invalid public observation JSON") from exc
        actions = raw.get("actions")
        if not isinstance(observation, dict) or not isinstance(actions, list):
            raise ControlledChanceUnavailable("Invalid public observation or actions")
        ids = tuple(a.get("actionId") for a in actions if isinstance(a, dict))
        if (len(ids) != len(actions) or
                any(not isinstance(x, str) or not x for x in ids) or
                len(set(ids)) != len(ids)):
            raise ControlledChanceUnavailable("Malformed public action menu")
        if raw.get("terminal") is not False or not isinstance(raw.get("observationHash"), str):
            raise ControlledChanceUnavailable("Player-boundary status/hash mismatch")
        return FairPlayerFrame(observation, raw["observationHash"], ids, False)

    def step(self, frame: PlayableBoundary, action_id: str) -> PlayableBoundary:
        if frame.phase != "player":
            raise ValueError("Player action requires a player boundary")
        menu = self.observe(frame)
        if action_id not in menu.action_ids:
            raise ValueError("Illegal player action")
        raw = self.wire.call(
            "controlled_step", controlled_handle=frame.handle, action_id=action_id,
        )
        return self._child(raw, frame.handle)

    def fork(self, frame: PlayableBoundary) -> PlayableBoundary:
        raw = self.wire.call("controlled_fork", controlled_handle=frame.handle)
        if raw.get("fingerprint") != frame.fingerprint:
            raise ControlledChanceUnavailable("Playable fork changed state")
        # The short fork response carries only the new handle/fingerprint.
        return PlayableBoundary(
            raw["controlledHandle"], frame.fingerprint, frame.phase,
            frame.event, frame.native_event_id, frame.support_fingerprint,
            frame.site_phase,
        )

    def save(self, frame: PlayableBoundary) -> str:
        raw = self.wire.call("controlled_save", controlled_handle=frame.handle)
        if raw.get("schemaId") != PLAYABLE_SCHEMA or not isinstance(raw.get("snapshot"), str):
            raise ControlledChanceUnavailable("Playable snapshot schema mismatch")
        return raw["snapshot"]

    def load(self, snapshot: str) -> PlayableBoundary:
        if not snapshot:
            raise ValueError("Nonempty controlled snapshot required")
        raw = self.wire.call("controlled_load", snapshot=snapshot)
        # Current loader omits pending metadata; inspect at the returned handle.
        handle = raw.get("controlledHandle")
        trace = self.wire.call("controlled_trace", controlled_handle=handle)
        raw = {**raw, "pending": trace.get("pending")}
        return self._decode(raw)

    def release(self, *frames: PlayableBoundary) -> int:
        raw = self.wire.call(
            "controlled_release_many", controlled_handles=[x.handle for x in frames],
        )
        count = raw.get("released")
        if type(count) is not int or not 0 <= count <= len(frames):
            raise ControlledChanceUnavailable("Invalid released handle count")
        return count
