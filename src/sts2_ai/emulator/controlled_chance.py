"""Opt-in adapter to the emulator's *isolated* v111 Nature JSONL protocol.

The v111 bridge currently exposes only separate HP/intent fixture handles.
No ordinary playable StateHandle or player-policy observation is obtainable
from this interface. This module fails on missing or expanded capabilities
rather than silently routing outcomes through ordinary seeded `step`.
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sts2_ai.nature.protocol import (
    ISOLATED_EMULATOR_SCHEMA,
    ChanceEvent,
    NaturePolicy,
)
from sts2_ai.nature.trace import (
    ChanceTraceEntry,
    ChanceTranscript,
    record_entry,
)

WIRE_SCHEMA = "prototype-ai-jsonl-v0"
ISOLATED_SCOPE = "isolated-combat-entry-only"


class ControlledChanceUnavailable(RuntimeError):
    """Emulator cannot supply the requested Nature operation/protocol."""


class NatureWire(Protocol):
    def call(self, operation: str, **payload: Any) -> dict[str, Any]: ...


class NatureJsonlProcess:
    """Explicit user-supplied .NET CLI; no emulator checkout/gitlink changes."""

    def __init__(self, command: list[str], *, cwd: str | None = None):
        if not command:
            raise ValueError("Nature JSONL command is required")
        self._process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._lock = threading.RLock()
        self._counter = 0

    def call(self, operation: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            if self._process.poll() is not None:
                raise ControlledChanceUnavailable("Nature emulator process exited")
            request_id = str(self._counter)
            self._counter += 1
            assert self._process.stdin is not None
            assert self._process.stdout is not None
            self._process.stdin.write(
                json.dumps({"op": operation, "request_id": request_id, **payload}) + "\n"
            )
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise ControlledChanceUnavailable("Nature emulator JSONL disconnected")
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise ControlledChanceUnavailable("Malformed Nature response object")
            result: dict[str, Any] = decoded
            if result.get("requestId") != request_id:
                raise ControlledChanceUnavailable("Nature response request-id mismatch")
            if result.get("ok") is not True:
                raise ControlledChanceUnavailable(
                    f"{result.get('errorType')}: {result.get('error')}"
                )
            return result

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self.call("close")
            finally:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                self._process.wait(timeout=10)

    def __enter__(self) -> NatureJsonlProcess:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class IsolatedBoundary:
    handle: str
    kind: str
    fingerprint: str
    pending: ChanceEvent | None
    complete: bool

    @classmethod
    def decode(cls, raw: Mapping[str, Any]) -> IsolatedBoundary:
        if raw.get("kind") not in ("initial_hp", "intents"):
            raise ControlledChanceUnavailable("Unexpected Nature boundary kind")
        pending = raw.get("pending")
        if raw.get("complete") is not (pending is None):
            raise ControlledChanceUnavailable("Inconsistent Nature pending/complete fields")
        if not raw.get("fingerprint") or not raw.get("natureHandle"):
            raise ControlledChanceUnavailable("Missing Nature handle/fingerprint")
        event = (
            ChanceEvent.from_isolated_wire(pending, fingerprint=raw["fingerprint"])
            if pending is not None else None
        )
        return cls(
            handle=raw["natureHandle"], kind=raw["kind"],
            fingerprint=raw["fingerprint"], pending=event,
            complete=raw["complete"],
        )


@dataclass(frozen=True, slots=True)
class IsolatedNatureResult:
    final: IsolatedBoundary
    transcript: ChanceTranscript


class IsolatedNatureBridge:
    """Strict adapter of the existing, explicitly non-playable D3/D4 API."""

    def __init__(self, wire: NatureWire, *, emulator_revision: str):
        if not emulator_revision:
            raise ValueError("Explicit emulator git revision required for provenance")
        hello = wire.call("hello")
        if hello.get("wireSchemaId") != WIRE_SCHEMA:
            raise ControlledChanceUnavailable("Unexpected Nature JSONL wire schema")
        if hello.get("natureBoundarySchemaId") != ISOLATED_EMULATOR_SCHEMA:
            raise ControlledChanceUnavailable("Unsupported Nature boundary schema")
        if hello.get("natureBoundaryScope") != ISOLATED_SCOPE:
            raise ControlledChanceUnavailable("Nature scope changed: re-audit interface")
        if set(hello.get("natureBoundaryKinds", [])) != {"initial_hp", "intents"}:
            raise ControlledChanceUnavailable("Nature boundary capability set changed")
        self.wire = wire
        self.emulator_revision = emulator_revision

    def begin(
        self, *, seed: str, kind: str, act: int, encounter_id: str,
        native_entry: str, total_floor: int, ascension: int = 0,
        map_column: int = 3, map_row: int = 5,
    ) -> IsolatedBoundary:
        if kind not in ("initial_hp", "intents") or act not in (2, 3):
            raise ControlledChanceUnavailable("Unsupported isolated Nature fixture")
        return IsolatedBoundary.decode(self.wire.call(
            "nature_begin", seed=seed, kind=kind, act=act,
            encounter_id=encounter_id, native_entry=native_entry,
            total_floor=total_floor, ascension=ascension,
            map_column=map_column, map_row=map_row,
        ))

    def resolve(self, boundary: IsolatedBoundary, selected_index: int) -> IsolatedBoundary:
        if boundary.pending is None:
            raise ValueError("Cannot resolve a completed Nature boundary")
        boundary.pending.check_index(selected_index)
        child = IsolatedBoundary.decode(self.wire.call(
            "nature_resolve", nature_handle=boundary.handle,
            fingerprint=boundary.fingerprint, selected_index=selected_index,
        ))
        if child.kind != boundary.kind:
            raise ControlledChanceUnavailable("Nature resolution changed event kind")
        return child

    def advance(self, completed_hp: IsolatedBoundary) -> IsolatedBoundary:
        if completed_hp.kind != "initial_hp" or not completed_hp.complete:
            raise ControlledChanceUnavailable("Only completed HP enters intent choices")
        child = IsolatedBoundary.decode(self.wire.call(
            "nature_advance", nature_handle=completed_hp.handle,
        ))
        if child.kind != "intents":
            raise ControlledChanceUnavailable("Unexpected Nature continuation")
        return child

    def fork(self, boundary: IsolatedBoundary) -> IsolatedBoundary:
        return IsolatedBoundary.decode(
            self.wire.call("nature_fork", nature_handle=boundary.handle)
        )

    def save(self, boundary: IsolatedBoundary) -> str:
        response = self.wire.call("nature_save", nature_handle=boundary.handle)
        if response.get("kind") != boundary.kind:
            raise ControlledChanceUnavailable("Nature snapshot kind mismatch")
        snapshot = response.get("snapshot")
        if not isinstance(snapshot, str) or not snapshot:
            raise ControlledChanceUnavailable("Missing Nature snapshot")
        return snapshot

    def load(self, snapshot: str, *, kind: str) -> IsolatedBoundary:
        return IsolatedBoundary.decode(
            self.wire.call("nature_load", snapshot=snapshot, kind=kind)
        )

    def release(self, *boundaries: IsolatedBoundary) -> int:
        response = self.wire.call(
            "nature_release_many",
            nature_handles=[x.handle for x in boundaries],
        )
        released = response.get("released")
        if type(released) is not int or released < 0:
            raise ControlledChanceUnavailable("Malformed Nature release count")
        return released

    def run_fixture(
        self, policy: NaturePolicy, *, seed: str, kind: str, act: int,
        encounter_id: str, native_entry: str, total_floor: int,
        max_choices: int = 128,
    ) -> IsolatedNatureResult:
        if max_choices <= 0:
            raise ValueError("max_choices must be positive")
        current = self.begin(
            seed=seed, kind=kind, act=act,
            encounter_id=encounter_id, native_entry=native_entry,
            total_floor=total_floor,
        )
        entries: list[ChanceTraceEntry] = []
        while True:
            while not current.complete:
                if len(entries) >= max_choices:
                    raise ControlledChanceUnavailable("Nature decision cap exceeded")
                assert current.pending is not None
                proposal = policy.propose(current.pending)
                proposal.validate(current.pending)
                entries.append(record_entry(current.pending, proposal))
                current = self.resolve(current, proposal.selected_index)
            if current.kind == "initial_hp":
                current = self.advance(current)
            else:
                break
        return IsolatedNatureResult(
            final=current,
            transcript=ChanceTranscript(
                emulator_revision=self.emulator_revision,
                fixture_id=f"{seed}:{act}:{encounter_id}:{native_entry}:{total_floor}:{kind}",
                nature_policy_id=policy.policy_id,
                entries=tuple(entries),
            ),
        )
