from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TextIO, cast

from sts2_ai.emulator.protocol import (
    EmulatorBackend,
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
    Transition,
)


class JsonlClient(Protocol):
    """Synchronous request/response transport for the emulator JSONL bridge."""

    def request(self, payload: Mapping[str, object]) -> Mapping[str, object]: ...

    def close(self) -> None: ...


class JsonlRpcError(RuntimeError):
    """Raised when the emulator bridge returns a structured error response."""


class SubprocessJsonlClient:
    """Long-lived line-oriented JSON client backed by a subprocess."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
    ) -> None:
        self._process = subprocess.Popen(
            list(command),
            cwd=None if cwd is None else str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise RuntimeError("Failed to open emulator JSONL stdin/stdout")

        self._stdin: TextIO = self._process.stdin
        self._stdout: TextIO = self._process.stdout
        self._next_request_id = 1
        self._lock = threading.Lock()
        self._closed = False

    def request(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        with self._lock:
            if self._closed:
                raise RuntimeError("JSONL client is closed")

            request_id = f"py-{self._next_request_id}"
            self._next_request_id += 1
            message = dict(payload)
            message["request_id"] = request_id

            self._stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self._stdin.flush()

            line = self._stdout.readline()
            if line == "":
                stderr = ""
                if self._process.stderr is not None:
                    stderr = self._process.stderr.read()
                raise RuntimeError(
                    "Emulator JSONL bridge exited before responding"
                    + (f": {stderr.strip()}" if stderr.strip() else "")
                )

            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise RuntimeError("Emulator JSONL response must be an object")

            response = cast(dict[str, object], raw)
            if response.get("requestId") != request_id:
                raise RuntimeError(
                    "Emulator JSONL response request ID mismatch: "
                    f"expected {request_id!r}, got {response.get('requestId')!r}"
                )

            if response.get("ok") is not True:
                error_type = response.get("errorType", "EmulatorError")
                error = response.get("error", "unknown emulator bridge error")
                raise JsonlRpcError(f"{error_type}: {error}")

            return response

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                request_id = f"py-{self._next_request_id}"
                self._next_request_id += 1
                self._stdin.write(
                    json.dumps(
                        {"op": "close", "request_id": request_id},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self._stdin.flush()
                self._stdout.readline()
            except (BrokenPipeError, OSError):
                pass
            finally:
                self._closed = True
                if self._process.poll() is None:
                    self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

    def __enter__(self) -> SubprocessJsonlClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class PrototypeJsonlBackend(EmulatorBackend):
    """Python adapter for `sts2-emulator prototype-ai-jsonl`.

    The bridge owns immutable state handles. `step` and `expand` create new handles and leave
    their parents valid, which makes the backend directly usable by tree search.
    """

    expected_wire_schema = "prototype-ai-jsonl-v0"
    expected_ai_schema = "prototype-ai-v0"

    def __init__(
        self,
        client: JsonlClient,
        *,
        emulator_revision: str,
    ) -> None:
        self._client = client
        self._emulator_revision = emulator_revision

        hello = self._client.request({"op": "hello"})
        self._binding_version = _required_str(hello, "wireSchemaId")
        self._ai_schema = _required_str(hello, "aiSchemaId")
        self._fair_policy_id = _required_str(hello, "fairPolicyId")
        self._ruleset_id = _required_str(hello, "rulesetId")

        if self._binding_version != self.expected_wire_schema:
            raise RuntimeError(
                "Unsupported emulator wire schema "
                f"{self._binding_version!r}; expected {self.expected_wire_schema!r}"
            )
        if self._ai_schema != self.expected_ai_schema:
            raise RuntimeError(
                "Unsupported emulator AI schema "
                f"{self._ai_schema!r}; expected {self.expected_ai_schema!r}"
            )

    @classmethod
    def from_repo(
        cls,
        repo_root: Path,
        *,
        configuration: str = "Release",
        emulator_revision: str | None = None,
    ) -> PrototypeJsonlBackend:
        emulator_root = repo_root / "emulator"
        project = emulator_root / "src" / "Sts2Emulator.Cli" / "Sts2Emulator.Cli.csproj"
        client = SubprocessJsonlClient(
            [
                "dotnet",
                "run",
                "--project",
                str(project),
                "--configuration",
                configuration,
                "--",
                "prototype-ai-jsonl",
            ],
            cwd=repo_root,
        )
        revision = emulator_revision or _git_revision(emulator_root) or "unknown"
        try:
            return cls(client, emulator_revision=revision)
        except Exception:
            client.close()
            raise

    @property
    def binding_version(self) -> str:
        return self._binding_version

    @property
    def emulator_revision(self) -> str:
        return self._emulator_revision

    @property
    def fair_policy(self) -> InformationPolicy:
        return InformationPolicy(self._fair_policy_id)

    @property
    def ruleset_id(self) -> str:
        return self._ruleset_id

    def reset(self, seed: str) -> StateHandle:
        response = self._client.request({"op": "reset", "seed": seed})
        return _required_str(response, "stateHandle")

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]:
        response = self._client.request(
            {"op": "legal_actions", "state_handle": state}
        )
        return tuple(_parse_actions(response.get("actions")))

    def step(self, state: StateHandle, action: LegalAction) -> Transition:
        response = self._client.request(
            {
                "op": "step",
                "state_handle": state,
                "action_id": action.action_id,
            }
        )
        wire_action = _parse_action(_required_mapping(response, "action"))
        if wire_action.action_id != action.action_id:
            raise RuntimeError("Emulator bridge returned a different action than requested")

        return Transition(
            parent=_required_str(response, "parent"),
            action=wire_action,
            child=_required_str(response, "child"),
            terminal=_required_bool(response, "terminal"),
        )

    def fork(self, state: StateHandle) -> StateHandle:
        response = self._client.request({"op": "fork", "state_handle": state})
        return _required_str(response, "child")

    def exact_hash(self, state: StateHandle) -> str:
        response = self._client.request(
            {"op": "exact_hash", "state_handle": state}
        )
        return _required_str(response, "exactHash")

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation:
        response = self._client.request(
            {
                "op": "observe",
                "state_handle": state,
                "policy_id": policy.policy_id,
            }
        )
        return Observation(
            policy_id=_required_str(response, "policyId"),
            payload_json=_required_str(response, "payloadJson"),
            observation_hash=_required_str(response, "observationHash"),
        )

    def is_terminal(self, state: StateHandle) -> bool:
        response = self._client.request(
            {"op": "is_terminal", "state_handle": state}
        )
        return _required_bool(response, "terminal")

    def expand(self, state: StateHandle) -> tuple[Transition, ...]:
        response = self._client.request(
            {"op": "expand", "state_handle": state}
        )
        raw = response.get("expansions")
        if not isinstance(raw, list):
            raise RuntimeError("Emulator expand response must contain an expansions array")

        transitions: list[Transition] = []
        for item in raw:
            if not isinstance(item, dict):
                raise RuntimeError("Every emulator expansion must be an object")
            mapping = cast(dict[str, object], item)
            transitions.append(
                Transition(
                    parent=_required_str(mapping, "parent"),
                    action=_parse_action(_required_mapping(mapping, "action")),
                    child=_required_str(mapping, "child"),
                    terminal=_required_bool(mapping, "terminal"),
                )
            )
        return tuple(transitions)

    def release_many(self, states: Sequence[StateHandle]) -> int:
        response = self._client.request(
            {"op": "release_many", "state_handles": list(states)}
        )
        released = response.get("released")
        if not isinstance(released, int) or isinstance(released, bool):
            raise RuntimeError("Emulator release_many response has invalid released count")
        return released

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PrototypeJsonlBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _parse_actions(raw: object) -> list[LegalAction]:
    if not isinstance(raw, list):
        raise RuntimeError("Emulator legal-actions response must contain an actions array")
    result: list[LegalAction] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("Every emulator action must be an object")
        result.append(_parse_action(cast(dict[str, object], item)))
    return result


def _parse_action(raw: Mapping[str, object]) -> LegalAction:
    return LegalAction(
        action_id=_required_str(raw, "actionId"),
        kind=_required_str(raw, "kind"),
        payload_json=_required_str(raw, "payloadJson"),
    )


def _required_mapping(
    raw: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Emulator response field {key!r} must be an object")
    return cast(dict[str, object], value)


def _required_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Emulator response field {key!r} must be a string")
    return value


def _required_bool(raw: Mapping[str, object], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"Emulator response field {key!r} must be a boolean")
    return value


def _git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None
