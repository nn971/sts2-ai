from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .protocol import (
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
    Transition,
)

type JsonObject = dict[str, Any]


class PrototypeJsonlBackend:
    """Long-lived adapter to the prototype C# emulator JSONL bridge."""

    FAIR_POLICY_ID = "prototype-fair-v0"

    def __init__(
        self,
        emulator_root: Path | None = None,
        command: Sequence[str] | None = None,
    ) -> None:
        self._emulator_root = (
            emulator_root.resolve()
            if emulator_root is not None
            else (Path(__file__).resolve().parents[3] / "emulator").resolve()
        )
        self._command = tuple(command) if command is not None else self._default_command()
        self._next_request_id = 1
        self._closed = False

        self._process: subprocess.Popen[str] = subprocess.Popen(
            self._command,
            cwd=self._emulator_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            raise RuntimeError("Failed to open emulator JSONL stdin/stdout")

        hello = self._request("hello")
        self._binding_version = self._required_str(hello, "wireSchemaId")
        self._ai_schema_id = self._required_str(hello, "aiSchemaId")
        self._fair_policy_id = self._required_str(hello, "fairPolicyId")
        self._emulator_revision = self._read_emulator_revision()

    def _default_command(self) -> tuple[str, ...]:
        project = self._emulator_root / "src" / "Sts2Emulator.Cli"
        return (
            "dotnet",
            "run",
            "--project",
            str(project),
            "--configuration",
            "Release",
            "--",
            "prototype-ai-jsonl",
        )

    @property
    def binding_version(self) -> str:
        return self._binding_version

    @property
    def ai_schema_id(self) -> str:
        return self._ai_schema_id

    @property
    def fair_policy_id(self) -> str:
        return self._fair_policy_id

    @property
    def emulator_revision(self) -> str:
        return self._emulator_revision

    def reset(self, seed: str) -> StateHandle:
        response = self._request("reset", seed=seed)
        return self._required_str(response, "stateHandle")

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]:
        response = self._request("legal_actions", state_handle=state)
        raw_actions = response.get("actions")
        if not isinstance(raw_actions, list):
            raise RuntimeError("Emulator response field 'actions' is not a list")
        return tuple(self._parse_action(item) for item in raw_actions)

    def step(self, state: StateHandle, action: LegalAction) -> Transition:
        response = self._request(
            "step",
            state_handle=state,
            action_id=action.action_id,
        )
        child = self._required_str(response, "child")
        returned_action = self._parse_action(response.get("action"))
        if returned_action.action_id != action.action_id:
            raise RuntimeError(
                "Emulator returned a different action from the requested action"
            )
        return Transition(
            parent=self._required_str(response, "parent"),
            action=returned_action,
            child=child,
            terminal=self._required_bool(response, "terminal"),
        )

    def fork(self, state: StateHandle) -> StateHandle:
        response = self._request("fork", state_handle=state)
        return self._required_str(response, "child")

    def exact_hash(self, state: StateHandle) -> str:
        response = self._request("exact_hash", state_handle=state)
        return self._required_str(response, "exactHash")

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation:
        response = self._request(
            "observe",
            state_handle=state,
            policy_id=policy.policy_id,
        )
        return Observation(
            policy_id=self._required_str(response, "policyId"),
            payload_json=self._required_str(response, "payloadJson"),
            observation_hash=self._required_str(response, "observationHash"),
        )

    def is_terminal(self, state: StateHandle) -> bool:
        response = self._request("is_terminal", state_handle=state)
        return self._required_bool(response, "terminal")

    def release_many(self, states: Sequence[StateHandle]) -> int:
        if not states:
            return 0
        response = self._request("release_many", state_handles=list(states))
        released = response.get("released")
        if not isinstance(released, int):
            raise RuntimeError("Emulator response field 'released' is not an integer")
        return released

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self._request("close")
        finally:
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

    def __enter__(self) -> PrototypeJsonlBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _request(self, operation: str, **fields: object) -> JsonObject:
        if self._closed:
            raise RuntimeError("Emulator backend is closed")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"Emulator subprocess exited with code {self._process.returncode}"
            )

        request_id = f"py-{self._next_request_id}"
        self._next_request_id += 1
        request: JsonObject = {
            "request_id": request_id,
            "op": operation,
            **fields,
        }

        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise RuntimeError("Emulator JSONL pipes are unavailable")

        stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        stdin.flush()

        line = stdout.readline()
        if line == "":
            raise RuntimeError("Emulator JSONL bridge closed stdout unexpectedly")

        decoded = json.loads(line)
        if not isinstance(decoded, dict):
            raise RuntimeError("Emulator JSONL response is not an object")
        response = cast(JsonObject, decoded)

        if response.get("requestId") != request_id:
            raise RuntimeError(
                "Emulator JSONL response request ID does not match the request"
            )
        if response.get("ok") is not True:
            error_type = response.get("errorType", "EmulatorError")
            message = response.get("error", "unknown emulator error")
            raise RuntimeError(f"{error_type}: {message}")
        return response

    def _read_emulator_revision(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self._emulator_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return result.stdout.strip() or "unknown"

    @staticmethod
    def _parse_action(value: object) -> LegalAction:
        if not isinstance(value, dict):
            raise RuntimeError("Emulator action is not an object")
        action = cast(JsonObject, value)
        return LegalAction(
            action_id=PrototypeJsonlBackend._required_str(action, "actionId"),
            kind=PrototypeJsonlBackend._required_str(action, "kind"),
            payload_json=PrototypeJsonlBackend._required_str(action, "payloadJson"),
        )

    @staticmethod
    def _required_str(value: JsonObject, key: str) -> str:
        field = value.get(key)
        if not isinstance(field, str):
            raise RuntimeError(f"Emulator response field '{key}' is not a string")
        return field

    @staticmethod
    def _required_bool(value: JsonObject, key: str) -> bool:
        field = value.get(key)
        if not isinstance(field, bool):
            raise RuntimeError(f"Emulator response field '{key}' is not a boolean")
        return field
