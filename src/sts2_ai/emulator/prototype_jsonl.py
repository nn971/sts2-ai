from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from sts2_ai.emulator.protocol import (
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
    Transition,
)


class PrototypeJsonlBackend:
    """Long-lived subprocess adapter for the prototype emulator JSONL bridge."""

    EXPECTED_WIRE_SCHEMA = "prototype-ai-jsonl-v0"
    EXPECTED_AI_SCHEMA = "prototype-ai-v0"
    EXPECTED_POLICY = "prototype-fair-v0"

    def __init__(
        self,
        command: Sequence[str],
        *,
        emulator_revision: str = "unknown",
        cwd: Path | None = None,
    ) -> None:
        if not command:
            raise ValueError("JSONL backend command must not be empty")

        self._process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._request_counter = 0
        self._emulator_revision = emulator_revision
        self._closed = False

        hello = self._request("hello")
        wire_schema = self._required_string(hello, "wireSchemaId")
        ai_schema = self._required_string(hello, "aiSchemaId")
        fair_policy = self._required_string(hello, "fairPolicyId")

        if wire_schema != self.EXPECTED_WIRE_SCHEMA:
            self.close()
            raise RuntimeError(
                f"Unsupported emulator wire schema {wire_schema!r}; "
                f"expected {self.EXPECTED_WIRE_SCHEMA!r}"
            )
        if ai_schema != self.EXPECTED_AI_SCHEMA:
            self.close()
            raise RuntimeError(
                f"Unsupported emulator AI schema {ai_schema!r}; "
                f"expected {self.EXPECTED_AI_SCHEMA!r}"
            )
        if fair_policy != self.EXPECTED_POLICY:
            self.close()
            raise RuntimeError(
                f"Unsupported emulator fair-information policy {fair_policy!r}; "
                f"expected {self.EXPECTED_POLICY!r}"
            )

        self._binding_version = wire_schema
        self._ruleset_id = self._required_string(hello, "rulesetId")
        self._fair_policy_id = fair_policy

    @classmethod
    def from_repo(
        cls,
        repo_root: Path,
        *,
        ensure_built: bool = True,
    ) -> Self:
        root = repo_root.resolve()
        emulator = root / "emulator"
        project = emulator / "src" / "Sts2Emulator.Cli" / "Sts2Emulator.Cli.csproj"

        if not project.is_file():
            raise FileNotFoundError(
                f"Emulator CLI project is missing at {project}; "
                "initialize the emulator submodule first."
            )

        if ensure_built:
            subprocess.run(
                ["dotnet", "build", str(project), "-c", "Release"],
                cwd=root,
                check=True,
            )

        revision = cls._git_revision(emulator)
        command = [
            "dotnet",
            "run",
            "--project",
            str(project),
            "-c",
            "Release",
            "--no-build",
            "--",
            "prototype-ai-jsonl",
        ]
        return cls(command, emulator_revision=revision, cwd=root)

    @property
    def binding_version(self) -> str:
        return self._binding_version

    @property
    def emulator_revision(self) -> str:
        return self._emulator_revision

    @property
    def ruleset_id(self) -> str:
        return self._ruleset_id

    @property
    def fair_policy(self) -> InformationPolicy:
        return InformationPolicy(self._fair_policy_id)

    def reset(self, seed: str) -> StateHandle:
        response = self._request("reset", seed=seed)
        return self._required_string(response, "stateHandle")

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]:
        response = self._request("legal_actions", state_handle=state)
        raw_actions = response.get("actions")
        if not isinstance(raw_actions, list):
            raise RuntimeError("Emulator legal_actions response has no action list")

        actions: list[LegalAction] = []
        for raw in raw_actions:
            if not isinstance(raw, dict):
                raise RuntimeError("Emulator returned a non-object legal action")
            actions.append(
                LegalAction(
                    action_id=self._required_string(raw, "actionId"),
                    kind=self._required_string(raw, "kind"),
                    payload_json=self._required_string(raw, "payloadJson"),
                )
            )
        return tuple(actions)

    def step(self, state: StateHandle, action: LegalAction) -> Transition:
        response = self._request(
            "step",
            state_handle=state,
            action_id=action.action_id,
        )
        raw_action = response.get("action")
        if not isinstance(raw_action, dict):
            raise RuntimeError("Emulator step response has no action object")

        echoed = LegalAction(
            action_id=self._required_string(raw_action, "actionId"),
            kind=self._required_string(raw_action, "kind"),
            payload_json=self._required_string(raw_action, "payloadJson"),
        )
        if echoed.action_id != action.action_id:
            raise RuntimeError(
                f"Emulator stepped action {echoed.action_id!r}, "
                f"not requested action {action.action_id!r}"
            )

        return Transition(
            parent=self._required_string(response, "parent"),
            action=echoed,
            child=self._required_string(response, "child"),
            terminal=self._required_bool(response, "terminal"),
        )

    def fork(self, state: StateHandle) -> StateHandle:
        response = self._request("fork", state_handle=state)
        return self._required_string(response, "child")

    def exact_hash(self, state: StateHandle) -> str:
        response = self._request("exact_hash", state_handle=state)
        return self._required_string(response, "exactHash")

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation:
        response = self._request(
            "observe",
            state_handle=state,
            policy_id=policy.policy_id,
        )
        return Observation(
            policy_id=self._required_string(response, "policyId"),
            payload_json=self._required_string(response, "payloadJson"),
            observation_hash=self._required_string(response, "observationHash"),
        )

    def is_terminal(self, state: StateHandle) -> bool:
        response = self._request("is_terminal", state_handle=state)
        return self._required_bool(response, "terminal")

    def manifest(self) -> dict[str, Any]:
        response = self._request("manifest")
        manifest = response.get("manifest")
        if not isinstance(manifest, dict):
            raise RuntimeError("Emulator manifest response has no manifest object")
        return manifest

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._process.poll() is None:
            try:
                self._request("close", allow_closed=True)
            except (BrokenPipeError, RuntimeError):
                pass

        if self._process.stdin is not None:
            self._process.stdin.close()

        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _request(
        self,
        operation: str,
        *,
        allow_closed: bool = False,
        **payload: object,
    ) -> dict[str, Any]:
        if self._closed and not allow_closed:
            raise RuntimeError("Prototype JSONL backend is closed")
        if self._process.poll() is not None:
            raise RuntimeError(self._terminated_message())

        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise RuntimeError("Prototype JSONL backend has no process pipes")

        self._request_counter += 1
        request_id = f"py-{self._request_counter}"
        request = {"request_id": request_id, "op": operation, **payload}
        stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        stdin.flush()

        line = stdout.readline()
        if not line:
            raise RuntimeError(self._terminated_message())

        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Emulator emitted non-JSON protocol output: {line.rstrip()!r}"
            ) from error

        if not isinstance(response, dict):
            raise RuntimeError("Emulator JSONL response is not an object")
        if response.get("requestId") != request_id:
            raise RuntimeError(
                f"Emulator response request ID {response.get('requestId')!r} "
                f"does not match {request_id!r}"
            )
        if response.get("ok") is not True:
            error_type = response.get("errorType", "EmulatorError")
            message = response.get("error", "unknown emulator error")
            raise RuntimeError(f"{error_type}: {message}")

        return response

    def _terminated_message(self) -> str:
        code = self._process.poll()
        stderr = ""
        if self._process.stderr is not None and code is not None:
            stderr = self._process.stderr.read().strip()
        suffix = f": {stderr}" if stderr else ""
        return f"Prototype emulator process terminated with code {code}{suffix}"

    @staticmethod
    def _required_string(value: dict[str, Any], name: str) -> str:
        result = value.get(name)
        if not isinstance(result, str):
            raise RuntimeError(f"Emulator response field {name!r} must be a string")
        return result

    @staticmethod
    def _required_bool(value: dict[str, Any], name: str) -> bool:
        result = value.get(name)
        if not isinstance(result, bool):
            raise RuntimeError(f"Emulator response field {name!r} must be a boolean")
        return result

    @staticmethod
    def _git_revision(repository: Path) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return result.stdout.strip() or "unknown"
