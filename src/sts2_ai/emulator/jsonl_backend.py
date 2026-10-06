from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from .protocol import (
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
    Transition,
)


class JsonlPrototypeBackend:
    """Developer bridge to the C# prototype emulator.

    This transport is intentionally simple and correctness-oriented. It keeps one long-lived
    emulator CLI process and exchanges one JSON object per line. A future native binding should
    implement the same EmulatorBackend protocol without changing search/agent code.
    """

    def __init__(
        self,
        emulator_root: Path,
        *,
        configuration: str = "Release",
        build: bool = True,
    ) -> None:
        self._emulator_root = emulator_root.resolve()
        self._configuration = configuration
        self._next_request_id = 1
        self._closed = False

        project = self._emulator_root / "src" / "Sts2Emulator.Cli" / "Sts2Emulator.Cli.csproj"
        if not project.is_file():
            raise FileNotFoundError(f"Emulator CLI project not found: {project}")

        if build:
            subprocess.run(
                ["dotnet", "build", str(project), "-c", configuration, "--nologo"],
                check=True,
                capture_output=True,
                text=True,
            )

        bin_root = project.parent / "bin" / configuration
        candidates = sorted(bin_root.glob("*/Sts2Emulator.Cli.dll"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one built emulator CLI under {bin_root}, found {len(candidates)}"
            )

        self._process = subprocess.Popen(
            ["dotnet", str(candidates[0]), "prototype-ai-jsonl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        hello = self._call({"op": "hello"})
        self._binding_version = _required_str(hello, "wireSchemaId")
        self._ai_schema_id = _required_str(hello, "aiSchemaId")
        self._ruleset_id = _required_str(hello, "rulesetId")
        self._fair_policy_id = _required_str(hello, "fairPolicyId")
        self._emulator_revision = _git_revision(self._emulator_root)

    @property
    def binding_version(self) -> str:
        return self._binding_version

    @property
    def emulator_revision(self) -> str:
        return self._emulator_revision

    @property
    def ai_schema_id(self) -> str:
        return self._ai_schema_id

    @property
    def ruleset_id(self) -> str:
        return self._ruleset_id

    @property
    def fair_policy_id(self) -> str:
        return self._fair_policy_id

    def reset(self, seed: str) -> StateHandle:
        response = self._call({"op": "reset", "seed": seed})
        return _required_str(response, "stateHandle")

    def legal_actions(self, state: StateHandle) -> Sequence[LegalAction]:
        response = self._call(
            {
                "op": "legal_actions",
                "state_handle": state,
            }
        )
        raw_actions = response.get("actions")
        if not isinstance(raw_actions, list):
            raise RuntimeError("Bridge legal_actions response has no action list")
        return tuple(_parse_action(item) for item in raw_actions)

    def step(self, state: StateHandle, action: LegalAction) -> Transition:
        response = self._call(
            {
                "op": "step",
                "state_handle": state,
                "action_id": action.action_id,
            }
        )
        raw_action = response.get("action")
        if not isinstance(raw_action, dict):
            raise RuntimeError("Bridge step response has no action")
        returned_action = _parse_action(raw_action)
        if returned_action.action_id != action.action_id:
            raise RuntimeError(
                "Bridge returned a different action ID than the requested transition"
            )

        return Transition(
            parent=_required_str(response, "parent"),
            action=returned_action,
            child=_required_str(response, "child"),
            terminal=_required_bool(response, "terminal"),
        )

    def fork(self, state: StateHandle) -> StateHandle:
        response = self._call(
            {
                "op": "fork",
                "state_handle": state,
            }
        )
        return _required_str(response, "child")

    def exact_hash(self, state: StateHandle) -> str:
        response = self._call(
            {
                "op": "exact_hash",
                "state_handle": state,
            }
        )
        return _required_str(response, "exactHash")

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation:
        response = self._call(
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
        response = self._call(
            {
                "op": "is_terminal",
                "state_handle": state,
            }
        )
        return _required_bool(response, "terminal")

    def capability_manifest(self) -> dict[str, Any]:
        response = self._call({"op": "manifest"})
        manifest = response.get("manifest")
        if not isinstance(manifest, dict):
            raise RuntimeError("Bridge manifest response has no manifest object")
        return manifest

    def close(self) -> None:
        if self._closed:
            return

        try:
            if self._process.poll() is None:
                self._call({"op": "close"})
                self._process.wait(timeout=5)
        finally:
            if self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=5)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _call(self, payload: dict[str, object]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Prototype emulator bridge is closed")
        if self._process.poll() is not None:
            raise RuntimeError(self._dead_process_message())

        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise RuntimeError("Prototype emulator bridge pipes are unavailable")

        request = dict(payload)
        request["request_id"] = str(self._next_request_id)
        self._next_request_id += 1

        stdin.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
        stdin.flush()

        line = stdout.readline()
        if not line:
            raise RuntimeError(self._dead_process_message())

        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise RuntimeError("Prototype emulator bridge returned a non-object response")
        response: dict[str, Any] = raw
        if response.get("ok") is not True:
            error_type = response.get("errorType", "BridgeError")
            error = response.get("error", "unknown bridge error")
            raise RuntimeError(f"{error_type}: {error}")
        return response

    def _dead_process_message(self) -> str:
        stderr = self._process.stderr
        detail = ""
        if stderr is not None and self._process.poll() is not None:
            detail = stderr.read().strip()
        return (
            f"Prototype emulator bridge exited with code {self._process.poll()}"
            + (f": {detail}" if detail else "")
        )


def _parse_action(raw: object) -> LegalAction:
    if not isinstance(raw, dict):
        raise RuntimeError("Bridge action is not an object")
    return LegalAction(
        action_id=_required_str(raw, "actionId"),
        kind=_required_str(raw, "kind"),
        payload_json=_required_str(raw, "payloadJson"),
    )


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Bridge field '{key}' is not a string")
    return value


def _required_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"Bridge field '{key}' is not a boolean")
    return value


def _git_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"
