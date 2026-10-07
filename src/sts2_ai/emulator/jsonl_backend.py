from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from .protocol import (
    InformationPolicy,
    LegalAction,
    Observation,
    StateHandle,
    Transition,
)

WIRE_SCHEMA_ID = "prototype-ai-jsonl-v0"
OBSERVATION_SCHEMA_ID = "prototype-ai-v0"
FAIR_POLICY_ID = "prototype-fair-v0"
RULESET_ID = "prototype-silent-v0"


class JsonlBridgeError(RuntimeError):
    """Raised when the emulator JSONL bridge rejects or violates a request."""


@dataclass(frozen=True, slots=True)
class BridgeOperationStats:
    calls: int
    total_seconds: float

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0


def _run_git(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise JsonlBridgeError(f"Unable to inspect git revision under {root}") from exc
    return completed.stdout.strip()


def _pinned_emulator_revision(repo_root: Path) -> str:
    line = _run_git(repo_root, "ls-tree", "HEAD", "emulator")
    fields = line.split()
    if len(fields) < 3 or fields[1] != "commit":
        raise JsonlBridgeError(
            "Repository HEAD does not contain an emulator gitlink; initialize from sts2-ai."
        )
    return fields[2]


class JsonlEmulatorBackend:
    """Long-lived Python adapter for the prototype-ai-jsonl emulator bridge."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        emulator_root: Path | None = None,
        expected_emulator_revision: str | None = None,
        dotnet: str = "dotnet",
        build: bool = True,
    ) -> None:
        self._repo_root = (
            repo_root.resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[3]
        )
        self._emulator_root = (
            emulator_root.resolve()
            if emulator_root is not None
            else self._repo_root / "emulator"
        )
        self._project = self._emulator_root / "src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj"
        if not self._project.is_file():
            raise JsonlBridgeError(
                f"Emulator submodule is unavailable at {self._emulator_root}. "
                "Run git submodule update --init --recursive."
            )

        pinned = (
            expected_emulator_revision
            if expected_emulator_revision is not None
            else _pinned_emulator_revision(self._repo_root)
        )
        actual = _run_git(self._emulator_root, "rev-parse", "HEAD")
        if actual != pinned:
            raise JsonlBridgeError(
                "Emulator checkout does not match the sts2-ai pin: "
                f"expected {pinned}, found {actual}."
            )
        self._emulator_revision = actual
        self._dotnet = dotnet
        self._next_request_id = 1
        self._closed = False
        self._operation_profile: dict[str, tuple[int, float]] = {}

        if build:
            try:
                subprocess.run(
                    [
                        self._dotnet,
                        "build",
                        str(self._project),
                        "--configuration",
                        "Release",
                        "--nologo",
                        "--verbosity",
                        "quiet",
                    ],
                    cwd=self._emulator_root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                detail = ""
                if isinstance(exc, subprocess.CalledProcessError):
                    detail = (exc.stderr or exc.stdout or "").strip()
                suffix = f": {detail}" if detail else ""
                raise JsonlBridgeError(f"Failed to build emulator bridge{suffix}") from exc

        try:
            self._process = subprocess.Popen(
                [
                    self._dotnet,
                    "run",
                    "--project",
                    str(self._project),
                    "--configuration",
                    "Release",
                    "--no-build",
                    "--no-restore",
                    "--",
                    "prototype-ai-jsonl",
                ],
                cwd=self._emulator_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise JsonlBridgeError("Failed to start emulator JSONL bridge") from exc

        if self._process.stdin is None or self._process.stdout is None:
            self._terminate()
            raise JsonlBridgeError("Emulator bridge started without stdin/stdout pipes")

        hello = self._request("hello")
        expected = {
            "wireSchemaId": WIRE_SCHEMA_ID,
            "aiSchemaId": OBSERVATION_SCHEMA_ID,
            "rulesetId": RULESET_ID,
            "fairPolicyId": FAIR_POLICY_ID,
        }
        for key, value in expected.items():
            if hello.get(key) != value:
                self._terminate()
                raise JsonlBridgeError(
                    f"Bridge handshake mismatch for {key}: "
                    f"expected {value!r}, found {hello.get(key)!r}"
                )

        manifest = self._request("manifest").get("manifest")
        if not isinstance(manifest, dict):
            self._terminate()
            raise JsonlBridgeError("Bridge manifest response is malformed")
        if manifest.get("rulesetId") != RULESET_ID:
            self._terminate()
            raise JsonlBridgeError("Bridge manifest ruleset does not match handshake")
        if manifest.get("aiSchemaId") != OBSERVATION_SCHEMA_ID:
            self._terminate()
            raise JsonlBridgeError("Bridge manifest observation schema does not match handshake")
        self._manifest = cast(dict[str, Any], manifest)

    @property
    def binding_version(self) -> str:
        return WIRE_SCHEMA_ID

    @property
    def emulator_revision(self) -> str:
        return self._emulator_revision

    @property
    def capability_manifest(self) -> Mapping[str, Any]:
        return self._manifest

    def operation_profile(self) -> Mapping[str, BridgeOperationStats]:
        return {
            operation: BridgeOperationStats(calls=calls, total_seconds=total_seconds)
            for operation, (calls, total_seconds) in sorted(self._operation_profile.items())
        }

    def reset_operation_profile(self) -> None:
        self._operation_profile.clear()

    def reset(self, seed: str, ascension: int = 0) -> StateHandle:
        response = self._request("reset", seed=seed, ascension=ascension)
        return self._require_string(response, "stateHandle")

    def legal_actions(self, state: StateHandle) -> tuple[LegalAction, ...]:
        response = self._request("legal_actions", state_handle=state)
        raw_actions = response.get("actions")
        if not isinstance(raw_actions, list):
            raise JsonlBridgeError("legal_actions response has no actions array")
        return tuple(self._parse_action(item) for item in raw_actions)

    def step(self, state: StateHandle, action: LegalAction) -> Transition:
        response = self._request(
            "step",
            state_handle=state,
            action_id=action.action_id,
        )
        transition = self._parse_transition(response)
        if transition.parent != state:
            raise JsonlBridgeError("step response parent does not match request")
        if transition.action.action_id != action.action_id:
            raise JsonlBridgeError("step response action does not match request")
        return transition

    def batch_step(
        self,
        items: Sequence[tuple[StateHandle, LegalAction]],
    ) -> tuple[Transition, ...]:
        requested = list(items)
        response = self._request(
            "batch_step",
            items=[
                {"state_handle": state, "action_id": action.action_id}
                for state, action in requested
            ],
        )
        raw_transitions = response.get("transitions")
        if not isinstance(raw_transitions, list) or len(raw_transitions) != len(requested):
            raise JsonlBridgeError("batch_step response shape does not match request")

        transitions = tuple(self._parse_transition(item) for item in raw_transitions)
        for (expected_parent, expected_action), transition in zip(
            requested,
            transitions,
            strict=True,
        ):
            if transition.parent != expected_parent:
                raise JsonlBridgeError("batch_step parent order does not match request")
            if transition.action.action_id != expected_action.action_id:
                raise JsonlBridgeError("batch_step action order does not match request")
        return transitions

    def fork(self, state: StateHandle) -> StateHandle:
        response = self._request("fork", state_handle=state)
        return self._require_string(response, "child")

    def expand(self, state: StateHandle) -> tuple[Transition, ...]:
        response = self._request("expand", state_handle=state)
        raw = response.get("expansions")
        if not isinstance(raw, list):
            raise JsonlBridgeError("expand response has no expansions array")
        transitions = tuple(self._parse_transition(item) for item in raw)
        if any(item.parent != state for item in transitions):
            raise JsonlBridgeError("expand response contains the wrong parent")
        return transitions

    def batch_expand(
        self,
        states: Sequence[StateHandle],
    ) -> tuple[tuple[Transition, ...], ...]:
        requested = list(states)
        response = self._request("batch_expand", state_handles=requested)
        raw_batches = response.get("batches")
        if not isinstance(raw_batches, list) or len(raw_batches) != len(requested):
            raise JsonlBridgeError("batch_expand response shape does not match request")

        result: list[tuple[Transition, ...]] = []
        for expected_parent, raw_batch in zip(requested, raw_batches, strict=True):
            if not isinstance(raw_batch, dict):
                raise JsonlBridgeError("batch_expand batch is not an object")
            if raw_batch.get("parent") != expected_parent:
                raise JsonlBridgeError("batch_expand parent order does not match request")
            raw_expansions = raw_batch.get("expansions")
            if not isinstance(raw_expansions, list):
                raise JsonlBridgeError("batch_expand batch has no expansions array")
            transitions = tuple(self._parse_transition(item) for item in raw_expansions)
            if any(item.parent != expected_parent for item in transitions):
                raise JsonlBridgeError("batch_expand contains the wrong parent")
            result.append(transitions)
        return tuple(result)

    def release_many(self, states: Sequence[StateHandle]) -> int:
        response = self._request("release_many", state_handles=list(states))
        released = response.get("released")
        if not isinstance(released, int):
            raise JsonlBridgeError("release_many response has no integer released count")
        return released

    def exact_hash(self, state: StateHandle) -> str:
        response = self._request("exact_hash", state_handle=state)
        return self._require_string(response, "exactHash")

    def observe(self, state: StateHandle, policy: InformationPolicy) -> Observation:
        response = self._request(
            "observe",
            state_handle=state,
            policy_id=policy.policy_id,
        )
        schema_id = self._require_string(response, "schemaId")
        if schema_id != OBSERVATION_SCHEMA_ID:
            raise JsonlBridgeError(
                f"Unexpected observation schema {schema_id!r}; "
                f"expected {OBSERVATION_SCHEMA_ID!r}"
            )
        response_policy = self._require_string(response, "policyId")
        if response_policy != policy.policy_id:
            raise JsonlBridgeError("observe response information policy does not match request")
        return Observation(
            policy_id=response_policy,
            payload_json=self._require_string(response, "payloadJson"),
            observation_hash=self._require_string(response, "observationHash"),
        )

    def batch_observe(
        self,
        states: Sequence[StateHandle],
        policy: InformationPolicy,
    ) -> tuple[Observation, ...]:
        requested = list(states)
        response = self._request(
            "batch_observe",
            state_handles=requested,
            policy_id=policy.policy_id,
        )
        raw_observations = response.get("observations")
        if not isinstance(raw_observations, list) or len(raw_observations) != len(requested):
            raise JsonlBridgeError("batch_observe response shape does not match request")

        observations: list[Observation] = []
        for expected_state, raw in zip(requested, raw_observations, strict=True):
            if not isinstance(raw, dict):
                raise JsonlBridgeError("batch_observe item is not an object")
            if raw.get("stateHandle") != expected_state:
                raise JsonlBridgeError("batch_observe state order does not match request")
            schema_id = self._require_string(raw, "schemaId")
            if schema_id != OBSERVATION_SCHEMA_ID:
                raise JsonlBridgeError(
                    f"Unexpected observation schema {schema_id!r}; "
                    f"expected {OBSERVATION_SCHEMA_ID!r}"
                )
            response_policy = self._require_string(raw, "policyId")
            if response_policy != policy.policy_id:
                raise JsonlBridgeError(
                    "batch_observe information policy does not match request"
                )
            observations.append(
                Observation(
                    policy_id=response_policy,
                    payload_json=self._require_string(raw, "payloadJson"),
                    observation_hash=self._require_string(raw, "observationHash"),
                )
            )
        return tuple(observations)

    def is_terminal(self, state: StateHandle) -> bool:
        response = self._request("is_terminal", state_handle=state)
        terminal = response.get("terminal")
        if not isinstance(terminal, bool):
            raise JsonlBridgeError("is_terminal response has no boolean terminal field")
        return terminal

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                self._request("close")
                self._process.wait(timeout=5)
        except (JsonlBridgeError, subprocess.TimeoutExpired):
            self._terminate()
        finally:
            self._closed = True

    def __enter__(self) -> JsonlEmulatorBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _request(self, op: str, **payload: object) -> dict[str, Any]:
        if self._closed:
            raise JsonlBridgeError("Emulator bridge is already closed")
        if self._process.poll() is not None:
            raise JsonlBridgeError(self._dead_process_message())

        started = time.perf_counter()
        request_id = str(self._next_request_id)
        self._next_request_id += 1
        request: dict[str, object] = {
            "request_id": request_id,
            "op": op,
            **payload,
        }
        stdin = self._process.stdin
        stdout = self._process.stdout
        if stdin is None or stdout is None:
            raise JsonlBridgeError("Emulator bridge pipes are unavailable")

        try:
            stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            stdin.flush()
            line = stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise JsonlBridgeError(self._dead_process_message()) from exc

        if not line:
            raise JsonlBridgeError(self._dead_process_message())

        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JsonlBridgeError(f"Bridge emitted non-JSON output: {line.rstrip()!r}") from exc
        if not isinstance(raw, dict):
            raise JsonlBridgeError("Bridge response is not a JSON object")
        response = cast(dict[str, Any], raw)
        if response.get("requestId") != request_id:
            raise JsonlBridgeError(
                "Bridge response requestId mismatch: "
                f"expected {request_id!r}, found {response.get('requestId')!r}"
            )
        if response.get("ok") is not True:
            error_type = response.get("errorType", "BridgeError")
            error = response.get("error", "unknown bridge failure")
            raise JsonlBridgeError(f"{error_type}: {error}")

        elapsed = time.perf_counter() - started
        calls, total_seconds = self._operation_profile.get(op, (0, 0.0))
        self._operation_profile[op] = (calls + 1, total_seconds + elapsed)
        return response

    def _dead_process_message(self) -> str:
        detail = ""
        stderr = self._process.stderr
        if stderr is not None and self._process.poll() is not None:
            detail = stderr.read().strip()
        suffix = f": {detail}" if detail else ""
        return f"Emulator JSONL bridge exited unexpectedly{suffix}"

    def _terminate(self) -> None:
        if hasattr(self, "_process") and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)

    @staticmethod
    def _require_string(response: Mapping[str, Any], key: str) -> str:
        value = response.get(key)
        if not isinstance(value, str):
            raise JsonlBridgeError(f"Bridge response field {key!r} is not a string")
        return value

    @staticmethod
    def _parse_action(raw: object) -> LegalAction:
        if not isinstance(raw, dict):
            raise JsonlBridgeError("Bridge action is not an object")
        action_id = raw.get("actionId")
        kind = raw.get("kind")
        payload_json = raw.get("payloadJson")
        if not isinstance(action_id, str) or not isinstance(kind, str):
            raise JsonlBridgeError("Bridge action is missing actionId/kind")
        if not isinstance(payload_json, str):
            raise JsonlBridgeError("Bridge action payloadJson is not a string")
        return LegalAction(
            action_id=action_id,
            kind=kind,
            payload_json=payload_json,
        )

    @classmethod
    def _parse_transition(cls, raw: object) -> Transition:
        if not isinstance(raw, dict):
            raise JsonlBridgeError("Bridge transition is not an object")
        parent = raw.get("parent")
        child = raw.get("child")
        terminal = raw.get("terminal")
        exact_hash = raw.get("exactHash")
        if not isinstance(parent, str) or not isinstance(child, str):
            raise JsonlBridgeError("Bridge transition is missing parent/child")
        if not isinstance(terminal, bool):
            raise JsonlBridgeError("Bridge transition terminal is not boolean")
        if exact_hash is not None and not isinstance(exact_hash, str):
            raise JsonlBridgeError("Bridge transition exactHash is not a string")
        return Transition(
            parent=parent,
            action=cls._parse_action(raw.get("action")),
            child=child,
            terminal=terminal,
            exact_hash=exact_hash,
        )
