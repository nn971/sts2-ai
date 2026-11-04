#!/usr/bin/env python3
"""Protocol-only v1 JSONL P2 profiler smoke; no H16/v0 neural inference.

Exercises the P1 power-refactor emulator's v1 public frames using a direct
versioned JSONL client. Runs the same exact rooted action once with counters
disabled and once enabled. H16 v0 training remains a separate matched control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "sts2-p2-v1-native-transport-smoke-v1"
P2_NATIVE_SCHEMA = "sts2-p2-jsonl-performance-v1"
POLICY = "prototype-fair-v0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RawJsonl:
    def __init__(self, emulator_root: Path) -> None:
        project = emulator_root / "src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj"
        self.process = subprocess.Popen(
            ["dotnet", "run", "--project", str(project),
             "-c", "Release", "--no-build", "--no-restore",
             "--", "prototype-ai-jsonl"],
            cwd=emulator_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.next_id = 0

    def request(self, op: str, **payload: object) -> dict[str, Any]:
        self.next_id += 1
        request_id = str(self.next_id)
        data = {"request_id": request_id, "op": op, **payload}
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(json.dumps(data, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(f"JSONL server terminated during {op}")
        response = json.loads(response_line)
        if (not isinstance(response, dict) or response.get("requestId") != request_id
                or response.get("ok") is not True):
            raise RuntimeError(f"JSONL {op} failed: {response}")
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.request("close")
                self.process.wait(timeout=5)
            except BaseException:
                self.process.kill()
                self.process.wait(timeout=5)
                raise


def run_root(client: RawJsonl, seed: str, *, profile: bool) -> dict[str, Any]:
    if profile:
        reset = client.request("p2_perf_reset")
        if reset.get("schemaId") != P2_NATIVE_SCHEMA:
            raise RuntimeError("P2 native profiler schema mismatch")
    started = time.perf_counter()
    root = client.request(
        "reset_native_overgrowth", seed=seed, ascension=0,
        compact_card_selection=True,
    )["stateHandle"]
    exact = client.request("exact_hash", state_handle=root)["exactHash"]
    clone = client.request("fork", state_handle=root)
    fork = clone["child"]
    if clone["exactHash"] != exact:
        raise AssertionError("Fork changed exact state hash")
    actions = client.request("legal_actions", state_handle=root)["actions"]
    if not actions:
        raise RuntimeError("No legal native root actions")
    action_id = actions[0]["actionId"]
    descendants = []
    public_frames = []
    try:
        for parent in (root, fork):
            (frame,) = client.request(
                "batch_rollout_step_frame", policy_id=POLICY,
                items=[{"state_handle": parent, "action_id": action_id}],
            )["frames"]
            child = frame["child"]
            descendants.append(child)
            full_hash = client.request("exact_hash", state_handle=child)["exactHash"]
            public_frames.append({
                "exact_hash": full_hash,
                "observation": frame["payloadJson"],
                "legal_action_ids": [
                    a["actionId"] for a in frame["legalActions"]
                ],
                "terminal": frame["terminal"],
            })
        if public_frames[0] != public_frames[1]:
            raise AssertionError("Native fork + step public/exact parity failed")
    finally:
        released = client.request(
            "release_many", state_handles=[root, fork, *descendants],
        )["released"]
        if released != 2 + len(descendants):
            raise RuntimeError("P2 native protocol leaked a retained state handle")
    elapsed = time.perf_counter() - started
    native = None
    if profile:
        stopped = client.request("p2_perf_stop")
        native = stopped["profile"]
        if native.get("schemaId") != P2_NATIVE_SCHEMA:
            raise RuntimeError("P2 stop response schema mismatch")
        if stopped["liveStateHandles"] != 0 or stopped["peakStateHandles"] < 4:
            raise AssertionError("P2 live/peak handle accounting failed")
        operations = native["operations"]
        for op in (
            "p2.fork.run_state", "p2.fork.exact_hash",
            "p2.rollout.step_and_public_frame",
            "p2.rollout.serialize_observation",
            "p2.rollout.serialize_legal_actions",
            "p2.jsonl.op.fork", "p2.jsonl.response",
        ):
            if op not in operations or operations[op]["count"] < 1:
                raise AssertionError(f"Missing native P2 profiler stage: {op}")
    return {
        "elapsed_wall_seconds": elapsed,
        "seed": seed,
        "root_exact_hash": exact,
        "action_id": action_id,
        "child_exact_hash": public_frames[0]["exact_hash"],
        "public_observation": public_frames[0]["observation"],
        "legal_action_ids": public_frames[0]["legal_action_ids"],
        "terminal": public_frames[0]["terminal"],
        "native_profiler": native,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulator-root", type=Path, required=True)
    parser.add_argument("--emulator-revision", required=True)
    parser.add_argument("--seed", default="p2-native-v1-transport-smoke-0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.emulator_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != args.emulator_revision:
        raise SystemExit("Emulator SHA mismatch; native smoke must be pinned")
    client = RawJsonl(args.emulator_root.resolve())
    try:
        hello = client.request("hello")
        if hello.get("aiSchemaId") != "prototype-ai-v1":
            raise RuntimeError("P1 native profile requires public-v1 emulator")
        control = run_root(client, args.seed, profile=False)
        instrumented = run_root(client, args.seed, profile=True)
        for key in ("root_exact_hash", "action_id", "child_exact_hash",
                    "public_observation", "legal_action_ids", "terminal"):
            if control[key] != instrumented[key]:
                raise AssertionError(f"P2 profile-on/off parity failed: {key}")
    finally:
        client.close()
    dlls = list((args.emulator_root / "src/Sts2Emulator.Cli/bin/Release").glob(
        "*/Sts2Emulator.Cli.dll",
    ))
    if len(dlls) != 1:
        raise RuntimeError("Expected exactly one pinned Release CLI DLL")
    report = {
        "schema": SCHEMA,
        "evidence": "v1 JSONL transport-only and allocation smoke; not H16 search",
        "identity": {
            "emulator_sha": revision,
            "cli_sha256": sha256(dlls[0]),
            "public_observation_schema": "prototype-ai-v1",
            "seed": args.seed,
            "ascension": 0,
            "map": "generated native-structure Overgrowth",
            "rng": "emulator-native-structure experimental",
        },
        "controls": {"off": control, "on": instrumented},
        "public_and_exact_state_parity": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "parity": True,
        "measured_ops": len(instrumented["native_profiler"]["operations"]),
    }))


if __name__ == "__main__":
    main()
