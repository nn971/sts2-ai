#!/usr/bin/env python3
"""Read-only P2.1 preflight: exact root-conditioned fork/prefix replay.

This probes a fixed Native RNG *root-conditioned* continuation. It creates two
independent exact forks for each traversed prefix and applies the same legal
action to both. It verifies canonical exact state hash, public observation,
legal menu and terminal state before proceeding. Handles are released even
when a check fails. It NEVER retains a continuation for another Nature draw,
never caches a node, and does not certify a standalone RNG receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy
from sts2_ai.emulator.jsonl_backend import JsonlEmulatorBackend
from sts2_ai.emulator.protocol import EmulatorBackend, StateHandle
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run

if __package__:
    from .profile_p2_joint_throughput import ROOT, dll_identity, git_head
else:
    from profile_p2_joint_throughput import ROOT, dll_identity, git_head

SCHEMA = "sts2-p2-root-conditioned-exact-prefix-preflight-v1"
POLICY = InformationPolicy(FAIR_POLICY_ID)


def _frame_signature(backend: EmulatorBackend, state: StateHandle) -> tuple:
    observation = backend.observe(state, POLICY)
    actions = tuple(backend.legal_actions(state))
    if observation.policy_id != FAIR_POLICY_ID:
        raise ValueError("Prefix observation changed public information policy")
    return observation, actions, backend.is_terminal(state)


def verify_exact_fork_prefix(
    backend: EmulatorBackend, root: StateHandle, *, max_steps: int = 8,
) -> dict[str, Any]:
    """Verify duplicate exact forks and steps; never use a public hash as cache key."""
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    owned: list[StateHandle] = []
    visited: list[dict[str, Any]] = []
    current = root
    released = 0
    try:
        for depth in range(max_steps):
            root_hash = backend.exact_hash(current)
            if not isinstance(root_hash, str) or not root_hash:
                raise ValueError("Missing canonical exact state hash")
            root_signature = _frame_signature(backend, current)
            if root_signature[2]:
                break
            if not root_signature[1]:
                raise ValueError("Nonterminal exact prefix has no legal actions")

            left = backend.fork(current)
            owned.append(left)
            right = backend.fork(current)
            owned.append(right)
            if len({left, right, current}) != 3:
                raise ValueError("Exact fork handles are not distinct")
            for handle in (left, right):
                if backend.exact_hash(handle) != root_hash:
                    raise ValueError(
                        f"Exact state/RNG prefix hash diverged on fork at depth={depth}"
                    )
                if _frame_signature(backend, handle) != root_signature:
                    raise ValueError(
                        f"Public frame or legal menu diverged on fork at depth={depth}"
                    )

            action = root_signature[1][0]  # deterministic legal-menu index 0
            steps = backend.batch_rollout_step_frame(
                ((left, action), (right, action)), POLICY,
            )
            if len(steps) != 2:
                raise ValueError("Two-fork step returned wrong number of frames")
            # Track both new handles *before* any later validation can raise.
            for item in steps:
                owned.append(item.transition.child)
            first, second = steps
            for expected, item in ((left, first), (right, second)):
                if item.transition.parent != expected or item.transition.action != action:
                    raise ValueError("Fused exact-fork step changed action/parent")
                if item.observation.policy_id != FAIR_POLICY_ID:
                    raise ValueError("Fused exact-fork step changed public policy")
            hash_a = backend.exact_hash(first.transition.child)
            hash_b = backend.exact_hash(second.transition.child)
            if hash_a != hash_b:
                raise ValueError(
                    f"Exact state/RNG prefix hash diverged after action at depth={depth}"
                )
            if (
                first.observation != second.observation
                or first.legal_actions != second.legal_actions
                or first.transition.terminal != second.transition.terminal
            ):
                raise ValueError(
                    f"Public continuation or legal actions diverged at depth={depth}"
                )
            if _frame_signature(backend, first.transition.child) != (
                first.observation, first.legal_actions,
                first.transition.terminal,
            ):
                raise ValueError("Fused step frame differs from independent observation")

            visited.append({
                "depth": depth + 1,
                "action_id": action.action_id,
                "parent_exact_hash": root_hash,
                "child_exact_hash": hash_a,
                "public_observation_hash": first.observation.observation_hash,
                "legal_action_count": len(first.legal_actions),
                "terminal": first.transition.terminal,
            })
            current = first.transition.child
            if first.transition.terminal:
                break
    finally:
        if owned:
            released = backend.release_many(owned)
            if released != len(owned):
                raise RuntimeError(
                    f"Exact-prefix probe leaked handles: "
                    f"allocated={len(owned)}, released={released}"
                )
    return {
        "schema": SCHEMA,
        "capability": "h16-native-act1-root-conditioned-privileged-v1",
        "scope": "same deterministic exact-root-conditioned native continuation",
        "steps_checked": len(visited),
        "forks_checked": 2 * len(visited),
        "owned_handles_released": released,
        "verified": True,
        "independent_rng_receipt_available": False,
        "continuation_cache_enabled": False,
        "ready_for_p2_1_caching": False,
        "trace": visited,
        "limitations": [
            "Canonical exact-state hashes are checked but a separately exposed "
            "RNG/Nature receipt has not been certified",
            "Exact forks from one root do not license reuse across independently "
            "sampled Nature/fair-history continuations",
            "Only a bounded single deterministic first-legal-action path is covered",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--seed", default="p2-native-exact-prefix-20261104-0")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not a.execute:
        raise SystemExit("Pass --execute to run the native exact-prefix preflight")
    if a.output.exists():
        raise SystemExit(f"Refusing to overwrite preflight evidence: {a.output}")
    if git_head(a.emulator_root) != a.emulator_revision:
        raise SystemExit("Emulator source SHA mismatch")
    dll = dll_identity(a.emulator_root)
    if dll is None:
        raise SystemExit("Missing unique compiled Release emulator DLL")

    with JsonlEmulatorBackend(
        repo_root=ROOT, emulator_root=a.emulator_root,
        expected_emulator_revision=a.emulator_revision, build=False,
    ) as backend:
        root = reset_training_run(
            backend, a.seed, NATIVE_OVERGROWTH, ascension=a.ascension,
        )
        try:
            result = verify_exact_fork_prefix(
                backend, root, max_steps=a.max_steps,
            )
        finally:
            if backend.release_many([root]) != 1:
                raise RuntimeError("P2 exact-prefix preflight did not release root")
    result["source"] = {
        "ai_sha": git_head(ROOT), "emulator_sha": a.emulator_revision,
        "emulator_cli": dll, "seed_sha256": hashlib.sha256(
            a.seed.encode("utf-8")
        ).hexdigest(),
        "ascension": a.ascension,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "output": str(a.output), "verified": result["verified"],
        "steps_checked": result["steps_checked"],
        "ready_for_p2_1_caching": result["ready_for_p2_1_caching"],
    }, indent=2))


if __name__ == "__main__":
    main()
