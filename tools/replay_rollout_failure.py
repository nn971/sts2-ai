#!/usr/bin/env python3
"""Replay the exact player-visible action path from a failed self-play run.

Example (fish):
    python tools/replay_rollout_failure.py --failure results/native-9700x-report.failure.json

The emulator is reset with the recorded seed and receives the same legal
action IDs in order. No learned policy, oracle state or RNG injection is used.
Exit status 0 means the *same* emulator step error reproduced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sts2_ai.emulator import InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.protocol import EmulatorBackend
from sts2_ai.emulator.run_environment import reset_training_run
from sts2_ai.training.rollout_failure import SCHEMA


def replay_public_failure(
    backend: EmulatorBackend, record: dict[str, Any],
) -> dict[str, Any]:
    if record.get("schema") != SCHEMA:
        raise ValueError("Unsupported public rollout failure record schema")
    actions = record.get("chosen_action_ids")
    if not isinstance(actions, list) or not actions or any(
        not isinstance(item, str) for item in actions
    ):
        raise ValueError("Rollout failure has no valid public action path")
    if record.get("decision_index") != len(actions) - 1:
        raise ValueError("Rollout failure action history has an invalid length")

    policy = InformationPolicy(record["policy_id"])
    state = reset_training_run(backend, record["seed"], record["environment"])
    try:
        for index, action_id in enumerate(actions):
            observation = backend.observe(state, policy)
            if index == len(actions) - 1 and (
                observation.observation_hash
                != record["failing_observation_hash"]
            ):
                raise ValueError(
                    f"Replay diverged at failing public observation: "
                    f"decision={index}"
                )
            legal = tuple(backend.legal_actions(state))
            matches = [action for action in legal if action.action_id == action_id]
            if len(matches) != 1:
                raise ValueError(
                    f"Replay diverged at decision={index}: "
                    f"action ID {action_id!r} is absent or ambiguous"
                )
            try:
                next_state = backend.step(state, matches[0]).child
            except Exception as exc:
                if index != len(actions) - 1:
                    raise ValueError(
                        f"Replay crashed earlier than recorded: decision={index}"
                    ) from exc
                reproduced = (
                    type(exc).__name__ == record["error_type"]
                    and str(exc) == record["error_message"]
                )
                return {
                    "reproduced": reproduced,
                    "decision_index": index,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            previous = state
            state = next_state
            backend.release_many((previous,))
        return {
            "reproduced": False,
            "decision_index": len(actions) - 1,
            "reason": "Recorded failing transition now succeeds",
        }
    finally:
        backend.release_many((state,))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    record = json.loads(args.failure.read_text())
    if not isinstance(record, dict):
        parser.error("Failure report must be a JSON object")
    with JsonlEmulatorBackend(build=args.build) as backend:
        if record.get("emulator_revision") != backend.emulator_revision:
            parser.error(
                "Emulator revision differs from failure report. "
                "Replay with the original pinned submodule first."
            )
        result = replay_public_failure(backend, record)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["reproduced"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
