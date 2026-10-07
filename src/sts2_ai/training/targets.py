from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from sts2_ai.strategy_db import (
    SearchActionEvidence,
    SearchObservationEvidence,
    SearchRootEvidence,
)


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    action_id: str
    probability: float
    action_kind: str = ""
    action_payload_json: str = "{}"
    search_value: float = 0.0
    visits: int = 0


@dataclass(frozen=True, slots=True)
class TrainingExample:
    observation_hash: str
    information_policy: str
    policy_targets: tuple[PolicyTarget, ...]
    value_target: float
    source_search_id: str
    emulator_revision: str
    observation_json: str = "{}"
    source_state_hash: str = ""
    search_regime: str = "oracle-exact"
    search_budget: int = 0
    search_version: str = ""
    game_build: str = "unknown"


def build_training_example(
    root: SearchRootEvidence,
    actions: tuple[SearchActionEvidence, ...],
    observation: SearchObservationEvidence,
) -> TrainingExample:
    """Distill one searched root into a supervised policy/value example.

    The policy target is the normalized root visit distribution. The value target is
    the best visited root-action mean, matching the current oracle agent's decision
    rule while remaining well-defined for historical records whose chosen action may
    have been unvisited.
    """

    if observation.observation_hash != root.observation_hash:
        raise ValueError("Root and observation hashes do not match")
    if observation.information_policy != root.information_policy:
        raise ValueError("Root and observation information policies do not match")
    if not actions:
        raise ValueError("A training example requires root-action evidence")

    by_id = {action.action_id: action for action in actions}
    if set(by_id) != set(root.legal_action_ids):
        raise ValueError("Action evidence does not match the root legal-action set")

    total_visits = sum(max(0, action.visits) for action in actions)
    uniform = 1.0 / len(actions)
    targets = tuple(
        PolicyTarget(
            action_id=action.action_id,
            probability=(
                max(0, action.visits) / total_visits
                if total_visits > 0
                else uniform
            ),
            action_kind=action.action_kind,
            action_payload_json=action.action_payload_json,
            search_value=action.value,
            visits=action.visits,
        )
        for action in sorted(actions, key=lambda item: item.action_id)
    )

    visited = [action for action in actions if action.visits > 0]
    value_candidates = visited or list(actions)
    value_target = max(action.value for action in value_candidates)

    source_search_id = _source_search_id(root)
    return TrainingExample(
        observation_hash=root.observation_hash,
        information_policy=root.information_policy,
        policy_targets=targets,
        value_target=value_target,
        source_search_id=source_search_id,
        emulator_revision=root.emulator_revision,
        observation_json=observation.payload_json,
        source_state_hash=root.state_hash,
        search_regime=root.search_regime,
        search_budget=root.search_budget,
        search_version=root.search_version,
        game_build=root.game_build,
    )


def _source_search_id(root: SearchRootEvidence) -> str:
    payload = json.dumps(
        {
            "state_hash": root.state_hash,
            "information_policy": root.information_policy,
            "search_regime": root.search_regime,
            "search_budget": root.search_budget,
            "search_version": root.search_version,
            "emulator_revision": root.emulator_revision,
            "game_build": root.game_build,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
