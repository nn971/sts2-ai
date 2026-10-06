from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PolicyTarget:
    action_id: str
    probability: float


@dataclass(frozen=True, slots=True)
class TrainingExample:
    observation_hash: str
    information_policy: str
    policy_targets: tuple[PolicyTarget, ...]
    value_target: float
    source_search_id: str
    emulator_revision: str
