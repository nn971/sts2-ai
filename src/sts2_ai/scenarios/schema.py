from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scenario:
    scenario_id: str
    state_blob_uri: str
    exact_state_hash: str
    game_build: str
    emulator_revision: str
    information_policy: str
    reason: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplayScenarioAction:
    action_id: str
    kind: str
    value: float
    visits: int
    uncertainty: float | None


@dataclass(frozen=True, slots=True)
class ReplayScenario:
    """Durable prototype scenario reconstructed from a seed and stable action prefix."""

    scenario_id: str
    run_seed: str
    action_history: tuple[str, ...]
    decision_index: int
    exact_state_hash: str
    game_build: str
    emulator_revision: str
    information_policy: str
    reason: str
    value_margin: float
    max_uncertainty: float | None
    actions: tuple[ReplayScenarioAction, ...]
    tags: tuple[str, ...] = ()
