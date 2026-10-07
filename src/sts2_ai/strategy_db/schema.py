from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategicEvidence:
    """Legacy compact exact-state action record kept for compatibility."""

    state_hash: str
    information_policy: str
    action_id: str
    value: float
    visits: int
    uncertainty: float | None
    search_version: str
    model_id: str | None
    emulator_revision: str
    game_build: str


@dataclass(frozen=True, slots=True)
class SearchRootEvidence:
    state_hash: str
    observation_hash: str
    information_policy: str
    search_regime: str
    legal_action_ids: tuple[str, ...]
    chosen_action_id: str
    search_budget: int
    expanded_nodes: int
    transitions: int
    transposition_hits: int
    search_version: str
    model_id: str | None
    emulator_revision: str
    game_build: str
    rollout_count: int = 0
    terminal_rollouts: int = 0
    cutoff_rollouts: int = 0
    rollout_steps: int = 0


@dataclass(frozen=True, slots=True)
class SearchActionEvidence:
    state_hash: str
    information_policy: str
    search_regime: str
    action_id: str
    action_kind: str
    action_payload_json: str
    value: float
    visits: int
    uncertainty: float | None
    search_budget: int
    search_version: str
    model_id: str | None
    emulator_revision: str
    game_build: str


@dataclass(frozen=True, slots=True)
class SearchObservationEvidence:
    observation_hash: str
    information_policy: str
    payload_json: str
