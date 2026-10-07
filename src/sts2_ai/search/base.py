from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sts2_ai.emulator import LegalAction, StateHandle


@dataclass(frozen=True, slots=True)
class SearchBudget:
    max_simulations: int | None = None
    max_nodes: int | None = None
    max_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    action: LegalAction
    value: float
    visits: int
    uncertainty: float | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    root_state_hash: str
    root_observation_hash: str
    root_observation_json: str
    evaluations: tuple[ActionEvaluation, ...]
    expanded_nodes: int
    transitions: int
    transposition_hits: int
    search_version: str
    rollout_count: int = 0
    terminal_rollouts: int = 0
    boundary_rollouts: int = 0
    cutoff_rollouts: int = 0
    rollout_steps: int = 0


class SearchAlgorithm(Protocol):
    def search(self, state: StateHandle, budget: SearchBudget) -> SearchResult: ...
