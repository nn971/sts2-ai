from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sts2_ai.emulator import LegalAction, StateHandle


@dataclass(frozen=True, slots=True)
class SearchBudget:
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
    evaluations: tuple[ActionEvaluation, ...]
    expanded_nodes: int
    search_version: str


class SearchAlgorithm(Protocol):
    def search(self, state: StateHandle, budget: SearchBudget) -> SearchResult: ...
