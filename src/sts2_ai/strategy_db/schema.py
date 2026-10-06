from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategicEvidence:
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
