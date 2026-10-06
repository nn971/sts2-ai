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
