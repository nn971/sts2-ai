from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import StrategicEvidence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategic_evidence (
    state_hash TEXT NOT NULL,
    information_policy TEXT NOT NULL,
    action_id TEXT NOT NULL,
    value REAL NOT NULL,
    visits INTEGER NOT NULL,
    uncertainty REAL,
    search_version TEXT NOT NULL,
    model_id TEXT,
    emulator_revision TEXT NOT NULL,
    game_build TEXT NOT NULL,
    PRIMARY KEY (
        state_hash,
        information_policy,
        action_id,
        search_version,
        emulator_revision,
        game_build
    )
);
CREATE INDEX IF NOT EXISTS idx_evidence_state
ON strategic_evidence(state_hash, information_policy);
"""


class SQLiteStrategyStore:
    """Small development store for exact-state strategic evidence.

    The schema is intentionally conservative. It makes provenance unavoidable while
    leaving graph/storage design open for later measurements.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection = sqlite3.connect(path)
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SQLiteStrategyStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def upsert(self, evidence: StrategicEvidence) -> None:
        self._connection.execute(
            """
            INSERT INTO strategic_evidence (
                state_hash, information_policy, action_id, value, visits,
                uncertainty, search_version, model_id, emulator_revision, game_build
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                state_hash, information_policy, action_id, search_version,
                emulator_revision, game_build
            ) DO UPDATE SET
                value = excluded.value,
                visits = excluded.visits,
                uncertainty = excluded.uncertainty,
                model_id = excluded.model_id
            """,
            (
                evidence.state_hash,
                evidence.information_policy,
                evidence.action_id,
                evidence.value,
                evidence.visits,
                evidence.uncertainty,
                evidence.search_version,
                evidence.model_id,
                evidence.emulator_revision,
                evidence.game_build,
            ),
        )
        self._connection.commit()

    def for_state(self, state_hash: str, information_policy: str) -> tuple[StrategicEvidence, ...]:
        rows = self._connection.execute(
            """
            SELECT state_hash, information_policy, action_id, value, visits,
                   uncertainty, search_version, model_id, emulator_revision, game_build
            FROM strategic_evidence
            WHERE state_hash = ? AND information_policy = ?
            ORDER BY visits DESC, action_id ASC
            """,
            (state_hash, information_policy),
        ).fetchall()
        return tuple(StrategicEvidence(*row) for row in rows)
