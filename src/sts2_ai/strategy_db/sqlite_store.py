from __future__ import annotations

import sqlite3
from pathlib import Path

from .schema import CachedActionEvaluation, StrategicEvidence

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

CREATE TABLE IF NOT EXISTS search_cache (
    state_hash TEXT NOT NULL,
    information_policy TEXT NOT NULL,
    action_id TEXT NOT NULL,
    value REAL NOT NULL,
    visits INTEGER NOT NULL,
    uncertainty REAL,
    search_version TEXT NOT NULL,
    search_config_id TEXT NOT NULL,
    model_id TEXT,
    emulator_revision TEXT NOT NULL,
    game_build TEXT NOT NULL,
    PRIMARY KEY (
        state_hash,
        information_policy,
        action_id,
        search_version,
        search_config_id,
        emulator_revision,
        game_build
    )
);
CREATE INDEX IF NOT EXISTS idx_search_cache_lookup
ON search_cache(
    state_hash,
    information_policy,
    search_version,
    search_config_id,
    emulator_revision,
    game_build
);
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


    def cache_action_evaluations(
        self,
        entries: tuple[CachedActionEvaluation, ...],
    ) -> None:
        if not entries:
            return

        self._connection.executemany(
            """
            INSERT INTO search_cache (
                state_hash, information_policy, action_id, value, visits,
                uncertainty, search_version, search_config_id, model_id,
                emulator_revision, game_build
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (
                state_hash, information_policy, action_id, search_version,
                search_config_id, emulator_revision, game_build
            ) DO UPDATE SET
                value = excluded.value,
                visits = excluded.visits,
                uncertainty = excluded.uncertainty,
                model_id = excluded.model_id
            """,
            [
                (
                    entry.state_hash,
                    entry.information_policy,
                    entry.action_id,
                    entry.value,
                    entry.visits,
                    entry.uncertainty,
                    entry.search_version,
                    entry.search_config_id,
                    entry.model_id,
                    entry.emulator_revision,
                    entry.game_build,
                )
                for entry in entries
            ],
        )
        self._connection.commit()

    def cached_action_evaluations(
        self,
        *,
        state_hash: str,
        information_policy: str,
        search_version: str,
        search_config_id: str,
        emulator_revision: str,
        game_build: str,
    ) -> tuple[CachedActionEvaluation, ...]:
        rows = self._connection.execute(
            """
            SELECT state_hash, information_policy, action_id, value, visits,
                   uncertainty, search_version, search_config_id, model_id,
                   emulator_revision, game_build
            FROM search_cache
            WHERE state_hash = ?
              AND information_policy = ?
              AND search_version = ?
              AND search_config_id = ?
              AND emulator_revision = ?
              AND game_build = ?
            ORDER BY action_id ASC
            """,
            (
                state_hash,
                information_policy,
                search_version,
                search_config_id,
                emulator_revision,
                game_build,
            ),
        ).fetchall()
        return tuple(CachedActionEvaluation(*row) for row in rows)
