from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from .schema import (
    SearchActionEvidence,
    SearchObservationEvidence,
    SearchRootEvidence,
    StrategicEvidence,
)

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

CREATE TABLE IF NOT EXISTS search_observation_evidence (
    observation_hash TEXT NOT NULL,
    information_policy TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (observation_hash, information_policy)
);

CREATE TABLE IF NOT EXISTS search_root_evidence (
    state_hash TEXT NOT NULL,
    observation_hash TEXT NOT NULL,
    information_policy TEXT NOT NULL,
    search_regime TEXT NOT NULL,
    legal_action_ids_json TEXT NOT NULL,
    chosen_action_id TEXT NOT NULL,
    search_budget INTEGER NOT NULL,
    expanded_nodes INTEGER NOT NULL,
    transitions INTEGER NOT NULL,
    transposition_hits INTEGER NOT NULL,
    rollout_count INTEGER NOT NULL DEFAULT 0,
    terminal_rollouts INTEGER NOT NULL DEFAULT 0,
    cutoff_rollouts INTEGER NOT NULL DEFAULT 0,
    rollout_steps INTEGER NOT NULL DEFAULT 0,
    search_version TEXT NOT NULL,
    model_id TEXT,
    emulator_revision TEXT NOT NULL,
    game_build TEXT NOT NULL,
    PRIMARY KEY (
        state_hash,
        information_policy,
        search_regime,
        search_budget,
        search_version,
        emulator_revision,
        game_build
    )
);
CREATE INDEX IF NOT EXISTS idx_search_root_observation
ON search_root_evidence(observation_hash, information_policy);

CREATE TABLE IF NOT EXISTS search_action_evidence (
    state_hash TEXT NOT NULL,
    information_policy TEXT NOT NULL,
    search_regime TEXT NOT NULL,
    action_id TEXT NOT NULL,
    action_kind TEXT NOT NULL DEFAULT '',
    action_payload_json TEXT NOT NULL DEFAULT '{}',
    value REAL NOT NULL,
    visits INTEGER NOT NULL,
    uncertainty REAL,
    search_budget INTEGER NOT NULL,
    search_version TEXT NOT NULL,
    model_id TEXT,
    emulator_revision TEXT NOT NULL,
    game_build TEXT NOT NULL,
    PRIMARY KEY (
        state_hash,
        information_policy,
        search_regime,
        action_id,
        search_budget,
        search_version,
        emulator_revision,
        game_build
    )
);
CREATE INDEX IF NOT EXISTS idx_search_action_state
ON search_action_evidence(state_hash, information_policy, search_regime);
"""


class SQLiteStrategyStore:
    """SQLite development store for reusable exact-state search evidence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.executescript(_SCHEMA)
        self._ensure_column(
            "search_action_evidence",
            "action_kind",
            "TEXT NOT NULL DEFAULT ''",
        )
        self._ensure_column(
            "search_action_evidence",
            "action_payload_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )
        for column in (
            "rollout_count",
            "terminal_rollouts",
            "cutoff_rollouts",
            "rollout_steps",
        ):
            self._ensure_column(
                "search_root_evidence",
                column,
                "INTEGER NOT NULL DEFAULT 0",
            )
        self._connection.commit()

    def _ensure_column(
        self,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        existing = {
            str(row[1])
            for row in self._connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing:
            self._connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

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

    def for_state(
        self,
        state_hash: str,
        information_policy: str,
    ) -> tuple[StrategicEvidence, ...]:
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

    def upsert_search_observation(
        self,
        observation: SearchObservationEvidence,
    ) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO search_observation_evidence (
                    observation_hash, information_policy, payload_json
                ) VALUES (?, ?, ?)
                ON CONFLICT (observation_hash, information_policy)
                DO UPDATE SET payload_json = excluded.payload_json
                """,
                (
                    observation.observation_hash,
                    observation.information_policy,
                    observation.payload_json,
                ),
            )

    def observation(
        self,
        observation_hash: str,
        information_policy: str,
    ) -> SearchObservationEvidence | None:
        row = self._connection.execute(
            """
            SELECT observation_hash, information_policy, payload_json
            FROM search_observation_evidence
            WHERE observation_hash = ? AND information_policy = ?
            """,
            (observation_hash, information_policy),
        ).fetchone()
        if row is None:
            return None
        return SearchObservationEvidence(*row)

    def upsert_search(
        self,
        root: SearchRootEvidence,
        actions: Sequence[SearchActionEvidence],
    ) -> None:
        """Atomically persist one searched root and all root-action statistics."""

        expected_ids = set(root.legal_action_ids)
        action_ids = {action.action_id for action in actions}
        if action_ids != expected_ids:
            raise ValueError(
                "Search action evidence must cover every legal root action exactly once"
            )
        if root.chosen_action_id not in expected_ids:
            raise ValueError("Chosen action must be one of the legal root actions")

        key = (
            root.state_hash,
            root.information_policy,
            root.search_regime,
            root.search_budget,
            root.search_version,
            root.emulator_revision,
            root.game_build,
        )
        for action in actions:
            action_key = (
                action.state_hash,
                action.information_policy,
                action.search_regime,
                action.search_budget,
                action.search_version,
                action.emulator_revision,
                action.game_build,
            )
            if action_key != key:
                raise ValueError("Root and action search provenance must match")

        legal_json = json.dumps(
            root.legal_action_ids,
            separators=(",", ":"),
            sort_keys=False,
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO search_root_evidence (
                    state_hash, observation_hash, information_policy, search_regime,
                    legal_action_ids_json, chosen_action_id, search_budget,
                    expanded_nodes, transitions, transposition_hits,
                    rollout_count, terminal_rollouts, cutoff_rollouts, rollout_steps,
                    search_version, model_id, emulator_revision, game_build
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    state_hash, information_policy, search_regime, search_budget,
                    search_version, emulator_revision, game_build
                ) DO UPDATE SET
                    observation_hash = excluded.observation_hash,
                    legal_action_ids_json = excluded.legal_action_ids_json,
                    chosen_action_id = excluded.chosen_action_id,
                    expanded_nodes = excluded.expanded_nodes,
                    transitions = excluded.transitions,
                    transposition_hits = excluded.transposition_hits,
                    rollout_count = excluded.rollout_count,
                    terminal_rollouts = excluded.terminal_rollouts,
                    cutoff_rollouts = excluded.cutoff_rollouts,
                    rollout_steps = excluded.rollout_steps,
                    model_id = excluded.model_id
                """,
                (
                    root.state_hash,
                    root.observation_hash,
                    root.information_policy,
                    root.search_regime,
                    legal_json,
                    root.chosen_action_id,
                    root.search_budget,
                    root.expanded_nodes,
                    root.transitions,
                    root.transposition_hits,
                    root.rollout_count,
                    root.terminal_rollouts,
                    root.cutoff_rollouts,
                    root.rollout_steps,
                    root.search_version,
                    root.model_id,
                    root.emulator_revision,
                    root.game_build,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO search_action_evidence (
                    state_hash, information_policy, search_regime, action_id,
                    action_kind, action_payload_json, value, visits, uncertainty,
                    search_budget, search_version, model_id, emulator_revision, game_build
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (
                    state_hash, information_policy, search_regime, action_id,
                    search_budget, search_version, emulator_revision, game_build
                ) DO UPDATE SET
                    action_kind = excluded.action_kind,
                    action_payload_json = excluded.action_payload_json,
                    value = excluded.value,
                    visits = excluded.visits,
                    uncertainty = excluded.uncertainty,
                    model_id = excluded.model_id
                """,
                [
                    (
                        action.state_hash,
                        action.information_policy,
                        action.search_regime,
                        action.action_id,
                        action.action_kind,
                        action.action_payload_json,
                        action.value,
                        action.visits,
                        action.uncertainty,
                        action.search_budget,
                        action.search_version,
                        action.model_id,
                        action.emulator_revision,
                        action.game_build,
                    )
                    for action in actions
                ],
            )

    def search_versions(
        self,
        information_policy: str,
        *,
        search_regime: str = "oracle-exact",
    ) -> tuple[str, ...]:
        rows = self._connection.execute(
            """
            SELECT DISTINCT search_version
            FROM search_root_evidence
            WHERE information_policy = ? AND search_regime = ?
            ORDER BY search_version ASC
            """,
            (information_policy, search_regime),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def root_selection_visit_counts(
        self,
        information_policy: str,
        *,
        search_regime: str = "oracle-exact",
        search_version: str | None = None,
    ) -> tuple[tuple[int, int, int], ...]:
        """Per-budget counts of searched roots and chosen actions with zero visits."""

        version_clause = ""
        parameters: tuple[str, ...] = (information_policy, search_regime)
        if search_version is not None:
            version_clause = " AND r.search_version = ?"
            parameters += (search_version,)

        rows = self._connection.execute(
            f"""
            SELECT r.search_budget,
                   COUNT(*) AS roots,
                   SUM(CASE WHEN a.visits = 0 THEN 1 ELSE 0 END) AS unvisited
            FROM search_root_evidence AS r
            JOIN search_action_evidence AS a
              ON a.state_hash = r.state_hash
             AND a.information_policy = r.information_policy
             AND a.search_regime = r.search_regime
             AND a.search_budget = r.search_budget
             AND a.search_version = r.search_version
             AND a.emulator_revision = r.emulator_revision
             AND a.game_build = r.game_build
             AND a.action_id = r.chosen_action_id
            WHERE r.information_policy = ? AND r.search_regime = ?
            {version_clause}
            GROUP BY r.search_budget
            ORDER BY r.search_budget ASC
            """,
            parameters,
        ).fetchall()
        return tuple(
            (int(budget), int(roots), int(unvisited or 0))
            for budget, roots, unvisited in rows
        )

    def rollout_horizon_counts(
        self,
        information_policy: str,
        *,
        search_regime: str = "oracle-exact",
        search_version: str | None = None,
    ) -> tuple[tuple[int, int, int, int, int], ...]:
        """Aggregate rollout termination telemetry by search budget."""

        version_clause = ""
        parameters: tuple[str, ...] = (information_policy, search_regime)
        if search_version is not None:
            version_clause = " AND search_version = ?"
            parameters += (search_version,)

        rows = self._connection.execute(
            f"""
            SELECT search_budget,
                   SUM(rollout_count),
                   SUM(terminal_rollouts),
                   SUM(cutoff_rollouts),
                   SUM(rollout_steps)
            FROM search_root_evidence
            WHERE information_policy = ? AND search_regime = ?
            {version_clause}
            GROUP BY search_budget
            ORDER BY search_budget ASC
            """,
            parameters,
        ).fetchall()
        return tuple(
            (
                int(budget),
                int(rollouts or 0),
                int(terminals or 0),
                int(cutoffs or 0),
                int(steps or 0),
            )
            for budget, rollouts, terminals, cutoffs, steps in rows
        )

    def disagreement_state_hashes(
        self,
        information_policy: str,
        *,
        search_regime: str = "oracle-exact",
        search_version: str | None = None,
    ) -> tuple[str, ...]:
        """Exact roots searched at multiple budgets that chose different actions."""

        version_clause = ""
        parameters: tuple[str, ...] = (information_policy, search_regime)
        if search_version is not None:
            version_clause = " AND search_version = ?"
            parameters += (search_version,)

        rows = self._connection.execute(
            f"""
            SELECT state_hash
            FROM search_root_evidence
            WHERE information_policy = ? AND search_regime = ?
            {version_clause}
            GROUP BY state_hash
            HAVING COUNT(DISTINCT search_budget) > 1
               AND COUNT(DISTINCT chosen_action_id) > 1
            ORDER BY state_hash ASC
            """,
            parameters,
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def search_for_state(
        self,
        state_hash: str,
        information_policy: str,
        *,
        search_regime: str = "oracle-exact",
        search_version: str | None = None,
    ) -> tuple[tuple[SearchRootEvidence, tuple[SearchActionEvidence, ...]], ...]:
        version_clause = ""
        parameters: tuple[str, ...] = (
            state_hash,
            information_policy,
            search_regime,
        )
        if search_version is not None:
            version_clause = " AND search_version = ?"
            parameters += (search_version,)

        roots = self._connection.execute(
            f"""
            SELECT state_hash, observation_hash, information_policy, search_regime,
                   legal_action_ids_json, chosen_action_id, search_budget,
                   expanded_nodes, transitions, transposition_hits,
                   rollout_count, terminal_rollouts, cutoff_rollouts, rollout_steps,
                   search_version, model_id, emulator_revision, game_build
            FROM search_root_evidence
            WHERE state_hash = ? AND information_policy = ? AND search_regime = ?
            {version_clause}
            ORDER BY search_budget ASC, search_version ASC
            """,
            parameters,
        ).fetchall()

        results = []
        for row in roots:
            legal_ids_raw = json.loads(row[4])
            if not isinstance(legal_ids_raw, list) or not all(
                isinstance(item, str) for item in legal_ids_raw
            ):
                raise RuntimeError("Stored legal action ids are malformed")
            root = SearchRootEvidence(
                state_hash=row[0],
                observation_hash=row[1],
                information_policy=row[2],
                search_regime=row[3],
                legal_action_ids=tuple(legal_ids_raw),
                chosen_action_id=row[5],
                search_budget=row[6],
                expanded_nodes=row[7],
                transitions=row[8],
                transposition_hits=row[9],
                rollout_count=row[10],
                terminal_rollouts=row[11],
                cutoff_rollouts=row[12],
                rollout_steps=row[13],
                search_version=row[14],
                model_id=row[15],
                emulator_revision=row[16],
                game_build=row[17],
            )
            action_rows = self._connection.execute(
                """
                SELECT state_hash, information_policy, search_regime, action_id,
                       action_kind, action_payload_json, value, visits, uncertainty,
                       search_budget, search_version, model_id, emulator_revision, game_build
                FROM search_action_evidence
                WHERE state_hash = ?
                  AND information_policy = ?
                  AND search_regime = ?
                  AND search_budget = ?
                  AND search_version = ?
                  AND emulator_revision = ?
                  AND game_build = ?
                ORDER BY action_id ASC
                """,
                (
                    root.state_hash,
                    root.information_policy,
                    root.search_regime,
                    root.search_budget,
                    root.search_version,
                    root.emulator_revision,
                    root.game_build,
                ),
            ).fetchall()
            actions = tuple(SearchActionEvidence(*action_row) for action_row in action_rows)
            results.append((root, actions))
        return tuple(results)
