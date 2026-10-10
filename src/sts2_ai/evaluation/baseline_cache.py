"""Strictly keyed cache of fixed warm-start held-out evaluation rows.

This caches only public evaluation summaries, not emulator RNG or transitions.
A key includes model bytes, emulator revision, seeds, goal, policy and decision
cap; a mismatch is an error, never a silent fallback to stale measurements.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "sts2-warm-start-heldout-cache-v1"


def baseline_identity(
    *, warm_start: Path, emulator_revision: str, seed_prefix: str,
    count: int, episode_goal: str, max_decisions: int,
    environment: str, policy_id: str,
) -> dict[str, Any]:
    if count <= 0 or max_decisions <= 0:
        raise ValueError("Invalid baseline evaluation dimensions")
    return {
        "model_sha256": hashlib.sha256(warm_start.read_bytes()).hexdigest(),
        "emulator_revision": emulator_revision,
        "seed_prefix": seed_prefix,
        "count": count,
        "episode_goal": episode_goal,
        "max_decisions": max_decisions,
        "environment": environment,
        "policy_id": policy_id,
    }


def validate_rows(rows: Any, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != identity["count"]:
        raise ValueError("Warm-start cache row count mismatch")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Malformed warm-start cache row")
        if row.get("seed") != f"{identity['seed_prefix']}-{index}":
            raise ValueError("Warm-start cache evaluation seed mismatch")
        if row.get("episode_goal_version") != identity["episode_goal"]:
            raise ValueError("Warm-start cache episode goal mismatch")
        if row.get("censored") is not False and row.get("censored") is not True:
            raise ValueError("Warm-start cache missing censoring status")
    return rows


def load_cache(
    path: Path, identity: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError(f"Wrong warm-start cache schema: {path}")
    if data.get("identity") != identity:
        raise ValueError(
            f"Warm-start cache identity mismatch: {path}. "
            "Choose another cache path rather than mixing evaluations."
        )
    return validate_rows(data.get("rows"), identity)


def save_cache(
    path: Path, identity: dict[str, Any], rows: list[dict[str, Any]],
) -> None:
    validate_rows(rows, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".pending", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                {"schema": SCHEMA, "identity": identity, "rows": rows},
                stream, indent=2, sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
