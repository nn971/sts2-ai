from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_dirty(root: Path) -> bool | None:
    status = _git(root, "status", "--porcelain")
    return None if status is None else bool(status)


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    experiment_id: str
    created_at_utc: str
    ai_commit: str | None
    ai_dirty: bool | None
    emulator_commit: str | None
    emulator_dirty: bool | None
    game_build: str
    emulator_schema_version: str
    binding_version: str
    information_policy: str
    config: dict[str, Any]
    seeds: dict[str, int]
    model_ids: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()
    strategy_db_snapshot: str | None = None

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_experiment_manifest(
    *,
    repo_root: Path,
    experiment_id: str,
    game_build: str,
    emulator_schema_version: str,
    binding_version: str,
    information_policy: str,
    config: dict[str, Any],
    seeds: dict[str, int],
    model_ids: tuple[str, ...] = (),
    dataset_ids: tuple[str, ...] = (),
    strategy_db_snapshot: str | None = None,
) -> ExperimentManifest:
    emulator_root = repo_root / "emulator"
    return ExperimentManifest(
        experiment_id=experiment_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        ai_commit=_git(repo_root, "rev-parse", "HEAD"),
        ai_dirty=_git_dirty(repo_root),
        emulator_commit=(
            _git(emulator_root, "rev-parse", "HEAD") if emulator_root.exists() else None
        ),
        emulator_dirty=_git_dirty(emulator_root) if emulator_root.exists() else None,
        game_build=game_build,
        emulator_schema_version=emulator_schema_version,
        binding_version=binding_version,
        information_policy=information_policy,
        config=config,
        seeds=seeds,
        model_ids=model_ids,
        dataset_ids=dataset_ids,
        strategy_db_snapshot=strategy_db_snapshot,
    )
