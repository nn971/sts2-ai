from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    created_at_utc: str
    kind: str
    ai_commit: str
    emulator_commit: str
    game_build: str
    information_policy: str
    record_count: int
    content_digest: str
    storage_uri: str

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
