"""Versioned input recipes and immutable integrity checks for isolated Nature corpora.

This is an *isolated chance-boundary* corpus, not a replayable player-combat
dataset. Each row names a fixed source-shaped emulator fixture and the source
run/partition it belongs to. No hidden exact state enters player features.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .protocol import ISOLATED_EMULATOR_SCHEMA, canonical_sha256
from .trace import ChanceTranscript

FIXTURE_SCHEMA = "sts2-ai-nature-isolated-fixture-v1"
CORPUS_SCHEMA = "sts2-ai-nature-isolated-corpus-v1"
RECORD_SCHEMA = "sts2-ai-nature-isolated-record-v1"
PARTITIONS = frozenset({"train", "development", "evaluation"})


@dataclass(frozen=True, slots=True)
class NatureFixture:
    fixture_id: str
    seed: str
    source_policy: str
    partition: str
    ascension: int
    act: int
    encounter_id: str
    native_entry: str
    total_floor: int
    kind: str = "initial_hp"
    map_column: int = 3
    map_row: int = 5
    schema: str = FIXTURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FIXTURE_SCHEMA:
            raise ValueError("Unsupported Nature fixture schema")
        if any(not isinstance(x, str) or not x for x in (
            self.fixture_id, self.seed, self.source_policy,
            self.encounter_id, self.native_entry,
        )):
            raise ValueError("Missing Nature fixture provenance")
        if self.partition not in PARTITIONS:
            raise ValueError("Invalid Nature fixture partition")
        if self.kind not in ("initial_hp", "intents") or self.act not in (2, 3):
            raise ValueError("Current emulator supports only Hive/Glory HP/intent fixtures")
        for v in (self.ascension, self.total_floor, self.map_column, self.map_row):
            if type(v) is not int or v < 0:
                raise ValueError("Invalid Nature fixture integer parameters")
        if self.total_floor < 1:
            raise ValueError("Nature fixture floor must be positive")

    @property
    def source_key(self) -> tuple[str, str, int]:
        return self.source_policy, self.seed, self.ascension

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(asdict(self))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NatureFixture:
        # Dataclass construction rejects unknown/missing fields.
        return cls(**raw)

    def emulator_kwargs(self) -> dict[str, Any]:
        return {
            "seed": self.seed, "kind": self.kind, "act": self.act,
            "encounter_id": self.encounter_id,
            "native_entry": self.native_entry,
            "total_floor": self.total_floor, "ascension": self.ascension,
            "map_column": self.map_column, "map_row": self.map_row,
        }


def load_fixtures(path: Path) -> tuple[NatureFixture, ...]:
    items: list[NatureFixture] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("Expected a JSON object")
            items.append(NatureFixture.from_dict(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Nature fixture at line {line_no}: {exc}") from exc
    if not items:
        raise ValueError("Empty Nature fixture corpus")
    ordered = tuple(sorted(items, key=lambda item: item.fixture_id))
    if len(set(x.fixture_id for x in ordered)) != len(ordered):
        raise ValueError("Duplicate Nature fixture ID")
    source_partitions: dict[tuple[str, str, int], str] = {}
    for item in ordered:
        previous = source_partitions.setdefault(item.source_key, item.partition)
        if previous != item.partition:
            raise ValueError("Source run leaks across Nature train/development/evaluation")
    return ordered


def manifest_identity(
    fixtures: tuple[NatureFixture, ...], *,
    emulator_revision: str, emulator_artifact_sha256: str,
    nature_mode: str, nature_seed: int, max_choices: int,
) -> dict[str, Any]:
    if not fixtures or not emulator_revision or len(emulator_artifact_sha256) != 64:
        raise ValueError("Invalid Nature emulator pin/fixture provenance")
    if nature_mode not in ("first", "last", "uniform"):
        raise ValueError("Unknown Nature experiment policy")
    if type(nature_seed) is not int or type(max_choices) is not int or max_choices <= 0:
        raise ValueError("Invalid Nature seed/decision budget")
    identity = {
        "schema": CORPUS_SCHEMA,
        "emulator_revision": emulator_revision,
        "emulator_artifact_sha256": emulator_artifact_sha256,
        "chance_protocol": ISOLATED_EMULATOR_SCHEMA,
        "scope": "isolated-combat-entry-only",
        "nature_mode": nature_mode, "nature_seed": nature_seed,
        "max_choices": max_choices,
        "fixture_count": len(fixtures),
        "fixtures_sha256": canonical_sha256([asdict(x) for x in fixtures]),
        "source_run_count": len({x.source_key for x in fixtures}),
    }
    return {**identity, "identity_sha256": canonical_sha256(identity)}


def validated_record(
    raw: dict[str, Any], fixture: NatureFixture, manifest: dict[str, Any],
) -> dict[str, Any]:
    expected = raw.get("record_sha256")
    unsigned = {k: v for k, v in raw.items() if k != "record_sha256"}
    if not isinstance(expected, str) or expected != canonical_sha256(unsigned):
        raise ValueError("Nature record checksum mismatch")
    if (
        raw.get("schema") != RECORD_SCHEMA or
        raw.get("manifest_sha256") != manifest["identity_sha256"] or
        raw.get("fixture") != asdict(fixture) or
        raw.get("fixture_fingerprint") != fixture.fingerprint or
        raw.get("emulator_revision") != manifest["emulator_revision"] or
        raw.get("status") != "ISOLATED_COMPLETE" or
        raw.get("playable_combat") is not False or
        raw.get("calibrated_native_reference_law") is not False
    ):
        raise ValueError("Nature record provenance/status mismatch")
    transcript = ChanceTranscript.from_json(json.dumps(raw.get("transcript")))
    if (transcript.emulator_revision != manifest["emulator_revision"] or
            transcript.nature_policy_id != raw.get("nature_policy_id") or
            len(transcript.entries) != raw.get("decision_count")):
        raise ValueError("Nature record transcript inconsistency")
    if (not isinstance(raw.get("final_fingerprint"), str) or
            raw["final_fingerprint"] != raw.get("replay_fingerprint")):
        raise ValueError("Nature record replay mismatch")
    return raw


def seal_record(data: dict[str, Any]) -> dict[str, Any]:
    if "record_sha256" in data:
        raise ValueError("Nature record already sealed")
    return {**data, "record_sha256": canonical_sha256(data)}


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".nature-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_filename(index: int, fixture: NatureFixture) -> str:
    return f"{index:06d}-{fixture.fingerprint[:20]}.json"


def load_record(path: Path, fixture: NatureFixture, manifest: dict[str, Any]) -> dict[str, Any]:
    return validated_record(json.loads(path.read_text(encoding="utf-8")), fixture, manifest)
