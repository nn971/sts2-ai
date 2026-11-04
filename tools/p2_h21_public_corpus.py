"""Versioned public-only H21 benchmark corpus; no RunState, RNG or hidden state.

Live decisions are taken from the FAIR public observation and legal-action
endpoints. The canonical digest detects accidental mutation during replays.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sts2_ai.training.h19_resource_codec import (
    EMULATOR_PUBLIC_SCHEMA_V1,
    EMULATOR_PUBLIC_SCHEMA_V2,
)
from sts2_ai.training.h21_macro_offer_codec import H21_CODEC

CORPUS_SCHEMA = "sts2-p2-h21-public-decision-corpus-v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_SCHEMAS = {EMULATOR_PUBLIC_SCHEMA_V1, EMULATOR_PUBLIC_SCHEMA_V2}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def public_digest(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def make_public_record(
    seed: str, step: int, schema_id: str,
    frame: dict[str, Any], actions: list[dict[str, str]],
) -> dict[str, Any]:
    """Copy only a verified public view and currently legal action menu."""
    record = {
        "seed": seed, "step": step, "public_schema_id": schema_id,
        "frame": frame,
        "actions": [
            {"action_id": action["actionId"], "kind": action["kind"],
             "payload_json": action["payloadJson"]}
            for action in actions
        ],
    }
    record["public_sha256"] = public_digest(record)
    _verify_record(record)
    return record


def _verify_record(record: Any) -> None:
    if not isinstance(record, dict) or set(record) != {
        "seed", "step", "public_schema_id", "frame", "actions", "public_sha256",
    }:
        raise ValueError("Malformed H21 public corpus record")
    if (not isinstance(record["seed"], str) or not record["seed"]
            or type(record["step"]) is not int or record["step"] < 0
            or not isinstance(record["public_schema_id"], str)
            or record["public_schema_id"] not in _PUBLIC_SCHEMAS
            or not isinstance(record["frame"], dict)
            or not isinstance(record["actions"], list)
            or not record["actions"]):
        raise ValueError("Invalid H21 public frame/seed/schema/action menu")
    for action in record["actions"]:
        if (
            not isinstance(action, dict)
            or set(action) != {"action_id", "kind", "payload_json"}
            or any(not isinstance(action[k], str) or not action[k]
                   for k in action)
        ):
            raise ValueError("Malformed H21 public legal action")
        try:
            payload = json.loads(action["payload_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid H21 action payload JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("H21 action payload must be a JSON object")
    digest = record["public_sha256"]
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValueError("Invalid public frame digest")
    if digest != public_digest({k: v for k, v in record.items()
                               if k != "public_sha256"}):
        raise ValueError("H21 public corpus was modified")


def make_public_corpus(
    *, emulator_dll: Path, records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot capture an empty H21 public benchmark corpus")
    corpus = {
        "schema": CORPUS_SCHEMA,
        "codec": H21_CODEC,
        "emulator_cli_sha256": hashlib.sha256(emulator_dll.read_bytes()).hexdigest(),
        "records": records,
    }
    verify_public_corpus(corpus)
    return corpus


def verify_public_corpus(data: Any) -> dict[str, Any]:
    if (not isinstance(data, dict)
            or set(data) != {"schema", "codec", "emulator_cli_sha256", "records"}
            or data["schema"] != CORPUS_SCHEMA
            or data["codec"] != H21_CODEC
            or not isinstance(data["emulator_cli_sha256"], str)
            or not _HASH.fullmatch(data["emulator_cli_sha256"])
            or not isinstance(data["records"], list) or not data["records"]):
        raise ValueError("Incompatible H21 P2 public corpus version/provenance")
    seen: set[tuple[str, int]] = set()
    for record in data["records"]:
        _verify_record(record)
        key = record["seed"], record["step"]
        if key in seen:
            raise ValueError("Duplicate public H21 seed and decision number")
        seen.add(key)
    return data


def write_public_corpus(path: Path, corpus: dict[str, Any]) -> None:
    verify_public_corpus(corpus)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite H21 public corpus: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents accidental overwrites in paired benchmarks.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(corpus, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load_public_corpus(path: Path) -> dict[str, Any]:
    return verify_public_corpus(json.loads(path.read_text(encoding="utf-8")))
