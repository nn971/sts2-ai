"""N3 real-process isolation using a synthetic emulator-shaped JSONL subprocess.

CI verifies the multiprocess orchestration and persistence. Native game and
actual .NET integration are separate optional acceptance tests.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from sts2_ai.nature.corpus import NatureFixture, file_sha256, load_fixtures, record_filename
from sts2_ai.nature.worker_corpus import collect_isolated_corpus

FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_nature_jsonl.py"


def fixtures(count=20):
    return tuple(
        NatureFixture(
            fixture_id=f"fixture-{i:03d}", seed=f"source-seed-{i // 2}",
            source_policy="frozen-player-v8",
            partition="train" if i < 10 else "development",
            ascension=0, act=2,
            encounter_id="proto.native.hive.encounter.mytes_normal",
            native_entry="MytesNormal", total_floor=21,
        )
        for i in range(count)
    )


def prepared_fixtures(count=20):
    # Source runs must remain together across partitions.
    base = fixtures(count)
    return tuple(replace(
        x, partition="train" if i // 2 < 5 else "development",
    ) for i, x in enumerate(base))


def execute(items, output, workers=15, resume=False, mode="uniform", seed=29):
    return collect_isolated_corpus(
        items, output_root=output,
        emulator_command=(sys.executable, str(FAKE)),
        emulator_revision="fake-test-bridge-v1",
        emulator_artifact_sha256=file_sha256(FAKE),
        nature_mode=mode, nature_seed=seed,
        workers=workers, resume=resume,
        progress_every=7,
    )


def test_15_worker_processes_match_one_worker_corpus_and_replay(tmp_path: Path):
    items = prepared_fixtures()
    serial = execute(items, tmp_path / "serial", workers=1)
    parallel = execute(items, tmp_path / "parallel", workers=15)
    assert serial["records_digest"] == parallel["records_digest"]
    assert serial["fixture_count"] == 20
    assert serial["decision_count"] == 60
    assert serial["source_run_count"] == 10
    assert parallel["execution"]["workers_requested"] == 15
    assert parallel["execution"]["worker_processes_observed_this_invocation"] >= 1
    assert (tmp_path / "serial" / "corpus.jsonl").read_bytes() == (
        tmp_path / "parallel" / "corpus.jsonl"
    ).read_bytes()
    record = json.loads((tmp_path / "parallel" / "corpus.jsonl").read_text().splitlines()[0])
    assert record["playable_combat"] is False
    assert record["calibrated_native_reference_law"] is False
    assert record["final_fingerprint"] == record["replay_fingerprint"]


def test_resumption_retains_canonical_output_and_detects_corruption(tmp_path: Path):
    items = prepared_fixtures(4)
    root = tmp_path / "resume"
    first = execute(items, root, workers=2)
    with pytest.raises(ValueError, match="requires --resume"):
        execute(items, root, workers=2)
    path = root / "records" / record_filename(1, items[1])
    original = path.read_text()
    path.unlink()
    resumed = execute(items, root, workers=15, resume=True)
    assert resumed["records_digest"] == first["records_digest"]
    assert resumed["execution"]["fixtures_executed_this_invocation"] == 1
    assert path.read_text() == original
    unchanged = execute(items, root, workers=1, resume=True)
    assert unchanged["execution"]["fixtures_executed_this_invocation"] == 0
    assert unchanged["records_digest"] == first["records_digest"]
    path.write_text(original.replace('"status": "ISOLATED_COMPLETE"', '"status": "DEFEAT"'))
    with pytest.raises(ValueError, match="checksum"):
        execute(items, root, workers=1, resume=True)
    path.write_text(original)
    with pytest.raises(ValueError, match="manifest mismatch"):
        execute(items, root, workers=1, resume=True, seed=30)


def test_source_disjoint_and_fixture_input_validation(tmp_path: Path):
    items = prepared_fixtures(4)
    path = tmp_path / "fixtures.jsonl"
    path.write_text("\n".join(json.dumps(asdict(x)) for x in reversed(items)) + "\n")
    assert load_fixtures(path) == items
    leaked = (replace(items[0], partition="evaluation"), *items[1:])
    with pytest.raises(ValueError, match="leaks|leakage"):
        execute(leaked, tmp_path / "leak", workers=1)
    with pytest.raises(ValueError, match="Duplicate"):
        execute((items[0], items[0]), tmp_path / "dup", workers=1)
    with pytest.raises(ValueError, match="only Hive/Glory"):
        replace(items[0], act=1)
    with pytest.raises(ValueError, match="Unsupported Nature fixture schema"):
        replace(items[0], schema="unknown")

    path.write_text(json.dumps(asdict(items[0])) + "\n" + json.dumps(
        asdict(replace(items[0], fixture_id="different", partition="evaluation"))
    ))
    with pytest.raises(ValueError, match="leaks"):
        load_fixtures(path)


def test_protocol_and_policy_identity_rejected_on_resume(tmp_path: Path):
    items = prepared_fixtures(2)
    root = tmp_path / "strict"
    execute(items, root, workers=1, mode="first")
    with pytest.raises(ValueError, match="manifest mismatch"):
        execute(items, root, workers=1, resume=True, mode="last")
    extra = root / "records" / "foreign.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="Unknown Nature record"):
        execute(items, root, workers=1, resume=True)
    extra.unlink()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["scope"] == "isolated-combat-entry-only"
    assert manifest["fixture_count"] == 2
