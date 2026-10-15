"""15-process playable Nature diagnostic rollout and exact prefix replay tests.

The backing process is a deliberately synthetic JSONL fixture. Real .NET
gameplay remains a separate user-run integration acceptance gate.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from sts2_ai.nature.corpus import NatureFixture, file_sha256, record_filename
from sts2_ai.nature.playable_corpus import (
    STATUS_CENSORED,
    STATUS_TERMINAL,
    STATUS_UNSUPPORTED,
    collect_playable_coverage,
)

FAKE = Path(__file__).resolve().parent / "fixtures" / "fake_nature_playable_jsonl.py"


def source_fixtures(count: int = 18, *, blocked: bool = False):
    return tuple(
        NatureFixture(
            fixture_id=f"playable-{i:03d}",
            seed=(f"blocked-{i}" if blocked and i % 3 == 0 else f"seed-{i}"),
            source_policy="synthetic-fair-first-v1",
            partition="train" if i < count // 2 else "development",
            act=2, ascension=0,
            encounter_id="proto.native.hive.encounter.mytes_normal",
            native_entry="MytesNormal", total_floor=21,
        )
        for i in range(count)
    )


def collect(fixtures, root, *, workers=15, resume=False, mode="uniform",
            nature_seed=41, cap=30):
    return collect_playable_coverage(
        fixtures, output_root=root,
        emulator_command=(sys.executable, str(FAKE)),
        emulator_revision="synthetic-protocol-for-tests",
        emulator_artifact_sha256=file_sha256(FAKE),
        nature_mode=mode, nature_seed=nature_seed,
        max_boundaries=cap, workers=workers,
        resume=resume, progress_every=6,
    )


def test_15_process_terminal_prefixes_are_worker_invariant(tmp_path):
    corpus = source_fixtures()
    one = collect(corpus, tmp_path / "one", workers=1)
    many = collect(corpus, tmp_path / "many", workers=15)
    assert one["status_counts"][STATUS_TERMINAL] == 18
    assert one["status_counts"][STATUS_UNSUPPORTED] == 0
    assert one["records_digest"] == many["records_digest"]
    assert (tmp_path / "one" / "corpus.jsonl").read_bytes() == (
        tmp_path / "many" / "corpus.jsonl"
    ).read_bytes()
    record = json.loads((tmp_path / "many" / "corpus.jsonl").read_text().splitlines()[0])
    assert record["terminal_outcome_known"] is False
    assert record["chance_count"] == 5
    assert record["player_count"] == 2
    assert record["representation"] == "legacy-indexed-tickets"
    assert many["execution"]["requested_workers"] == 15


def test_unsupported_sites_are_counted_and_replayed_identically(tmp_path):
    fixtures = source_fixtures(blocked=True)
    one = collect(fixtures, tmp_path / "one", workers=1)
    fifteen = collect(fixtures, tmp_path / "fifteen", workers=15)
    assert one["records_digest"] == fifteen["records_digest"]
    assert one["status_counts"][STATUS_UNSUPPORTED] == 6
    assert one["status_counts"][STATUS_TERMINAL] == 12
    assert one["blocked_sites"] == {"combat.unmigrated_test / random_index": 6}
    rows = [json.loads(x) for x in (tmp_path / "one" / "corpus.jsonl").read_text().splitlines()]
    blocker = next(x for x in rows if x["status"] == STATUS_UNSUPPORTED)
    assert blocker["terminal_fingerprint"] is None
    assert blocker["decision_count"] == 4
    assert blocker["blocked_site"] == "combat.unmigrated_test"


def test_budget_censoring_distinct_from_game_terminal(tmp_path):
    fixtures = source_fixtures(4)
    result = collect(fixtures, tmp_path / "censored", workers=2, cap=1)
    assert result["status_counts"][STATUS_CENSORED] == 4
    assert result["status_counts"][STATUS_TERMINAL] == 0
    assert result["total_chance_decisions"] == 4


def test_resume_partial_completions_and_reject_wrong_manifest(tmp_path):
    fixtures = source_fixtures(4)
    root = tmp_path / "resume"
    original = collect(fixtures, root, workers=1)
    path = root / "records" / record_filename(1, fixtures[1])
    expected = path.read_bytes()
    path.unlink()
    replay = collect(fixtures, root, workers=15, resume=True)
    assert replay["records_digest"] == original["records_digest"]
    assert path.read_bytes() == expected
    assert replay["execution"]["executed_fixtures"] == 1
    assert collect(fixtures, root, workers=2, resume=True)["execution"]["executed_fixtures"] == 0
    with pytest.raises(ValueError, match="manifest mismatch"):
        collect(fixtures, root, workers=1, resume=True, nature_seed=42)
    path.write_bytes(expected.replace(b"legacy-indexed-tickets", b"semantic-deferred"))
    with pytest.raises(ValueError, match="checksum"):
        collect(fixtures, root, workers=1, resume=True)


def test_source_groups_and_unsupported_initialization_rejected(tmp_path):
    fixtures = source_fixtures(4)
    with pytest.raises(ValueError, match="initial_hp"):
        collect((replace(fixtures[0], kind="intents"),), tmp_path / "wrong-kind", workers=1)
    with pytest.raises(ValueError, match="source-run split leakage"):
        collect((
            replace(fixtures[0], seed="same-source", partition="train"),
            replace(fixtures[1], seed="same-source", partition="development"),
        ), tmp_path / "leaky", workers=1)
    with pytest.raises(ValueError, match="duplicated"):
        collect((fixtures[0], fixtures[0]), tmp_path / "duplicates", workers=1)
