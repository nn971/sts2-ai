"""Resumable 15-process diagnostics for *legacy indexed* playable Nature fixtures.

All card/enemy mechanics are owned by the selected external emulator DLL.
The actor is deliberately only `first-legal`, for structural coverage.
Unsupported chance events and boundary-budget expiry are measured as censored
data, never as deaths, victories, or training rewards.

The new deferred-draw semantic wire remains a completely separate future
capability and must not share this corpus' likelihood factors.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing.util import Finalize
from pathlib import Path
from typing import Any

from sts2_ai.emulator.controlled_chance import (
    ControlledChanceUnavailable,
    IsolatedNatureBridge,
    NatureJsonlProcess,
)
from sts2_ai.emulator.playable_chance import (
    PLAYABLE_SCHEMA,
    PlayableNatureBridge,
)
from sts2_ai.nature.corpus import (
    NatureFixture,
    atomic_write,
    load_fixtures,
    record_filename,
    seal_record,
)
from sts2_ai.nature.playable_trajectory import (
    PlayableDecision,
    play_controlled_fixture,
)
from sts2_ai.nature.policies import (
    FirstLegalNature,
    LastLegalNature,
    ScriptedNature,
    UniformNature,
)
from sts2_ai.nature.protocol import canonical_sha256

PLAYABLE_CORPUS_SCHEMA = "sts2-ai-nature-playable-coverage-corpus-v1"
PLAYABLE_RECORD_SCHEMA = "sts2-ai-nature-playable-coverage-record-v1"
STATUS_TERMINAL = "TERMINAL_OUTCOME_UNCLASSIFIED"
STATUS_CENSORED = "CENSORED_BOUNDARY_LIMIT"
STATUS_UNSUPPORTED = "UNSUPPORTED_CONTROLLED_CHANCE"
VALID_STATUSES = frozenset({STATUS_TERMINAL, STATUS_CENSORED, STATUS_UNSUPPORTED})

_worker_wire: NatureJsonlProcess | None = None
_worker_isolated: IsolatedNatureBridge | None = None
_worker_playable: PlayableNatureBridge | None = None


def _close_worker() -> None:
    global _worker_wire, _worker_isolated, _worker_playable
    if _worker_wire is not None:
        _worker_wire.close()
        _worker_wire = None
        _worker_isolated = None
        _worker_playable = None


def _initialize(command: tuple[str, ...], revision: str) -> None:
    global _worker_wire, _worker_isolated, _worker_playable
    if _worker_wire is not None:
        raise RuntimeError("A playable Nature worker was already initialized")
    _worker_wire = NatureJsonlProcess(list(command))
    try:
        _worker_isolated = IsolatedNatureBridge(
            _worker_wire, emulator_revision=revision,
        )
        _worker_playable = PlayableNatureBridge(
            _worker_wire, emulator_revision=revision,
        )
    except BaseException:
        _close_worker()
        raise
    Finalize(None, _close_worker, exitpriority=10)


def _initial_policy(mode: str, seed: int, fixture: NatureFixture):
    if mode == "first":
        return FirstLegalNature()
    if mode == "last":
        return LastLegalNature()
    if mode == "uniform":
        return UniformNature(
            seed=int(canonical_sha256([seed, fixture.fingerprint])[:16], 16)
        )
    raise ValueError("Unknown Nature proposal mode")


def _blocked_site(message: str) -> tuple[str | None, str | None]:
    site = re.search(r"\bsite=([^\s;,]+)", message)
    operation = re.search(r"\boperation=([^\s;,]+)", message)
    return (
        site.group(1) if site else None,
        operation.group(1) if operation else None,
    )


def _status(exc: Exception) -> tuple[str, str | None, str | None]:
    message = str(exc)
    if isinstance(exc, RuntimeError) and "censored" in message.lower():
        return STATUS_CENSORED, None, None
    if isinstance(exc, ControlledChanceUnavailable) and (
        STATUS_UNSUPPORTED in message
    ):
        site, operation = _blocked_site(message)
        if not site or not operation:
            raise ValueError("Unclassified unsupported Nature site") from exc
        return STATUS_UNSUPPORTED, site, operation
    raise exc


def _run_once(fixture: NatureFixture, policy: Any, *, cap: int):
    if _worker_isolated is None or _worker_playable is None:
        raise RuntimeError("Playable worker is not initialized")
    decisions: list[PlayableDecision] = []
    status = STATUS_TERMINAL
    site = operation = None
    terminal_fingerprint: str | None = None
    try:
        result = play_controlled_fixture(
            _worker_isolated, _worker_playable, policy,
            seed=fixture.seed, act=fixture.act,
            encounter_id=fixture.encounter_id,
            native_entry=fixture.native_entry,
            total_floor=fixture.total_floor,
            player_choose=lambda frame: frame.action_ids[0],
            max_boundaries=cap, on_decision=decisions.append,
        )
        if tuple(decisions) != result.decisions:
            raise RuntimeError("Playable callback disagrees with returned trace")
        terminal_fingerprint = result.terminal_fingerprint
    except (ControlledChanceUnavailable, RuntimeError) as exc:
        status, site, operation = _status(exc)
    return status, site, operation, terminal_fingerprint, tuple(decisions)


def _run_job(index: int, fixture: NatureFixture, manifest: dict[str, Any]):
    policy = _initial_policy(
        manifest["nature_mode"], manifest["nature_seed"], fixture,
    )
    cap = manifest["max_boundaries"]
    status, site, operation, terminal, decisions = _run_once(
        fixture, policy, cap=cap,
    )
    scripted = ScriptedNature.from_selections(
        (row.identity, int(row.selection))
        for row in decisions if row.kind == "nature"
    )
    # Reuse every observed player action; avoid assuming fixed first-legal
    # observation is stable in an adversarial Nature branch.
    # On a censored/unsupported prefix the second run must reach the same
    # blocked boundary and never consume any extra scripted outcomes.
    player_actions = iter(
        row.selection for row in decisions if row.kind == "player"
    )
    assert _worker_isolated is not None and _worker_playable is not None
    repeat: list[PlayableDecision] = []
    repeat_status = STATUS_TERMINAL
    repeat_site = repeat_operation = None
    repeat_terminal = None
    try:
        result = play_controlled_fixture(
            _worker_isolated, _worker_playable, scripted,
            seed=fixture.seed, act=fixture.act,
            encounter_id=fixture.encounter_id,
            native_entry=fixture.native_entry, total_floor=fixture.total_floor,
            player_choose=lambda frame: next(player_actions),
            max_boundaries=cap, on_decision=repeat.append,
        )
        repeat_terminal = result.terminal_fingerprint
        if tuple(repeat) != result.decisions:
            raise ValueError("Playable replay callback diverged")
    except (ControlledChanceUnavailable, RuntimeError) as exc:
        repeat_status, repeat_site, repeat_operation = _status(exc)
    scripted.assert_complete()
    if (
        repeat_status != status or repeat_site != site or
        repeat_operation != operation or repeat_terminal != terminal or
        tuple(repeat) != decisions
    ):
        raise ValueError(f"Joint playable chance/player replay diverged: {fixture.fixture_id}")
    record = seal_record({
        "schema": PLAYABLE_RECORD_SCHEMA,
        "manifest_sha256": manifest["identity_sha256"],
        "fixture": asdict(fixture),
        "fixture_fingerprint": fixture.fingerprint,
        "emulator_revision": manifest["emulator_revision"],
        "representation": "legacy-indexed-tickets",
        "probability_status": "UNVERIFIED",
        "status": status,
        "blocked_site": site,
        "blocked_operation": operation,
        "terminal_outcome_known": False,
        "terminal_fingerprint": terminal,
        "decision_count": len(decisions),
        "chance_count": sum(row.kind == "nature" for row in decisions),
        "player_count": sum(row.kind == "player" for row in decisions),
        "decisions": [asdict(row) for row in decisions],
    })
    return index, record, os.getpid()


def _validate(
    record: dict[str, Any], fixture: NatureFixture, manifest: dict[str, Any],
) -> dict[str, Any]:
    digest = record.get("record_sha256")
    if digest != canonical_sha256({
        k: v for k, v in record.items() if k != "record_sha256"
    }):
        raise ValueError("Playable Nature record checksum mismatch")
    if (
        record.get("schema") != PLAYABLE_RECORD_SCHEMA or
        record.get("manifest_sha256") != manifest["identity_sha256"] or
        record.get("fixture") != asdict(fixture) or
        record.get("fixture_fingerprint") != fixture.fingerprint or
        record.get("emulator_revision") != manifest["emulator_revision"] or
        record.get("representation") != "legacy-indexed-tickets" or
        record.get("probability_status") != "UNVERIFIED" or
        record.get("terminal_outcome_known") is not False or
        record.get("status") not in VALID_STATUSES
    ):
        raise ValueError("Playable Nature record provenance/status mismatch")
    rows = record.get("decisions")
    if (not isinstance(rows, list) or
            len(rows) != record.get("decision_count") or
            sum(row.get("kind") == "nature" for row in rows)
            != record.get("chance_count") or
            sum(row.get("kind") == "player" for row in rows)
            != record.get("player_count")):
        raise ValueError("Playable Nature record decision counts mismatch")
    try:
        typed = tuple(PlayableDecision(**row) for row in rows)
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed playable Nature trace") from exc
    for a, b in zip(typed, typed[1:]):
        if a.fingerprint_after != b.fingerprint_before:
            raise ValueError("Discontinuous playable Nature trajectory")
    if record["status"] == STATUS_TERMINAL:
        if (not isinstance(record.get("terminal_fingerprint"), str) or
                record["blocked_site"] is not None or
                record["blocked_operation"] is not None):
            raise ValueError("Terminal playable Nature record is inconsistent")
    elif record.get("terminal_fingerprint") is not None:
        raise ValueError("Censored or blocked Nature record has terminal fingerprint")
    if record["status"] == STATUS_UNSUPPORTED:
        if not record.get("blocked_site") or not record.get("blocked_operation"):
            raise ValueError("Nature blocker lacks site/operation")
    elif record.get("blocked_site") is not None or record.get("blocked_operation") is not None:
        raise ValueError("Unexpected Nature blocker site")
    return record


def collect_playable_coverage(
    fixtures: tuple[NatureFixture, ...], *, output_root: Path,
    emulator_command: tuple[str, ...], emulator_revision: str,
    emulator_artifact_sha256: str, nature_mode: str = "first",
    nature_seed: int = 29, max_boundaries: int = 512,
    workers: int = 15, resume: bool = False, progress_every: int = 10,
) -> dict[str, Any]:
    """Idempotent playable diagnostic corpus. No censoring-as-defeat."""
    if type(workers) is not int or workers < 1:
        raise ValueError("Positive process worker count required")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("Positive progress cadence required")
    if type(max_boundaries) is not int or max_boundaries <= 0:
        raise ValueError("Positive decision budget required")
    if not emulator_command:
        raise ValueError("Nature emulator command is required")
    fixtures = tuple(sorted(fixtures, key=lambda x: x.fixture_id))
    if not fixtures or len({f.fixture_id for f in fixtures}) != len(fixtures):
        raise ValueError("Empty or duplicated playable fixture ID")
    if any(f.kind != "initial_hp" for f in fixtures):
        raise ValueError("Playable Nature driver requires initial_hp source prefix")
    partitions: dict[tuple[str, str, int], str] = {}
    for f in fixtures:
        other = partitions.setdefault(f.source_key, f.partition)
        if f.partition != other:
            raise ValueError("Playable Nature source-run split leakage")
    if nature_mode not in ("first", "last", "uniform"):
        raise ValueError("Unsupported Nature proposal mode")
    if type(nature_seed) is not int:
        raise ValueError("Invalid Nature proposal seed")
    if len(emulator_artifact_sha256) != 64 or not emulator_revision:
        raise ValueError("Unpinned Nature emulator artifact or revision")
    identity = {
        "schema": PLAYABLE_CORPUS_SCHEMA,
        "emulator_revision": emulator_revision,
        "emulator_artifact_sha256": emulator_artifact_sha256,
        "chance_schema": PLAYABLE_SCHEMA,
        "nature_mode": nature_mode, "nature_seed": nature_seed,
        "player_mode": "first-legal-public-v1",
        "max_boundaries": max_boundaries,
        "fixture_count": len(fixtures),
        "source_run_count": len({f.source_key for f in fixtures}),
        "fixtures_sha256": canonical_sha256([asdict(f) for f in fixtures]),
    }
    manifest = {
        **identity, "identity_sha256": canonical_sha256(identity),
    }
    output_root = output_root.resolve()
    record_dir = output_root / "records"
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Existing playable corpus requires --resume")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Playable Nature resume manifest mismatch")
    else:
        if resume:
            raise ValueError("No existing playable manifest to resume")
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError("Nonempty Nature corpus directory")
        atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    record_dir.mkdir(exist_ok=True)
    expected = {
        record_dir / record_filename(i, f)
        for i, f in enumerate(fixtures)
    }
    if set(record_dir.iterdir()) - expected:
        raise ValueError("Unknown Nature corpus record file")
    missing = []
    for index, fixture in enumerate(fixtures):
        file = record_dir / record_filename(index, fixture)
        if file.exists():
            _validate(json.loads(file.read_text(encoding="utf-8")), fixture, manifest)
        else:
            missing.append((index, fixture))
    print(
        f"[nature-playable-n3] fixtures={len(fixtures)} already={len(fixtures)-len(missing)} "
        f"pending={len(missing)} workers={workers}",
        flush=True,
    )
    start = time.perf_counter()
    pids: set[int] = set()
    if missing:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(emulator_command, emulator_revision),
        ) as pool:
            jobs = {
                pool.submit(_run_job, i, f, manifest): (i, f)
                for i, f in missing
            }
            for completed, future in enumerate(as_completed(jobs), 1):
                index, fixture = jobs[future]
                try:
                    number, record, pid = future.result()
                    if number != index:
                        raise ValueError("Playable worker fixture index drift")
                    _validate(record, fixture, manifest)
                    atomic_write(
                        record_dir / record_filename(index, fixture),
                        json.dumps(record, sort_keys=True) + "\n",
                    )
                    pids.add(pid)
                    if completed % progress_every == 0 or completed == len(missing):
                        print(
                            f"[nature-playable-n3] completed={completed}/{len(missing)} "
                            f"last_status={record['status']} workers_observed={len(pids)}",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"[nature-playable-n3] FAILED fixture={fixture.fixture_id}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise
    records = [
        _validate(
            json.loads((record_dir / record_filename(i, f)).read_text(encoding="utf-8")),
            f, manifest,
        ) for i, f in enumerate(fixtures)
    ]
    statuses = {
        status: sum(row["status"] == status for row in records)
        for status in sorted(VALID_STATUSES)
    }
    blocked: dict[str, int] = {}
    for row in records:
        if row["status"] == STATUS_UNSUPPORTED:
            label = f"{row['blocked_site']} / {row['blocked_operation']}"
            blocked[label] = blocked.get(label, 0) + 1
    digest = canonical_sha256([row["record_sha256"] for row in records])
    atomic_write(
        output_root / "corpus.jsonl",
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in records),
    )
    summary = {
        "schema": "sts2-ai-nature-playable-coverage-summary-v1",
        "manifest_sha256": manifest["identity_sha256"],
        "records_digest": digest,
        "fixtures": len(fixtures),
        "source_runs": manifest["source_run_count"],
        "status_counts": statuses,
        "blocked_sites": dict(sorted(blocked.items())),
        "total_chance_decisions": sum(row["chance_count"] for row in records),
        "total_player_decisions": sum(row["player_count"] for row in records),
        "terminal_outcomes_classified": False,
        "representation": "legacy-indexed-tickets",
        "calibrated_native_reference_law": False,
    }
    atomic_write(
        output_root / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    seconds = time.perf_counter() - start
    execution = {
        "schema": "sts2-ai-nature-playable-coverage-execution-v1",
        "requested_workers": workers,
        "processes_observed": len(pids),
        "executed_fixtures": len(missing),
        "seconds": seconds, "fixtures_per_second": len(missing) / seconds,
        "records_digest": digest,
    }
    atomic_write(
        output_root / "execution.json",
        json.dumps(execution, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"[nature-playable-n3] DONE records={len(records)} "
        f"counts={statuses} blocked={blocked} digest={digest}",
        flush=True,
    )
    return {**summary, "execution": execution}
