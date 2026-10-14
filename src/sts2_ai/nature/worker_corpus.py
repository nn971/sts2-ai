"""Process-isolated deterministic N3 corpus collection from the *existing* HP/intent bridge.

Each Python worker owns its own long-lived emulator JSONL child. The bridge
is deliberately fixture-only; this does not run player actions or PPO.
Workers never share emulator handles. All persisted records are independent
of completion order, process IDs and worker count.
"""
from __future__ import annotations

import json
import multiprocessing
import os
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

from .corpus import (
    RECORD_SCHEMA,
    NatureFixture,
    atomic_write,
    load_record,
    manifest_identity,
    record_filename,
    seal_record,
    validated_record,
)
from .policies import FirstLegalNature, LastLegalNature, ScriptedNature, UniformNature
from .protocol import canonical_sha256
from .trace import ChanceTraceEntry, ChanceTranscript, record_entry

_worker_wire: NatureJsonlProcess | None = None
_worker_bridge: IsolatedNatureBridge | None = None


def _close_worker_wire() -> None:
    global _worker_wire, _worker_bridge
    if _worker_wire is not None:
        _worker_wire.close()
        _worker_wire = None
        _worker_bridge = None


def _init_worker(command: tuple[str, ...], emulator_revision: str) -> None:
    global _worker_wire, _worker_bridge
    if _worker_wire is not None:
        raise RuntimeError("Reinitializing a live Nature worker is forbidden")
    _worker_wire = NatureJsonlProcess(list(command))
    try:
        _worker_bridge = IsolatedNatureBridge(
            _worker_wire, emulator_revision=emulator_revision,
        )
    except BaseException:
        _close_worker_wire()
        raise
    # multiprocessing worker processes bypass normal atexit on some platforms.
    Finalize(None, _close_worker_wire, exitpriority=10)


def _policy(mode: str, seed: int, fixture: NatureFixture):
    if mode == "first":
        return FirstLegalNature()
    if mode == "last":
        return LastLegalNature()
    if mode == "uniform":
        derived = int(canonical_sha256([seed, fixture.fingerprint])[:16], 16)
        return UniformNature(seed=derived)
    raise ValueError("Unsupported Nature policy")


def _execute(
    bridge: IsolatedNatureBridge, fixture: NatureFixture, policy: Any,
    *, max_choices: int,
) -> tuple[ChanceTranscript, str]:
    boundary = bridge.begin(**fixture.emulator_kwargs())
    handles = [boundary]
    entries: list[ChanceTraceEntry] = []
    try:
        while True:
            while not boundary.complete:
                if len(entries) >= max_choices:
                    raise ControlledChanceUnavailable("Nature fixture decision cap exceeded")
                if boundary.pending is None:
                    raise ControlledChanceUnavailable("Missing Nature pending choice")
                proposal = policy.propose(boundary.pending)
                proposal.validate(boundary.pending)
                entries.append(record_entry(boundary.pending, proposal))
                boundary = bridge.resolve(boundary, proposal.selected_index)
                handles.append(boundary)
            if boundary.kind == "initial_hp":
                boundary = bridge.advance(boundary)
                handles.append(boundary)
            else:
                break
        transcript = ChanceTranscript(
            emulator_revision=bridge.emulator_revision,
            fixture_id=fixture.fixture_id,
            nature_policy_id=policy.policy_id,
            entries=tuple(entries),
        )
        if not entries:
            raise ControlledChanceUnavailable("Fixture had no externally selected chance events")
        return transcript, boundary.fingerprint
    finally:
        bridge.release(*handles)


def _run_worker_job(
    index: int, fixture: NatureFixture, manifest: dict[str, Any],
) -> tuple[int, dict[str, Any], int]:
    if _worker_bridge is None:
        raise RuntimeError("Nature worker did not initialize bridge")
    policy = _policy(manifest["nature_mode"], manifest["nature_seed"], fixture)
    transcript, fingerprint = _execute(
        _worker_bridge, fixture, policy, max_choices=manifest["max_choices"],
    )
    replay = ScriptedNature.from_selections(transcript.replay_script())
    replay_trace, replay_fingerprint = _execute(
        _worker_bridge, fixture, replay, max_choices=manifest["max_choices"],
    )
    replay.assert_complete()
    if replay_fingerprint != fingerprint or (
        replay_trace.replay_script() != transcript.replay_script()
    ):
        raise ValueError(f"Nature exact replay divergence for {fixture.fixture_id}")
    result = seal_record({
        "schema": RECORD_SCHEMA,
        "manifest_sha256": manifest["identity_sha256"],
        "emulator_revision": manifest["emulator_revision"],
        "fixture": asdict(fixture),
        "fixture_fingerprint": fixture.fingerprint,
        "status": "ISOLATED_COMPLETE",
        "playable_combat": False,
        "calibrated_native_reference_law": False,
        "nature_policy_id": policy.policy_id,
        "decision_count": len(transcript.entries),
        "transcript": json.loads(transcript.to_json()),
        "final_fingerprint": fingerprint,
        "replay_fingerprint": replay_fingerprint,
    })
    return index, result, os.getpid()


def collect_isolated_corpus(
    fixtures: tuple[NatureFixture, ...], *, output_root: Path,
    emulator_command: tuple[str, ...], emulator_revision: str,
    emulator_artifact_sha256: str, nature_mode: str = "uniform",
    nature_seed: int = 29, max_choices: int = 128,
    workers: int = 15, resume: bool = False, progress_every: int = 10,
) -> dict[str, Any]:
    """Collect source-ordered completed records with durable --resume.

    Worker count does not affect experiment identity: a pending run may
    resume at a different worker count and yield identical record content.
    All records must pass provenance and replay validation before publishing
    the canonical JSONL corpus.
    """
    if type(workers) is not int or workers <= 0:
        raise ValueError("Positive Nature process count required")
    if type(progress_every) is not int or progress_every <= 0:
        raise ValueError("Positive Nature progress cadence required")
    if not emulator_command:
        raise ValueError("Explicit Nature emulator process command required")
    ordered = tuple(sorted(fixtures, key=lambda x: x.fixture_id))
    if not ordered or len({f.fixture_id for f in ordered}) != len(ordered):
        raise ValueError("Empty/duplicate Nature fixtures")
    source_partitions: dict[tuple[str, str, int], str] = {}
    for fixture in ordered:
        previous = source_partitions.setdefault(fixture.source_key, fixture.partition)
        if previous != fixture.partition:
            raise ValueError("Nature fixture source-run split leakage")
    manifest = manifest_identity(
        ordered, emulator_revision=emulator_revision,
        emulator_artifact_sha256=emulator_artifact_sha256,
        nature_mode=nature_mode, nature_seed=nature_seed,
        max_choices=max_choices,
    )
    output_root = output_root.resolve()
    manifest_path = output_root / "manifest.json"
    record_dir = output_root / "records"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Existing Nature output requires --resume")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Nature resume manifest mismatch")
    else:
        if resume:
            raise ValueError("Nature --resume has no manifest")
        if output_root.exists() and any(output_root.iterdir()):
            raise ValueError("Nature output path is not empty")
        atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    record_dir.mkdir(exist_ok=True)
    expected_paths = {
        record_dir / record_filename(i, fixture)
        for i, fixture in enumerate(ordered)
    }
    unknown = set(record_dir.iterdir()) - expected_paths
    if unknown:
        raise ValueError(f"Unknown Nature record files: {sorted(str(x) for x in unknown)[:3]}")
    missing: list[tuple[int, NatureFixture]] = []
    for index, fixture in enumerate(ordered):
        path = record_dir / record_filename(index, fixture)
        if path.exists():
            load_record(path, fixture, manifest)
        else:
            missing.append((index, fixture))
    print(
        f"[nature-n3] fixtures={len(ordered)} completed={len(ordered) - len(missing)} "
        f"pending={len(missing)} workers={workers} mode={nature_mode}",
        flush=True,
    )
    started = time.perf_counter()
    observed_pids: set[int] = set()
    if missing:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx,
            initializer=_init_worker,
            initargs=(emulator_command, emulator_revision),
        ) as pool:
            futures = {
                pool.submit(_run_worker_job, i, fixture, manifest): (i, fixture)
                for i, fixture in missing
            }
            completed = len(ordered) - len(missing)
            for future in as_completed(futures):
                index, fixture = futures[future]
                try:
                    returned_index, result, pid = future.result()
                    if returned_index != index:
                        raise ValueError("Nature worker returned wrong job identity")
                    validated_record(result, fixture, manifest)
                    path = record_dir / record_filename(index, fixture)
                    if path.exists():
                        raise ValueError("Nature worker attempted to overwrite a fixture")
                    atomic_write(path, json.dumps(result, sort_keys=True) + "\n")
                    observed_pids.add(pid)
                    completed += 1
                    if completed % progress_every == 0 or completed == len(ordered):
                        print(
                            f"[nature-n3] complete={completed}/{len(ordered)} "
                            f"active_worker_pids={len(observed_pids)}",
                            flush=True,
                        )
                except Exception as exc:
                    print(
                        f"[nature-n3] FAILED fixture={fixture.fixture_id} "
                        f"type={type(exc).__name__}: {exc}", flush=True,
                    )
                    raise
    records = [
        load_record(record_dir / record_filename(i, fixture), fixture, manifest)
        for i, fixture in enumerate(ordered)
    ]
    corpus_text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in records
    )
    atomic_write(output_root / "corpus.jsonl", corpus_text)
    digest = canonical_sha256([row["record_sha256"] for row in records])
    summary = {
        "schema": "sts2-ai-nature-isolated-corpus-summary-v1",
        "manifest_sha256": manifest["identity_sha256"],
        "fixture_count": len(ordered),
        "source_run_count": manifest["source_run_count"],
        "decision_count": sum(row["decision_count"] for row in records),
        "records_digest": digest,
        "status": "ISOLATED_COMPLETE",
        "playable_combat": False,
        "calibrated_native_reference_law": False,
    }
    atomic_write(output_root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    execution = {
        "schema": "sts2-ai-nature-isolated-execution-v1",
        "workers_requested": workers,
        "worker_processes_observed_this_invocation": len(observed_pids),
        "fixtures_executed_this_invocation": len(missing),
        "seconds_this_invocation": time.perf_counter() - started,
        "manifest_sha256": manifest["identity_sha256"],
        "records_digest": digest,
    }
    atomic_write(\n        output_root / "execution.json",\n        json.dumps(execution, indent=2, sort_keys=True) + "\n",\n    )
    print(f"[nature-n3] DONE records_digest={digest} output={output_root}", flush=True)
    return {**summary, "execution": execution}
