#!/usr/bin/env python3
"""Paired, resumable native Act-1 checkpoint selection and fresh final validation.

Selection: same NEW 256 seeds for all candidate checkpoint models.
Final: separate NEW 512 seeds, comparing only the preselected winner to round40.
These stages never use the historical 128-seed cohort for model selection.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.evaluation.checkpoint_selection import (
    GOAL, JOURNAL_SCHEMA, REPORT_SCHEMA, analyze,
)

EXPECTED_REVISION = "9117b4af09f0164a19bb1c41b70f688c8948e0e3"
_WORKER_BACKEND: JsonlEmulatorBackend | None = None
_WORKER_AGENTS: dict[str, PhaseSplitGreedyAgent] = {}


def model_specs(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    result = {}
    for label, path in sorted(paths.items()):
        resolved = path.resolve(strict=True)
        # Validate model format and sizes before launching worker processes.
        PhaseSplitGreedyAgent.load(resolved)
        result[label] = {
            "path": str(resolved),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }
    return result


def final_candidates(selection: dict[str, Any]) -> dict[str, dict[str, str]]:
    if selection.get("schema") != REPORT_SCHEMA or selection.get("mode") != "selection":
        raise ValueError("Final evaluation requires a completed selection report")
    chosen = selection["analysis"]["selected_label"]
    reference = selection["reference_label"]
    specs = selection["identity"]["model_specs"]
    if chosen not in specs or reference not in specs:
        raise ValueError("Selected checkpoint is missing from selection manifest")
    return {label: specs[label] for label in sorted({chosen, reference})}


def identity_for(
    *, mode: str, specs: dict[str, dict[str, str]], prefix: str,
    n: int, max_decisions: int, emulator_revision: str,
    reference_label: str, selection_sha256: str | None,
) -> dict[str, Any]:
    if mode not in ("selection", "final"):
        raise ValueError("Invalid checkpoint evaluation mode")
    if n <= 0 or max_decisions <= 0 or not prefix:
        raise ValueError("Invalid evaluation cohort")
    return {
        "mode": mode, "model_specs": specs,
        "seed_prefix": prefix, "seeds": n,
        "max_decisions": max_decisions,
        "emulator_revision": emulator_revision,
        "environment": NATIVE_OVERGROWTH,
        "goal": GOAL, "fair_policy": FAIR_POLICY_ID,
        "reference_label": reference_label,
        "selection_report_sha256": selection_sha256,
    }


def load_journal(path: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        header_line = stream.readline()
        if not header_line:
            raise ValueError("Empty evaluation journal")
        header = json.loads(header_line)
        if header != {"schema": JOURNAL_SCHEMA, "identity": identity}:
            raise ValueError("Evaluation journal identity mismatch; refusing mixed runs")
        rows = []
        for raw in stream:
            if not raw.strip():
                continue
            record = json.loads(raw)
            index = len(rows)
            if record.get("index") != index or record.get("seed") != (
                f"{identity['seed_prefix']}-{index}"
            ):
                raise ValueError("Evaluation journal has missing/duplicated seed rows")
            if set(record.get("models", {})) != set(identity["model_specs"]):
                raise ValueError("Evaluation journal contains unexpected models")
            rows.append(record)
        if len(rows) > identity["seeds"]:
            raise ValueError("Evaluation journal exceeds configured seed count")
        return rows


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".pending", delete=False,
        ) as output:
            tmp = Path(output.name)
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _init_worker(
    repo_root: str, emulator_root: str, revision: str,
    specs: dict[str, dict[str, str]],
) -> None:
    global _WORKER_AGENTS, _WORKER_BACKEND
    _WORKER_BACKEND = JsonlEmulatorBackend(
        repo_root=Path(repo_root),
        emulator_root=Path(emulator_root),
        expected_emulator_revision=revision, build=False,
    )
    atexit.register(_WORKER_BACKEND.close)
    _WORKER_AGENTS = {
        name: PhaseSplitGreedyAgent.load(Path(spec["path"]))
        for name, spec in specs.items()
    }


def _evaluate_seed(request: tuple[int, str]) -> dict[str, Any]:
    index, seed = request
    if _WORKER_BACKEND is None:
        raise RuntimeError("Evaluation worker backend not initialized")
    by_model: dict[str, dict[str, Any]] = {}
    for name, agent in _WORKER_AGENTS.items():
        run = play_run(
            _WORKER_BACKEND, agent, seed=seed,
            policy=InformationPolicy(FAIR_POLICY_ID),
            max_decisions=4096,
            environment=NATIVE_OVERGROWTH,
            episode_goal_version=GOAL,
        )
        by_model[name] = {
            "seed": run.seed, "outcome": run.outcome,
            "episode_goal_version": run.episode_goal_version,
            "act1_cleared": run.act1_cleared, "censored": run.censored,
            "terminal_act": run.terminal_act,
            "terminal_floor": run.terminal_floor,
            "frontier_progress": run.frontier_progress,
            "decisions": run.decisions,
            "wall_seconds": run.wall_seconds,
            "agent_compute_seconds": run.agent_compute_seconds,
            "boss_progress": (
                asdict(run.boss_progress) if run.boss_progress is not None else None
            ),
        }
    return {"index": index, "seed": seed, "models": by_model}


def evaluate(
    *, identity: dict[str, Any], output: Path, workers: int,
    build: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("Workers must be positive")
    if output.exists():
        raise FileExistsError(
            f"Completed report already exists: {output}. "
            "Use a new output path for a new experiment."
        )
    journal = Path(str(output) + ".partial.jsonl")
    previous = load_journal(journal, identity)
    print(f"[act1-eval] resuming at {len(previous)}/{identity['seeds']}", flush=True)

    # One build/handshake on the main process; workers only start prebuilt CLI.
    with JsonlEmulatorBackend(build=build) as probe:
        revision = probe.emulator_revision
        if revision != identity["emulator_revision"]:
            raise ValueError("Emulator revision differs from evaluation manifest")
        repo_root = str(probe._repo_root)
        emulator_root = str(probe._emulator_root)
    if not journal.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"schema": JOURNAL_SCHEMA, "identity": identity},
                                    sort_keys=True) + "\n")
    requests = [
        (i, f"{identity['seed_prefix']}-{i}")
        for i in range(len(previous), identity["seeds"])
    ]
    rows = previous[:]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(repo_root, emulator_root, revision, identity["model_specs"]),
    ) as pool:
        with journal.open("a", encoding="utf-8") as stream:
            for record in pool.map(_evaluate_seed, requests, chunksize=1):
                rows.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                if len(rows) % 10 == 0 or len(rows) == identity["seeds"]:
                    wins = {
                        name: sum(x["models"][name]["act1_cleared"] is True for x in rows)
                        for name in identity["model_specs"]
                    }
                    print(
                        f"[act1-eval] {len(rows)}/{identity['seeds']} "
                        f"certified clears={wins}", flush=True,
                    )

    grouped = {
        name: [record["models"][name] for record in rows]
        for name in identity["model_specs"]
    }
    analysis = analyze(
        grouped,
        mode=identity["mode"],
        reference_label=identity["reference_label"],
    )
    report = {
        "schema": REPORT_SCHEMA, "mode": identity["mode"],
        "reference_label": identity["reference_label"],
        "identity": identity, "analysis": analysis, "runs": rows,
    }
    save_report(output, report)
    print(
        f"[act1-eval] complete: {output}; "
        f"selected={analysis['selected_label']}; "
        f"summary={analysis['models']}", flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("selection", "final"), required=True)
    parser.add_argument("--model", action="append", default=[],
                        help="For selection only: LABEL=checkpoint.json; repeat")
    parser.add_argument("--selection-report", type=Path,
                        help="For final only: output of the selection cohort")
    parser.add_argument("--reference-label", default="round40")
    parser.add_argument("--seed-prefix", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-decisions", type=int, default=4096)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_decisions != 4096:
        parser.error("The fixed evaluation worker requires --max-decisions 4096")
    if args.mode == "selection":
        if args.selection_report is not None or not args.model:
            parser.error("Selection requires --model entries, not --selection-report")
        paths: dict[str, Path] = {}
        for item in args.model:
            name, sep, path = item.partition("=")
            if not sep or not name or not path or name in paths:
                parser.error("Each --model must be a unique LABEL=path")
            paths[name] = Path(path)
        if args.reference_label not in paths or len(paths) < 2:
            parser.error("Selection needs round40 plus at least one competitor")
        specs = model_specs(paths)
        selection_sha = None
    else:
        if args.model or args.selection_report is None:
            parser.error("Final needs --selection-report and no --model")
        selection_bytes = args.selection_report.read_bytes()
        prior = json.loads(selection_bytes)
        if prior.get("identity", {}).get("seed_prefix") == args.seed_prefix:
            parser.error("Final evaluation must use a NEW seed prefix")
        if prior.get("reference_label") != args.reference_label:
            parser.error("Reference label differs from selection report")
        specs = final_candidates(prior)
        paths = {label: Path(spec["path"]) for label, spec in specs.items()}
        if model_specs(paths) != specs:
            parser.error("Checkpoint model bytes/paths differ from selection")
        selection_sha = hashlib.sha256(selection_bytes).hexdigest()

    if args.seed_prefix in ("ppo-v7-v8-slyfix-eval", "ppo-v7-v8-slyfix-train"):
        parser.error("Do not reuse the original 128-seed evaluation/training cohort")
    if args.seeds < 1 or args.workers < 1:
        parser.error("Seeds and workers must be positive")
    identity = identity_for(
        mode=args.mode, specs=specs, prefix=args.seed_prefix,
        n=args.seeds, max_decisions=args.max_decisions,
        emulator_revision=EXPECTED_REVISION,
        reference_label=args.reference_label,
        selection_sha256=selection_sha,
    )
    evaluate(
        identity=identity, output=args.output,
        workers=args.workers, build=args.build,
    )


if __name__ == "__main__":
    main()
