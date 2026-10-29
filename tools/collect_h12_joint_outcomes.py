#!/usr/bin/env python3
"""H12 M3: collect H6-C macro + frozen H9 combat native A10 joint outcomes.

This is a passive collection, not a counterfactual simulation. Every source
run has a disjoint seed. The output is an append-only, resumable JSONL
journal with checkpoint/model/emulator SHA identities. Give training and
holdout runs DIFFERENT seed prefixes and output journals.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import torch

from sts2_ai.agents.h10_hybrid import H10FrozenHybridAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.nature.corpus import file_sha256
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.training.h12_transition_data import PublicJointRunTrace

SCHEMA = "sts2-h12-native-joint-collection-v1"
_BACKEND: JsonlEmulatorBackend | None = None
_AGENT: H10FrozenHybridAgent | None = None
_IDENTITY: dict[str, str] | None = None


def _cleanup() -> None:
    global _BACKEND
    if _BACKEND is not None:
        _BACKEND.close()
        _BACKEND = None


def _init(emulator_root: str, revision: str, macro: str, combat: str,
          identity: dict[str, str]) -> None:
    global _BACKEND, _AGENT, _IDENTITY
    torch.set_num_threads(1)
    _BACKEND = JsonlEmulatorBackend(
        emulator_root=Path(emulator_root),
        expected_emulator_revision=revision,
        build=False,
    )
    atexit.register(_cleanup)
    _AGENT = H10FrozenHybridAgent.load(
        macro_model=Path(macro), combat_checkpoint=Path(combat),
    )
    _IDENTITY = identity


def _run(task: tuple[int, str, int]) -> tuple[int, dict[str, Any]]:
    index, seed, max_decisions = task
    if _BACKEND is None or _AGENT is None or _IDENTITY is None:
        raise RuntimeError("H12 worker was not initialized")
    trace = PublicJointRunTrace(seed)
    result = play_run(
        _BACKEND, _AGENT,
        seed=seed, ascension=10, max_decisions=max_decisions,
        policy=InformationPolicy(FAIR_POLICY_ID),
        environment=NATIVE_OVERGROWTH,
        episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        public_frame_observer=trace.observe,
        decision_observer=lambda handle, obs, legal, decision: trace.record_decision(
            obs, legal, decision,
        ),
    )
    return index, trace.finish(
        outcome=result.outcome, censored=result.censored,
        act1_clear=result.act1_cleared is True,
        frontier_progress=result.frontier_progress,
        decisions=result.decisions,
        policy_identity=_IDENTITY,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-macro-model", type=Path, required=True)
    p.add_argument("--combat-checkpoint", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-prefix", required=True)
    p.add_argument("--runs", type=int, default=1024)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (args.runs < 1 or args.start_index < 0 or args.workers < 1
            or args.max_decisions < 1 or not args.seed_prefix):
        p.error("Invalid source-run collection parameters")
    revision = subprocess.check_output(
        ["git", "-C", str(args.emulator_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != args.emulator_revision:
        p.error("Emulator revision mismatch")
    if not args.emulator_dll.is_file():
        p.error("Missing pinned emulator DLL")
    identity = {
        "combat_checkpoint_sha256": file_sha256(args.combat_checkpoint),
        "macro_checkpoint_sha256": file_sha256(args.baseline_macro_model),
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "emulator_revision": revision,
    }
    manifest = {
        "schema": SCHEMA,
        "source_seed_prefix": args.seed_prefix,
        "max_decisions": args.max_decisions,
        "identity": identity,
        "fingerprint": canonical_sha256([
            SCHEMA, args.seed_prefix, args.max_decisions, identity,
        ]),
    }
    print("[h12-collect] " + json.dumps({
        "execute": args.execute, "fingerprint": manifest["fingerprint"],
        "runs": args.runs, "workers": args.workers,
    }, sort_keys=True), flush=True)
    if not args.execute:
        return
    results: set[int] = set()
    if args.output.exists():
        if not args.resume:
            p.error("Existing journal requires --resume")
        with args.output.open(encoding="utf-8") as stream:
            first = json.loads(next(stream))
            if first != manifest:
                p.error("Resume journal identity mismatch")
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                index = row["source_run_index"]
                if (type(index) is not int or index in results or index < 0
                        or row["source_run_seed"] !=
                            f"{args.seed_prefix}-{index}"):
                    p.error("Invalid or duplicated source-run record")
                results.add(index)
    else:
        if args.resume:
            p.error("Resume journal missing")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    todo = [
        (i, f"{args.seed_prefix}-{i}", args.max_decisions)
        for i in range(args.start_index, args.start_index + args.runs)
        if i not in results
    ]
    if not todo:
        print("[h12-collect] all requested source runs already present", flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"),
        initializer=_init,
        initargs=(
            str(args.emulator_root.resolve()), revision,
            str(args.baseline_macro_model.resolve()),
            str(args.combat_checkpoint.resolve()), identity,
        ),
    ) as pool, args.output.open("a", encoding="utf-8") as stream:
        tasks = [pool.submit(_run, task) for task in todo]
        for task in as_completed(tasks):
            index, row = task.result()
            if index in results:
                raise ValueError("Duplicate H12 source-run result")
            row["source_run_index"] = index
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            results.add(index)
            print("[h12-collect] " + json.dumps({
                "complete": len(results),
                "seed": row["source_run_seed"],
                "outcome": row["outcome"],
                "censored": row["censored"],
                "combats": len(row["combats"]),
                "macro_states": len(row["macro_states"]),
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
