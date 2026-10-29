#!/usr/bin/env python3
"""Paired, source-seed-disjoint native A10 Act-clear evaluation for H11 macro PPO."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import torch

from sts2_ai.agents.h10_hybrid import H10FrozenHybridAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.evaluation import play_run
from sts2_ai.models.h7_public_policy import TorchH7PublicPolicy
from sts2_ai.models.shared_transformer import SharedObjectTransformer
from sts2_ai.nature.corpus import file_sha256
from sts2_ai.nature.protocol import canonical_sha256
from train_h11_shared_macro import FrozenCombatSampledMacro, _atomic_json, _macro_trainable

SCHEMA = "sts2-h11-native-fullrun-paired-v1"
_BACKEND = None
_BASELINE = None
_CANDIDATE = None


def _cleanup():
    global _BACKEND
    if _BACKEND is not None:
        _BACKEND.close()
        _BACKEND = None


def _init(emulator_root: str, revision: str, baseline_macro: str,
          frozen_combat: str, trained_macro: str):
    global _BACKEND, _BASELINE, _CANDIDATE
    torch.set_num_threads(1)
    _BACKEND = JsonlEmulatorBackend(
        emulator_root=Path(emulator_root),
        expected_emulator_revision=revision, build=False,
    )
    atexit.register(_cleanup)
    _BASELINE = H10FrozenHybridAgent.load(
        macro_model=Path(baseline_macro),
        combat_checkpoint=Path(frozen_combat),
    )
    combat = SharedObjectTransformer.load_checkpoint(Path(frozen_combat))
    trained = SharedObjectTransformer.load_checkpoint(Path(trained_macro))
    for name, original in combat.state_dict().items():
        if not _macro_trainable(name) and not torch.equal(original, trained.state_dict()[name]):
            raise ValueError(f"H11 trained checkpoint changed frozen combat/shared weight {name}")
    _CANDIDATE = TorchH7PublicPolicy(
        trained.eval(), model_id=file_sha256(Path(trained_macro)),
    )


def _run(pair: tuple[int, str, int]) -> tuple[int, dict]:
    index, seed, max_decisions = pair
    if _BACKEND is None or _BASELINE is None or _CANDIDATE is None:
        raise RuntimeError("H11 evaluation worker not initialized")
    def evaluate(agent):
        result = play_run(
            _BACKEND, agent, seed=seed,
            policy=InformationPolicy(FAIR_POLICY_ID),
            ascension=10, max_decisions=max_decisions,
            environment=NATIVE_OVERGROWTH,
            episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
        )
        return {
            "outcome": result.outcome,
            "censored": result.censored,
            "act1_clear": result.act1_cleared,
            "terminal_floor": result.terminal_floor,
            "frontier_progress": result.frontier_progress,
            "decisions": result.decisions,
            "terminal_hp": result.hp_trajectory[-1] if result.hp_trajectory else None,
        }
    baseline = evaluate(_BASELINE)
    candidate = evaluate(FrozenCombatSampledMacro(
        _CANDIDATE, seed=seed, temperature=1.0, greedy_macro=True,
    ))
    return index, {"seed": seed, "baseline": baseline, "candidate": candidate}


def _summarize(rows: list[dict], label: str) -> dict:
    values = [row[label] for row in rows]
    completed = [x for x in values if not x["censored"]]
    return {
        "runs": len(values),
        "censored": len(values) - len(completed),
        "act1_clears": sum(x["act1_clear"] is True for x in completed),
        "clear_rate_all_runs": (sum(x["act1_clear"] is True for x in completed)
                                / len(values)),
        "mean_frontier_progress": sum(x["frontier_progress"] for x in values) / len(values),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-macro-model", type=Path, required=True)
    p.add_argument("--combat-checkpoint", type=Path, required=True)
    p.add_argument("--macro-checkpoint", type=Path, required=True)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--runs", type=int, default=64)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--seed-prefix", default="h11-shared-macro-a10-heldout-v1")
    args = p.parse_args()
    if min(args.runs, args.workers, args.max_decisions) < 1 or not args.seed_prefix:
        p.error("Invalid full-run evaluation cohort")
    from subprocess import check_output
    revision = check_output(
        ["git", "-C", str(args.emulator_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if revision != args.emulator_revision:
        p.error("Emulator revision mismatch")
    if args.report.exists():
        p.error("Refusing to overwrite completed held-out report")
    identity = {
        "schema": SCHEMA, "revision": revision, "runs": args.runs,
        "seed_prefix": args.seed_prefix, "max_decisions": args.max_decisions,
        "baseline_macro": file_sha256(args.baseline_macro_model),
        "frozen_combat": file_sha256(args.combat_checkpoint),
        "trained_macro": file_sha256(args.macro_checkpoint),
    }
    fingerprint = canonical_sha256(identity)
    journal = Path(str(args.report) + ".partial.jsonl")
    results: dict[int, dict] = {}
    if journal.exists():
        with journal.open(encoding="utf-8") as stream:
            first = json.loads(next(stream))
            if first != {"fingerprint": fingerprint, "identity": identity}:
                p.error("Held-out journal fingerprint mismatch")
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                at = item["index"]
                if at in results or not 0 <= at < args.runs:
                    p.error("Invalid or duplicated held-out journal entry")
                results[at] = item["row"]
    else:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with journal.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps({"fingerprint": fingerprint, "identity": identity}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    todo = [(i, f"{args.seed_prefix}-{i}", args.max_decisions)
            for i in range(args.runs) if i not in results]
    if todo:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=get_context("spawn"),
            initializer=_init,
            initargs=(
                str(args.emulator_root), revision,
                str(args.baseline_macro_model.resolve()),
                str(args.combat_checkpoint.resolve()),
                str(args.macro_checkpoint.resolve()),
            ),
        ) as pool, journal.open("a", encoding="utf-8") as stream:
            futures = [pool.submit(_run, request) for request in todo]
            for future in as_completed(futures):
                at, row = future.result()
                if at in results:
                    raise ValueError("Duplicated held-out source seed")
                stream.write(json.dumps({"index": at, "row": row}, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                results[at] = row
                print(f"[h11-eval] {len(results)}/{args.runs} "
                      f"seed={row['seed']} "
                      f"h6={row['baseline']['outcome']} "
                      f"h11={row['candidate']['outcome']}", flush=True)
    rows = [results[i] for i in range(args.runs)]
    report = {
        "schema": SCHEMA, "identity": identity, "fingerprint": fingerprint,
        "baseline": _summarize(rows, "baseline"),
        "candidate": _summarize(rows, "candidate"), "rows": rows,
    }
    _atomic_json(args.report, report)
    print(json.dumps({"report": str(args.report),
                      "baseline": report["baseline"],
                      "candidate": report["candidate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
