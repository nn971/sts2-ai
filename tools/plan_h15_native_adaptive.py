#!/usr/bin/env python3
"""Resumable adaptive native-combat cohort planning for H15 PUCT training.

This tool selects only authentic CombatSnapshotRecipe entries. It does not
pretend that replayed native RNG is a fair PUCT continuation: the selected
cohort must be consumed by an independently certified native Chance Law search
backend. Fixed probes are drawn solely from the training corpus and never
serve as held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sts2_ai.evaluation.combat_snapshots import CombatSnapshotRecipe, load_recipes
from sts2_ai.training.native_adaptive_curriculum import (
    SCHEMA, NativeAdaptiveProgress, family_pools, feedback_means,
    fixed_probe_recipes, weighted_native_selection,
)

PLAN_SCHEMA = "sts2-h15-native-adaptive-round-plan-v1"
STATE_SCHEMA = "sts2-h15-native-adaptive-study-v1"


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) +
            "\n").encode("utf-8")


def jsonl_bytes(rows: Any) -> bytes:
    return b"".join(json_bytes(asdict(row)) for row in rows)


def write_once(path: Path, content: bytes) -> None:
    """Idempotent, no-clobber round manifests and cohorts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("Refusing to change an existing H15 plan: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(content)


def write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pending")
    with tmp.open("xb") as stream:
        stream.write(json_bytes(data))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def study_progress(
    path: Path, *, corpus_sha: str, families: tuple[str, ...],
    seed: int, probes_per_family: int,
) -> tuple[NativeAdaptiveProgress, int]:
    if not path.exists():
        return NativeAdaptiveProgress(families), 0
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (raw.get("schema") != STATE_SCHEMA
            or raw.get("corpus_sha256") != corpus_sha
            or raw.get("seed") != seed
            or raw.get("probes_per_family") != probes_per_family):
        raise ValueError("Cannot resume H15 adaptive study under changed provenance")
    progress = NativeAdaptiveProgress.from_dict(raw["progress"])
    if progress.families != families:
        raise ValueError("Adaptive encounter families changed across resume")
    next_round = raw.get("next_round")
    if type(next_round) is not int or next_round < 1:
        raise ValueError("Invalid adaptive study resume position")
    return progress, next_round


def read_feedback(
    path: Path, probes: tuple[CombatSnapshotRecipe, ...],
) -> dict[str, float]:
    """Consume one completed, unambiguous result for each fixed native probe."""
    expected = {(row.seed, row.exact_hash): row for row in probes}
    if len(expected) != len(probes):
        raise ValueError("Duplicate native fixed probe identity")
    actual: dict[tuple[str, str], SimpleNamespace] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Feedback line {line_number} must be an object")
            identity = (item.get("source_seed"), item.get("entry_hash"))
            if identity not in expected or identity in actual:
                raise ValueError("Unexpected or duplicate fixed-probe feedback")
            if item.get("outcome") not in ("victory", "defeat"):
                raise ValueError("Probe feedback lacks a terminal combat outcome")
            actual[identity] = SimpleNamespace(
                source_seed=identity[0], entry_hash=identity[1],
                outcome=item["outcome"], reward=item.get("reward"),
            )
    if set(actual) != set(expected):
        raise ValueError(
            f"Incomplete fixed-probe feedback: {len(actual)}/{len(expected)}"
        )
    return feedback_means(
        probes, [actual[(row.seed, row.exact_hash)] for row in probes],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "advance"))
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--combats", type=int, default=256)
    parser.add_argument("--probes-per-family", type=int, default=3)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--feedback", type=Path)
    args = parser.parse_args()

    if (args.round < 0 or args.seed < 0 or args.combats < 1
            or args.probes_per_family < 1):
        parser.error("Invalid adaptive H15 study configuration")
    if not args.train_corpus.is_file():
        parser.error("Missing authentic native training corpus")
    if args.operation == "plan":
        if args.teacher_checkpoint is None or not args.teacher_checkpoint.is_file():
            parser.error("Planning requires a pinned teacher checkpoint")
        if args.feedback is not None:
            parser.error("Use advance to record completed probe feedback")
    elif args.feedback is None or not args.feedback.is_file():
        parser.error("Advance requires completed fixed-probe feedback JSONL")

    corpus_sha = sha256_file(args.train_corpus)
    recipes = load_recipes(args.train_corpus)
    pools = family_pools(recipes)
    families = tuple(sorted(pools))
    progress, next_round = study_progress(
        args.study_root / "state.json", corpus_sha=corpus_sha,
        families=families, seed=args.seed,
        probes_per_family=args.probes_per_family,
    )
    if args.round != next_round:
        raise ValueError(
            f"Expected adaptive round {next_round}, got {args.round}"
        )
    root = args.study_root / f"round-{args.round:04d}"
    probes = fixed_probe_recipes(
        recipes, per_family=args.probes_per_family, study_seed=args.seed,
    )
    if args.operation == "plan":
        selected = weighted_native_selection(
            recipes, count=args.combats, study_seed=args.seed,
            round_index=args.round, weights=progress.weights,
        )
        manifest = {
            "schema": PLAN_SCHEMA,
            "round": args.round,
            "corpus_sha256": corpus_sha,
            "teacher_sha256": sha256_file(args.teacher_checkpoint),
            "seed": args.seed,
            "combats": args.combats,
            "probes_per_family": args.probes_per_family,
            "fixed_probe_count": len(probes),
            "encounter_weights": progress.weights,
        }
        write_once(root / "manifest.json", json_bytes(manifest))
        write_once(root / "train.jsonl", jsonl_bytes(selected))
        write_once(root / "probes.jsonl", jsonl_bytes(probes))
        print("[h15-adaptive-plan] " + json.dumps(
            {"round": args.round, "train_entries": len(selected),
             "probes": len(probes), "families": len(families),
             "teacher_sha256": manifest["teacher_sha256"],
             "output": str(root)}, sort_keys=True,
        ), flush=True)
        return

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Plan the H15 round before advancing it")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != PLAN_SCHEMA
            or manifest.get("round") != args.round
            or manifest.get("corpus_sha256") != corpus_sha
            or manifest.get("seed") != args.seed
            or manifest.get("probes_per_family") != args.probes_per_family
            or (root / "probes.jsonl").read_bytes() != jsonl_bytes(probes)):
        raise ValueError("Fixed-probe cohort or plan provenance changed")
    means = read_feedback(args.feedback, probes)
    summary = progress.update(means)
    next_state = {
        "schema": STATE_SCHEMA, "corpus_sha256": corpus_sha,
        "seed": args.seed, "probes_per_family": args.probes_per_family,
        "next_round": args.round + 1, "progress": progress.to_dict(),
        "last_feedback_sha256": sha256_file(args.feedback),
    }
    write_once(root / "feedback-summary.json", json_bytes(summary))
    write_state(args.study_root / "state.json", next_state)
    print("[h15-adaptive-advance] " + json.dumps(
        {"completed_round": args.round, "next_round": args.round + 1,
         "families": len(families), "weights": progress.weights},
        sort_keys=True,
    ), flush=True)


if __name__ == "__main__":
    main()
