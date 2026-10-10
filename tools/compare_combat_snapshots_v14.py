#!/usr/bin/env python3
"""Matched v8 / v12-width32 / v13-attention combat-snapshot comparison.

Evaluates each model once per authentic combat state, sharing deterministic
collector prefix replay across seven policies. Produces JSON, not HTML.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import JsonlEmulatorBackend
from sts2_ai.evaluation.combat_snapshots import (
    build_report,
    evaluate_model_suite,
    load_recipes,
    pair_model_suite,
)
from sts2_ai.models.neural import TACTICAL_ATTENTION_FORMAT

SUITE_SCHEMA = "sts2-combat-model-suite-v14"
REPLICAS = ("r1", "r2", "r3")


def model_paths(results: Path, *, attention_only: bool = False) -> dict[str, Path]:
    paths = {
        "v8": results / "ppo-v8-batched-40/models/stage-0120.json",
    }
    for rep in REPLICAS:
        if not attention_only:
            paths[f"h32-{rep}"] = (
                results / f"ppo-capacity-v12/{rep}/h32/models/stage-0080.json"
            )
        paths[f"attn-{rep}"] = (
            results / f"ppo-enemy-attention-v13/{rep}/attn/models/stage-0080.json"
        )
    return paths


def _hash(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while part := handle.read(1 << 20):
            sha.update(part)
    return sha.hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Atomic completed JSON files; never publish a partially written report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".incomplete")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def preflight(
    paths: dict[str, Path],
) -> tuple[dict[str, PhaseSplitGreedyAgent], dict[str, str]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen model files (no new training is needed):\n"
            + "\n".join(missing)
        )
    fingerprints = {label: _hash(path) for label, path in paths.items()}
    if len(set(fingerprints.values())) != len(fingerprints):
        raise ValueError("Different model labels resolve to identical checkpoint bytes")
    agents = {label: PhaseSplitGreedyAgent.load(path) for label, path in paths.items()}
    for label, agent in agents.items():
        is_attention = agent._model.combat.format_id == TACTICAL_ATTENTION_FORMAT
        if is_attention != label.startswith("attn-"):
            raise ValueError(
                f"{label}: checkpoint's tactical format does not match its intended architecture"
            )
    return agents, fingerprints


def _comparison(
    rows: Sequence[Any], baseline: str, candidate: str, *,
    revision: str, corpus: Path, paths: dict[str, Path],
    fingerprints: dict[str, str],
) -> dict[str, Any]:
    report = build_report(
        pair_model_suite(rows, baseline, candidate),
        emulator_revision=revision,
        corpus=str(corpus),
        baseline_model=str(paths[baseline]),
        candidate_model=str(paths[candidate]),
    )
    report["baseline_sha256"] = fingerprints[baseline]
    report["candidate_sha256"] = fingerprints[candidate]
    return report


def _summary_line(name: str, report: dict[str, Any]) -> str:
    s = report["summary"]
    multi = report["multi_enemy"]
    boss = report["by_tier"].get("boss", {})
    def fmt(value: float | None, digits: int = 3) -> str:
        return f"{value:+.{digits}f}" if value is not None else "n/a"

    return (
        f"{name:<17} "
        f"all={fmt(s['delta_win_rate'])} "
        f"boss={fmt(boss.get('delta_win_rate'))} "
        f"multi={fmt(multi['delta_win_rate'])} "
        f"HP={fmt(s['delta_exit_hp_all'], 2)} "
        f"discordant=+{s['candidate_only_wins']}/-{s['baseline_only_wins']}"
    )


def run(args: argparse.Namespace) -> None:
    paths = model_paths(args.results, attention_only=args.attention_only)
    agents, fingerprints = preflight(paths)
    recipes = load_recipes(args.corpus)
    if not recipes:
        raise ValueError("Empty snapshot corpus")
    revisions = {recipe.emulator_revision for recipe in recipes}
    if len(revisions) != 1:
        raise ValueError("Mixed emulator revisions are forbidden")
    corpus_hash = _hash(args.corpus)
    print(
        f"[suite] {len(recipes)} authentic combats, {len(agents)} checkpoints; "
        f"emulator={next(iter(revisions))[:12]}",
        flush=True,
    )
    for label, path in paths.items():
        print(f"[suite] {label:<9} {path} sha256={fingerprints[label][:12]}", flush=True)
    if args.dry_run:
        print("[suite] preflight complete; no emulator launched", flush=True)
        return
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory contains files: {args.output}; use --overwrite "
            "only after preserving previous experiment reports"
        )
    with JsonlEmulatorBackend(build=args.build) as backend:
        if backend.emulator_revision not in revisions:
            raise ValueError("Snapshot corpus was produced by a different emulator")
        completed_runs = 0

        def progress(seed: str, count: int) -> None:
            nonlocal completed_runs
            completed_runs += 1
            print(
                f"[suite] source_run={completed_runs} seed={seed} "
                f"comparisons={count * len(agents)}",
                flush=True,
            )

        rows = evaluate_model_suite(
            backend, recipes, agents,
            max_combat_decisions=args.max_combat_decisions,
            on_run_complete=progress,
        )
        revision = backend.emulator_revision
    suite = {
        "schema": SUITE_SCHEMA,
        "emulator_revision": revision,
        "corpus_sha256": corpus_hash,
        "corpus": str(args.corpus),
        "model_paths": {name: str(path) for name, path in paths.items()},
        "model_sha256": fingerprints,
        "scenarios": len(rows),
        "rows": [asdict(row) for row in rows],
    }
    _write_json(args.output / "suite.json", suite)
    comparisons: dict[str, dict[str, Any]] = {}
    for rep in REPLICAS:
        for baseline in (["v8"] if args.attention_only else ["v8", f"h32-{rep}"]):
            candidate = f"attn-{rep}"
            key = f"{baseline}-vs-{candidate}"
            report = _comparison(
                rows, baseline, candidate, revision=revision,
                corpus=args.corpus, paths=paths, fingerprints=fingerprints,
            )
            comparisons[key] = report
            _write_json(args.output / f"{key}.json", report)
            print("[suite] " + _summary_line(key, report), flush=True)
    aggregate = {
        "schema": SUITE_SCHEMA + "-summary",
        "emulator_revision": revision,
        "corpus_sha256": corpus_hash,
        "replicas": list(REPLICAS),
        "note": (
            "One authentic v8-sourced corpus; each checkpoint evaluated on "
            "identical exact entry states. Replicas are separate training runs."
        ),
        "comparisons": {
            key: {
                "baseline_model": report["baseline_model"],
                "candidate_model": report["candidate_model"],
                "summary": report["summary"],
                "boss": report["by_tier"].get("boss"),
                "multi_enemy": report["multi_enemy"],
                "by_progress_and_tier": report["by_progress_and_tier"],
            }
            for key, report in comparisons.items()
        },
    }
    _write_json(args.output / "summary.json", aggregate)
    print(f"[suite] wrote {len(comparisons)} comparisons and summary to {args.output}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="Root containing frozen v8, v12, v13 results")
    parser.add_argument("--corpus", type=Path,
                        default=Path("results/combat-snapshots-v14/source-A.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/combat-snapshots-v14/v14-model-suite"))
    parser.add_argument("--attention-only", action="store_true",
                        help="Skip matched v12 controls when files are unavailable; "
                             "attention effect then remains confounded with more PPO rounds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify model paths, formats and corpus before emulator work")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--max-combat-decisions", type=int, default=256)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"[suite] ERROR: {exc}\n")


if __name__ == "__main__":
    main()
