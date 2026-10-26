#!/usr/bin/env python3
"""Replay selected H6 macro runs into public-only JSON, JSONL, HTML and Markdown.

Replays use the pinned semantic emulator, not the native-game inspector.
From a comparison JSON, select interesting 80-vs-160 cases automatically;
or pass individual --seed values. This never mutates training checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sts2_ai.agents.phase_split_agent import PhaseSplitGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.episode_goal import NATIVE_ACT1_BOSS_GOAL
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, require_environment
from sts2_ai.evaluation import play_run
from sts2_ai.evaluation.macro_trace import (
    REPORT_SCHEMA,
    TraceRecorder,
    render_html,
    render_markdown,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.corpus import atomic_write, file_sha256

PINNED_EMULATOR = "0469424fc20679fd81c0d5e4ee8d9a484ec19108"


def select_seeds(comparison: dict[str, Any], labels: tuple[str, str],
                 count: int) -> list[str]:
    """Choose mixed success/failure pairs deterministically, without cherry-picking wins."""
    worlds = comparison["worlds"]
    one, two = worlds[labels[0]], worlds[labels[1]]
    if len(one) != len(two):
        raise ValueError("Unpaired confirmation results")
    categories: dict[str, list[tuple[float, str]]] = {
        "first_wins": [], "second_wins": [],
        "both_boss_fail": [], "both_early_fail": [],
    }
    for a, b in zip(one, two, strict=True):
        if a["seed"] != b["seed"] or a["censored"] or b["censored"]:
            continue
        aw, bw = a["act1_cleared"] is True, b["act1_cleared"] is True
        key = (
            "first_wins" if aw and not bw else
            "second_wins" if bw and not aw else
            "both_boss_fail" if not aw and not bw
            and a["floor"] == 16 and b["floor"] == 16 else
            "both_early_fail" if not aw and not bw
            and a["floor"] <= 8 and b["floor"] <= 8 else None
        )
        if key is None:
            continue
        metric = abs(a["frontier_progress"] - b["frontier_progress"])
        categories[key].append((metric, a["seed"]))
    for entries in categories.values():
        entries.sort(reverse=True)
    # Rotate categories to avoid exclusively sampling cases favorable to H6-C-80.
    selected: list[str] = []
    keys = tuple(categories)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if categories[key] and len(selected) < count:
                selected.append(categories[key].pop(0)[1])
                progressed = True
        if not progressed:
            break
    return selected


def _write_trace(path: Path, trace: dict[str, Any]) -> None:
    atomic_write(Path(str(path) + ".json"), json.dumps(
        trace, indent=2, sort_keys=True,
    ) + "\n")
    header = {key: value for key, value in trace.items()
              if key not in ("decisions", "map_snapshots")}
    lines = [json.dumps({"type": "metadata", **header}, sort_keys=True)]
    for row in trace["map_snapshots"]:
        lines.append(json.dumps({"type": "map", **row}, sort_keys=True))
    for row in trace["decisions"]:
        lines.append(json.dumps({"type": "decision", **row}, sort_keys=True))
    atomic_write(Path(str(path) + ".jsonl"), "\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", action="append", required=True,
                   help="Exactly two label=model.json pairs")
    p.add_argument("--seed", action="append", default=[])
    p.add_argument("--comparison", type=Path,
                   help="Existing paired comparison JSON for automatic seed selection")
    p.add_argument("--sample", type=int, default=10)
    p.add_argument("--max-decisions", type=int, default=4096)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--emulator-revision", default=PINNED_EMULATOR)
    args = p.parse_args()
    if len(args.player) != 2 or args.sample < 0 or args.max_decisions < 1:
        p.error("Require exactly two players and valid positive max-decisions")
    paths = {}
    for raw in args.player:
        label, delim, model_path = raw.partition("=")
        if not delim or not label or label in paths or not Path(model_path).is_file():
            p.error("Expected two distinct label=existing-model.json players")
        paths[label] = Path(model_path)
    labels = tuple(paths)
    models = {key: PhaseSplitNeuralModel.load(path)
              for key, path in paths.items()}
    frozen = models[labels[0]].combat.to_dict()
    if models[labels[1]].combat.to_dict() != frozen:
        p.error("The frozen combat models differ")
    seeds = list(dict.fromkeys(args.seed))
    comparison = None
    if args.comparison:
        comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
        config = comparison.get("config", {})
        if (config.get("emulator_revision") != args.emulator_revision or
                config.get("ascension") != 10):
            p.error("Confirmation emulator revision / ascension mismatch")
        for label in labels:
            actual = file_sha256(paths[label])
            if config.get("player_models", {}).get(label) != actual:
                p.error(f"Comparison checkpoint fingerprint mismatch: {label}")
        seeds.extend(select_seeds(comparison, labels, args.sample))
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        p.error("Provide --seed or --comparison with available samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with JsonlEmulatorBackend(
        expected_emulator_revision=args.emulator_revision, build=False,
    ) as backend:
        require_environment(backend, NATIVE_OVERGROWTH)
        manifest = []
        for seed in seeds:
            traces = []
            for label in labels:
                model = models[label]
                recorder = TraceRecorder(model, model_label=label)
                result = play_run(
                    backend, PhaseSplitGreedyAgent(model), seed=seed,
                    policy=InformationPolicy(FAIR_POLICY_ID), ascension=10,
                    environment=NATIVE_OVERGROWTH,
                    episode_goal_version=NATIVE_ACT1_BOSS_GOAL,
                    max_decisions=args.max_decisions,
                    decision_observer=recorder,
                )
                trace = recorder.export(
                    seed, result, backend.emulator_revision,
                    file_sha256(paths[label]),
                )
                traces.append(trace)
                basename = args.output_dir / (seed + "." + label)
                _write_trace(basename, trace)
                print(f"[h6c-replay] {seed} {label}: {result.outcome} "
                      f"floor={result.terminal_floor} "
                      f"decisions={len(trace['decisions'])}", flush=True)
            html_path = args.output_dir / (seed + ".html")
            md_path = args.output_dir / (seed + ".md")
            atomic_write(html_path, render_html(traces))
            atomic_write(md_path, render_markdown(traces))
            manifest.append({
                "seed": seed, "html": str(html_path), "markdown": str(md_path),
                "outcomes": {t["label"]: t["outcome"] for t in traces},
                "boss": {t["label"]: t["boss_identity"] for t in traces},
            })
        atomic_write(args.output_dir / "manifest.json", json.dumps({
            "schema": REPORT_SCHEMA, "emulator_revision": backend.emulator_revision,
            "models": {key: file_sha256(path) for key, path in paths.items()},
            "runs": manifest,
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
