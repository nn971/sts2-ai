#!/usr/bin/env python3
"""Paired H4 greedy vs UCT comparison on genuine C# boss combats.

MCTS uses public player actions and the emulator's declared reference Chance
Law for unresolved stochastic sites. The current semantic continuation is
root-conditioned, so the search is a simulator baseline, NOT a fully
belief-state-correct fair-information benchmark.

The benchmark uses a common set of source worlds and checkpoints but evaluates
search with fresh chance draws: equal source seeds do not force the same
mid-combat draws after policies take different actions.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sts2_ai.emulator.semantic_p0_frozen_player import FrozenPlayerSpec
from sts2_ai.nature.combat_scenarios import parse_scenario_pool, parse_deck_pool
from sts2_ai.nature.contextual_draw import restore_nature
from sts2_ai.nature.corpus import atomic_write, file_sha256
from sts2_ai.nature.draw_only_workers import draw_only_batch
from sts2_ai.nature.h2_benchmark import balanced_h2_source_seeds
from sts2_ai.nature.h4_reward import H4_REWARD
from sts2_ai.nature.ppo_draw import PpoDrawNature
from sts2_ai.nature.protocol import canonical_sha256


def _nature(path: Path | None) -> PpoDrawNature:
    if path is None:
        return PpoDrawNature(seed=109, mix=0.75)
    if path.suffix == ".pt":
        import torch
        raw = torch.load(path, map_location="cpu", weights_only=True)
        payload = raw["nature"]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))["nature"]
    nature = restore_nature(payload)
    if type(nature) is not PpoDrawNature:
        raise ValueError("Expected an H2/H4 contextual PPO Nature checkpoint")
    return nature


def _summarize(episodes: tuple) -> dict:
    groups = sorted({(e.scenario_id, e.deck_preset_id) for e in episodes})
    return {
        "score": statistics.mean(e.score for e in episodes),
        "victory": statistics.mean(e.outcome == "victory" for e in episodes),
        "count": len(episodes),
        "cells": {
            "/".join(cell): {
                "score": statistics.mean(e.score for e in episodes
                                         if (e.scenario_id, e.deck_preset_id) == cell),
                "victory": statistics.mean(
                    e.outcome == "victory" for e in episodes
                    if (e.scenario_id, e.deck_preset_id) == cell),
            }
            for cell in groups
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--player", action="append", required=True,
                   help="Label=path to exported player JSON; may repeat")
    p.add_argument("--nature-checkpoint", type=Path)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--emulator-bindings", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenario-pool", default=(
        "ceremonial-beast-boss-a10,vantom-boss-a10"))
    p.add_argument("--deck-pool", default=(
        "silent-boss-cycle-20-v1,silent-boss-poison-20-v1"))
    p.add_argument("--eval-episodes", type=int, default=120)
    p.add_argument("--eval-split", choices=("selection", "confirmation"),
                   default="selection")
    p.add_argument("--mcts-simulations", type=int, default=16)
    p.add_argument("--mcts-max-boundaries", type=int, default=1200)
    p.add_argument("--workers", type=int, default=15)
    p.add_argument("--actor-seed", type=int, default=31337)
    p.add_argument("--reference-seed", type=int, default=23387)
    p.add_argument("--max-boundaries", type=int, default=1200)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    scenarios = parse_scenario_pool(args.scenario_pool)
    decks = parse_deck_pool(args.deck_pool)
    if (args.eval_episodes < 4 or
            args.eval_episodes % (len(scenarios) * len(decks)) or
            args.mcts_simulations <= 0 or args.workers <= 0 or
            args.mcts_max_boundaries <= 0 or args.max_boundaries <= 0):
        p.error("Invalid paired comparison study parameters")
    models = {}
    for entry in args.player:
        label, sep, location = entry.partition("=")
        if not sep or not label or label in models:
            p.error("Use unique --player label=path")
        path = Path(location)
        if not path.is_file():
            p.error(f"Player checkpoint missing: {path}")
        models[label] = path
    if not args.emulator_dll.is_file():
        p.error("Pinned emulator DLL missing")
    if not (args.emulator_bindings / "sts2_emulator" / "__init__.py").is_file():
        p.error("Missing emulator Python bindings")
    if args.nature_checkpoint and not args.nature_checkpoint.is_file():
        p.error("Nature checkpoint missing")
    seeds = balanced_h2_source_seeds(
        args.eval_split, args.eval_episodes, scenarios, decks,
    )
    config = {
        "schema": "sts2-ai-h4-greedy-mcts-paired-v1",
        "source_digest": canonical_sha256(seeds),
        "split": args.eval_split,
        "reward_version": H4_REWARD,
        "player_models": {key: file_sha256(path) for key, path in models.items()},
        "nature_checkpoint_sha256": (file_sha256(args.nature_checkpoint)
                                     if args.nature_checkpoint else None),
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": file_sha256(args.emulator_dll),
        "scenarios": scenarios,
        "decks": decks,
        "episodes": args.eval_episodes,
        "mcts_simulations": args.mcts_simulations,
        "mcts_max_boundaries": args.mcts_max_boundaries,
        "workers": args.workers,
    }
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    nature = _nature(args.nature_checkpoint)
    command = ("dotnet", str(args.emulator_dll.resolve()), "prototype-ai-jsonl")
    reports = {}
    trajectories = {}
    for label, path in models.items():
        spec = FrozenPlayerSpec.from_file(
            path, seed=args.actor_seed, temperature=1.0,
            greedy=True, action_scope="all-legal-v1",
        )
        for policy, sim_count in (("greedy", 0),
                                  (f"uct-{args.mcts_simulations}",
                                   args.mcts_simulations)):
            for arm in ("reference", "learned"):
                result = draw_only_batch(
                    seeds, nature=nature, spec=spec,
                    emulator_command=command,
                    emulator_bindings=args.emulator_bindings,
                    emulator_revision=args.emulator_revision,
                    reference_seed=args.reference_seed,
                    workers=args.workers,
                    max_boundaries=args.max_boundaries,
                    scenario_pool=scenarios, deck_pool=decks,
                    draw_mode=arm,
                    reward_version=H4_REWARD,
                    mcts_simulations=sim_count,
                    mcts_max_boundaries=args.mcts_max_boundaries,
                    progress_every=max(1, args.eval_episodes // 4),
                )
                episodes = result["episodes"]
                for expected_seed, episode in zip(seeds, episodes, strict=True):
                    if episode.source_seed != expected_seed or (
                            episode.reward_version != H4_REWARD):
                        raise RuntimeError("Comparison lost original source or reward version")
                key = f"{label}/{policy}/{arm}"
                reports[key] = {
                    **_summarize(episodes),
                    "rollout_seconds": result["elapsed_seconds"],
                }
                trajectories[key] = episodes
                print(f"[h4-mcts] {key}: score={reports[key]['score']:.4f} "
                      f"win={reports[key]['victory']:.3f} "
                      f"seconds={result['elapsed_seconds']:.1f}", flush=True)
        for arm in ("reference", "learned"):
            greedy = trajectories[f"{label}/greedy/{arm}"]
            searched = trajectories[f"{label}/uct-{args.mcts_simulations}/{arm}"]
            key = f"{label}/uct-minus-greedy/{arm}"
            reports[key] = {
                "paired_score_delta": statistics.mean(
                    b.score - a.score for a, b in zip(greedy, searched, strict=True)),
                "paired_victory_delta": statistics.mean(
                    (b.outcome == "victory") - (a.outcome == "victory")
                    for a, b in zip(greedy, searched, strict=True)),
            }
    atomic_write(args.output, json.dumps({
        "config": config, "results": reports,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
