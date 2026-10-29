#!/usr/bin/env python3
"""Collect H15 search labels under an explicitly declared H8 reference model.

This is a *combat-fixture* experiment, not a native full-run posterior.
Every search simulation starts a fresh H8 Reference Nature session with an
independent candidate seed and matches the entire observed combat prefix.

Example fish invocation is documented in docs/H15_COMBAT_SEARCH_GUIDED.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.h15_h8_reference import (
    H8_REFERENCE_MODEL,
    H8FixtureEntry,
    H8ReferenceCombatBackend,
    H8ReferenceNaturePosterior,
)
from sts2_ai.evaluation.combat_snapshots import POLICY
from sts2_ai.models.h15_policy_value import H15CombatEvaluator
from sts2_ai.nature.combat_scenarios import DECK_PRESETS, SCENARIOS
from sts2_ai.training.h15_native_search_collector import collect_live_combat_search

REPORT_SCHEMA = "sts2-h15-h8-reference-combat-collection-v1"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emulator-root", type=Path, required=True)
    p.add_argument("--emulator-revision", required=True)
    p.add_argument("--emulator-dll", type=Path, required=True)
    p.add_argument("--h9-checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--scenario",
        choices=sorted([
            key for key, value in SCENARIOS.items() if value.ascension == 10
        ]),
        default="mawler-a10",
    )
    p.add_argument("--deck", choices=DECK_PRESETS, default="silent-starter-v1")
    p.add_argument("--fixture-seed", default="h15-m3-fixed-fixture-v1")
    p.add_argument("--source-nature-seed", type=int, default=113)
    p.add_argument("--search-seed", type=int, default=20261029)
    p.add_argument("--simulations", type=int, default=8)
    p.add_argument("--max-searches", type=int, default=1)
    p.add_argument("--max-candidates", type=int, default=4096)
    p.add_argument("--max-decisions", type=int, default=512)
    p.add_argument("--tree-depth", type=int, default=16)
    p.add_argument("--rollout-depth", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if (not args.emulator_revision or len(args.emulator_revision) != 40
            or not 0 <= args.source_nature_seed < (1 << 63)
            or min(args.simulations, args.max_searches, args.max_candidates,
                   args.max_decisions, args.tree_depth, args.rollout_depth) < 1):
        p.error("Invalid H8 research source or search budget")
    if not args.emulator_dll.is_file():
        p.error("Emulator DLL does not exist")
    if not args.h9_checkpoint.is_file():
        p.error("Frozen H9 checkpoint does not exist")

    scenario = SCENARIOS[args.scenario]
    entry = H8FixtureEntry(
        seed=args.fixture_seed, encounter_id=scenario.encounter_id,
        native_entry=scenario.native_entry,
        total_floor=scenario.total_floor, act=scenario.act,
        ascension=scenario.ascension,
        first_monster_combat_id=scenario.first_monster_combat_id,
        map_column=scenario.map_column, map_row=scenario.map_row,
        fixture_deck_preset_id=args.deck,
    )
    dll_hash = hashlib.sha256(args.emulator_dll.read_bytes()).hexdigest()
    model_hash = hashlib.sha256(args.h9_checkpoint.read_bytes()).hexdigest()
    config = {
        "schema": REPORT_SCHEMA,
        "chance_model": H8_REFERENCE_MODEL,
        "emulator_revision": args.emulator_revision,
        "emulator_dll_sha256": dll_hash,
        "h9_teacher_sha256": model_hash,
        "fixture": asdict(entry),
        "source_nature_seed": args.source_nature_seed,
        "search_seed": args.search_seed,
        "simulations": args.simulations,
        "max_candidates": args.max_candidates,
        "max_searches": args.max_searches,
        "max_decisions": args.max_decisions,
        "tree_depth": args.tree_depth,
        "rollout_depth": args.rollout_depth,
    }
    print("[h15-h8-reference-preflight] " + json.dumps(config, sort_keys=True), flush=True)
    if not args.execute:
        print("[h15-h8-reference-preflight] pass --execute for reference search", flush=True)
        return

    binding_dir = args.emulator_root / "bindings/python"
    if not binding_dir.is_dir():
        p.error("Missing H8 emulator Python binding")
    sys.path.insert(0, str(binding_dir))
    from sts2_emulator import Session  # type: ignore[import-not-found]

    teacher = H15CombatEvaluator.load(args.h9_checkpoint, device=args.device)
    with Session(["dotnet", str(args.emulator_dll), "prototype-ai-jsonl"]) as session:
        backend = H8ReferenceCombatBackend(
            session, entry, emulator_revision=args.emulator_revision,
        )
        sampler = H8ReferenceNaturePosterior(
            backend, max_candidates=args.max_candidates,
        )
        actual = backend.begin_reference(args.source_nature_seed)
        try:
            observation = backend.observe(actual, POLICY)
            history = (PublicHistoryStep(
                observation, None, backend.legal_actions(actual),
            ),)
        except BaseException:
            backend.release_many((actual,))
            raise
        report = collect_live_combat_search(
            backend, actual, history,
            fair_sampler=sampler, teacher=teacher,
            source_seed=args.fixture_seed + ":reference:" + str(args.source_nature_seed),
            entry_hash=observation.observation_hash,
            encounter_tier=(
                "boss" if "boss" in args.scenario
                else "weak" if args.scenario in ("shrinker-a10", "nibbits-a10")
                else "normal"
            ),
            simulations=args.simulations,
            max_decisions=args.max_decisions, max_searches=args.max_searches,
            seed=args.search_seed, tree_depth=args.tree_depth,
            rollout_depth=args.rollout_depth,
        )
        if backend.live_handles:
            raise RuntimeError("H8 search leaked an owned emulator handle")

    args.output_root.mkdir(parents=True, exist_ok=True)
    data_path = args.output_root / "targets.jsonl"
    if data_path.exists():
        raise FileExistsError("Refusing to overwrite existing H15 search targets")
    with data_path.open("w", encoding="utf-8") as f:
        for sample in report.targets:
            sample["chance_prior_id"] = H8_REFERENCE_MODEL
            sample["experiment_regime"] = "fixed-h8-fixture-reference-nature"
            f.write(json.dumps(sample, sort_keys=True) + "\n")
    result = {
        **config,
        "outcome": report.outcome, "realized_combat_reward": report.reward,
        "exit_hp": report.exit_hp, "searched_decisions": report.searched_decisions,
        "combat_decisions": report.combat_decisions,
        "nested_choice_uniform_fallbacks":
            teacher.public_selection_fallbacks,
        "targets_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
    }
    (args.output_root / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print("[h15-h8-reference-result] " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
