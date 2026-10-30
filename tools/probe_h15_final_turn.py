#!/usr/bin/env python3
"""H15 A10 difficult-combat screen and complete final-player-turn action audit.

Choose a boss/deck fixture by *empirically measured* frozen-policy win rate,
then independently validate its rate and record every actual action of the
last turn (including defeats). No search, optimizer updates, original-game
RNG parity, or survival-biased case selection. All outcomes are under the
emulator's declared fixed-fixture ReferenceNature chance law.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from sts2_ai.nature.combat_scenarios import (
    DECK_PRESETS,
    SCENARIOS,
    deck_for_seed,
    scenario_for_seed,
)

SCHEMA = "sts2-h15-reference-final-turn-probe-v1"
DEFAULT_SCENARIOS = "ceremonial-beast-boss-a10,vantom-boss-a10"
# The fixed H8 boss encounter requires a source-eligible Boss room.
# The emulator gives that room identity only to boss-conditioned decks
# (or to a separate explicit ResearchCombatFixture, unused here).
BOSS_DECKS = (
    "silent-boss-cycle-20-v1",
    "silent-boss-poison-20-v1",
)
DEFAULT_DECKS = ",".join(BOSS_DECKS)


def validate_boss_sources(
    scenarios: tuple[str, ...], decks: tuple[str, ...],
) -> None:
    """Reject H8 boss-entry / normal-room combinations before worker launch."""
    if (not scenarios or not decks
            or len(scenarios) != len(set(scenarios))
            or len(decks) != len(set(decks))
            or any(s not in SCENARIOS or SCENARIOS[s].ascension != 10
                   or SCENARIOS[s].total_floor != 6 for s in scenarios)
            or any(d not in DECK_PRESETS or d not in BOSS_DECKS for d in decks)):
        raise ValueError(
            "Only A10 boss encounters and boss-conditioned deck presets "
            "are valid for the H8 fixed boss-room entry"
        )


def choose_closest_half(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Choose from wins, never from hindsight decisions or hidden states."""
    if not entries or any(
        type(e.get("wins")) is not int or type(e.get("combats")) is not int
        or e["combats"] < 1 or not 0 <= e["wins"] <= e["combats"]
        for e in entries
    ):
        raise ValueError("Expected completed win-rate screening cohorts")
    # Prefer evidence closer to 50%; break ties by more source cases and stable ID.
    return min(
        entries,
        key=lambda e: (
            abs(e["wins"] / e["combats"] - 0.5),
            -e["combats"], e["scenario"], e["deck"],
        ),
    )


def wilson_interval(wins: int, trials: int) -> tuple[float, float]:
    if not 0 <= wins <= trials or trials < 1:
        raise ValueError("Invalid binomial win-rate data")
    p = wins / trials
    z = 1.959963984540054
    factor = 1 + z * z / trials
    middle = (p + z * z / (2 * trials)) / factor
    margin = z / factor * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2))
    return max(0.0, middle - margin), min(1.0, middle + margin)


def case_ids_for_pair(
    prefix: str, scenario: str, deck: str, n: int,
    scenario_pool: tuple[str, ...], deck_pool: tuple[str, ...],
) -> list[str]:
    """Condition scenario assignment without selecting favorable Nature outcomes."""
    if n < 1 or scenario not in scenario_pool or deck not in deck_pool:
        raise ValueError("Unsupported final-turn source cohort")
    ids = []
    index = 0
    while len(ids) < n:
        candidate = f"{prefix}:candidate:{index:06d}"
        index += 1
        if (scenario_for_seed(candidate, scenario_pool).scenario_id == scenario
                and deck_for_seed(candidate, deck_pool) == deck):
            ids.append(candidate)
        if index > 1000000:
            raise RuntimeError("Cannot populate the requested source condition")
    return ids


def summarize_final_turns(records: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for record in records:
        trace = record.get("final_player_turn")
        if not isinstance(trace, dict) or not trace.get("actions"):
            raise ValueError("Missing actual completed-combat final-turn trace")
        last_action = trace["actions"][-1]
        entries.append({
            "case_id": record["case_id"], "encounter": record["encounter"],
            "deck": record["deck"], "outcome": record["outcome"],
            "exit_hp": record["exit_hp"], "realized_reward": record["reward"],
            "final_turn": trace["turn"], "actions": trace["actions"],
            "final_turn_decisions": len(trace["actions"]),
            "first_final_turn_critic_reward": (
                trace["actions"][0]["predicted_encounter_reward"]
            ),
            "last_decision_critic_reward": (
                last_action["predicted_encounter_reward"]
            ),
        })
    errors = [
        abs(e["first_final_turn_critic_reward"] - e["realized_reward"])
        for e in entries
    ]
    return {
        "cases": entries, "total_cases": len(entries),
        "wins": sum(e["outcome"] == "victory" for e in entries),
        "defeats": sum(e["outcome"] == "defeat" for e in entries),
        "mean_final_turn_decisions": sum(
            e["final_turn_decisions"] for e in entries
        ) / len(entries),
        "mean_first_final_turn_critic_absolute_error": sum(errors) / len(errors),
    }


def main() -> None:
    from train_h15_reference_rounds import (
        aggregate,
        atomic_json,
        collect_cases,
        sha256,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--decks", default=DEFAULT_DECKS)
    parser.add_argument("--screen-per-pair", type=int, default=12)
    parser.add_argument("--validate-combats", type=int, default=32)
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--near-half-tolerance", type=float, default=0.2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (min(args.round, args.screen_per_pair, args.validate_combats, args.workers) < 1
            or not 0 <= args.near_half_tolerance < 0.5):
        parser.error("Invalid final-turn calibration/evaluation budget")

    scenarios = tuple(x.strip() for x in args.scenarios.split(","))
    decks = tuple(x.strip() for x in args.decks.split(","))
    try:
        validate_boss_sources(scenarios, decks)
    except ValueError as exc:
        parser.error(str(exc))

    original = args.source_run.resolve()
    if args.round < 1:
        parser.error("Round must be positive")
    source_cfg = original / "config.json"
    folder = original / f"round-{args.round:04d}"
    checkpoint = folder / "student.pt"
    model_hash = sha256(checkpoint)
    source_config = json.loads(source_cfg.read_text(encoding="utf-8"))
    round_record = json.loads((folder / "round.json").read_text(encoding="utf-8"))
    if round_record["updated_sha256"] != model_hash:
        raise ValueError("Source H15 checkpoint does not match completed round")

    cfg = dict(source_config)
    if sha256(Path(cfg["emulator_dll"])) != cfg["emulator_dll_sha256"]:
        raise ValueError("Pinned H15 source emulator DLL has changed")
    cfg.update({
        "scenarios": list(scenarios), "decks": list(decks),
        "workers": args.workers, "max_searches": 0,
        "search_schedule": "first", "record_final_turn": True,
    })
    manifest = {
        "schema": SCHEMA, "frozen_model_sha256": model_hash,
        "source_config_sha256": sha256(source_cfg),
        "source_round": args.round, "emulator_revision": cfg["emulator_revision"],
        "emulator_dll_sha256": cfg["emulator_dll_sha256"],
        "chance_prior": "h15-h8-fixed-fixture-declared-chance-law-v1",
        "scenarios": list(scenarios), "decks": list(decks),
        "screen_per_pair": args.screen_per_pair,
        "validate_combats": args.validate_combats, "workers": args.workers,
        "near_half_tolerance": args.near_half_tolerance,
        "code_sha256": {
            name: sha256(Path(__file__).resolve().parents[1] / name)
            for name in (
                "tools/probe_h15_final_turn.py",
                "tools/train_h15_reference_rounds.py",
                "src/sts2_ai/training/h15_native_search_collector.py",
                "src/sts2_ai/emulator/h15_h8_reference.py",
                "src/sts2_ai/models/h15_policy_value.py",
            )
        },
    }
    print("[h15-finalturn-preflight] " + json.dumps(manifest, sort_keys=True), flush=True)
    if not args.execute:
        print("[h15-finalturn-preflight] pass --execute for emulator reference combats")
        return

    out = args.output_root.resolve()
    if out == original or out in original.parents or original in out.parents:
        raise ValueError("Final-turn probe requires new, separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "manifest.json"
    if manifest_file.exists():
        if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Incompatible resumed final-turn probe or code revision")
    else:
        atomic_json(manifest_file, manifest)

    # Run the whole screening grid in a *single* 15-worker emulator cohort.
    # All fixtures are chosen by source IDs before any outcome is observed.
    ids_by_pair: dict[tuple[str, str], list[str]] = {}
    for scenario in scenarios:
        for deck in decks:
            ids_by_pair[(scenario, deck)] = case_ids_for_pair(
                f"h15-finalturn-screen-v1:{scenario}:{deck}",
                scenario, deck, args.screen_per_pair, scenarios, decks,
            )
    screen_ids = [
        case for pair_ids in ids_by_pair.values() for case in pair_ids
    ]
    screen_records = collect_cases(
        cfg, checkpoint, screen_ids, "greedy", out / "screen",
    )
    by_case = {r["case_id"]: r for r in screen_records}
    screened: list[dict[str, Any]] = []
    for (scenario, deck), ids in ids_by_pair.items():
        records = [by_case[case] for case in ids]
        if any(r["encounter"] != scenario or r["deck"] != deck for r in records):
            raise ValueError("Screening case differs from selected A10 fixture")
        wins = sum(r["outcome"] == "victory" for r in records)
        screened.append({
            "scenario": scenario, "deck": deck, "wins": wins,
            "combats": len(records), "win_rate": wins / len(records),
            "wilson_95": wilson_interval(wins, len(records)),
            "mean_exit_hp": aggregate(records)["mean_exit_hp"],
        })
    atomic_json(out / "progress.json", {
        "schema": SCHEMA, "phase": "screened", "screening": screened,
    })
    selected = choose_closest_half(screened)
    closest_distance = abs(selected["win_rate"] - 0.5)
    if closest_distance > args.near_half_tolerance:
        print(
            "[h15-finalturn-warning] No screened scenario/deck combination "
            "has observed win rate near 50%; validating the closest one, "
            "without claiming a near-half fixture.",
            flush=True,
        )

    ids = case_ids_for_pair(
        f"h15-finalturn-validation-v1:{selected['scenario']}:{selected['deck']}",
        selected["scenario"], selected["deck"], args.validate_combats,
        scenarios, decks,
    )
    records = collect_cases(
        cfg, checkpoint, ids, "greedy", out / "validation",
    )
    validation = aggregate(records)
    validation_wilson = wilson_interval(validation["victories"], len(records))
    traces = summarize_final_turns(records)
    atomic_json(out / "final-turn-actions.json", traces)
    result = {
        "schema": SCHEMA, "phase": "complete",
        "source_model_sha256": model_hash,
        "screening": screened, "selected_fixture": selected,
        "screened_near_half": closest_distance <= args.near_half_tolerance,
        "validation": {**validation, "wilson_95": validation_wilson},
        "validated_near_half": (
            abs(validation["win_rate"] - 0.5) <= args.near_half_tolerance
        ),
        "last_turn_summary": {
            key: value for key, value in traces.items() if key != "cases"
        },
        "last_turn_actions_file": str(out / "final-turn-actions.json"),
    }
    atomic_json(out / "progress.json", result)
    print("[h15-finalturn-complete] " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
