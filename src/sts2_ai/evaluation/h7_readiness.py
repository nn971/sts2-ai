"""Auditable H7 real-emulator rollout throughput and provenance diagnostics.

This reports mechanical and throughput evidence, never player quality.
It is intentionally independent of Torch and the C# emulator runtime.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from sts2_ai.nature.h5_curriculum import TRAIN_CELLS, cell_for_source

REPORT_SCHEMA = "sts2-h7-rollout-readiness-v1"


def make_readiness_report(
    cohort: dict[str, Any],
    seeds: tuple[str, ...],
    *,
    workers: int,
    model_sha256: str,
    parameter_count: int,
) -> dict[str, Any]:
    """Check all source worlds before reporting full-size inference throughput."""
    episodes = cohort["episodes"]
    elapsed = cohort["elapsed_seconds"]
    processes = cohort["process_count"]
    if (not seeds or len(seeds) != len(set(seeds)) or len(episodes) != len(seeds)
            or type(workers) is not int or workers < 1
            or type(parameter_count) is not int or parameter_count < 1
            or not isinstance(model_sha256, str) or len(model_sha256) != 64
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed <= 0
            or type(processes) is not int or not 1 <= processes <= workers):
        raise ValueError("Invalid H7 source-complete throughput cohort")

    cells: Counter[str] = Counter()
    player_decisions = 0
    total_boundaries = 0
    victories = 0
    for expected_seed, episode in zip(seeds, episodes, strict=True):
        cell = cell_for_source(expected_seed)
        if (cell not in TRAIN_CELLS
                or episode.source_seed != expected_seed
                or episode.scenario_id != cell.scenario
                or episode.deck_preset_id != cell.deck
                or episode.draw_mode != "reference"
                or episode.outcome not in ("victory", "defeat")
                or not episode.terminal_replay_verified):
            raise ValueError("H7 throughput used an unverified or uncurated source")
        cells[cell.key] += 1
        victories += episode.outcome == "victory"
        total_boundaries += len(episode.decisions)
        player_decisions += sum(
            step.phase in ("player", "player_choice")
            for step in episode.decisions
        )
    if set(cells) != {cell.key for cell in TRAIN_CELLS} or player_decisions < 1:
        raise ValueError("H7 throughput cohort must cover all train cells and decisions")
    return {
        "schema": REPORT_SCHEMA,
        "chance_law": "reference",
        "evidence": "real-emulator-rollout-throughput-not-policy-quality",
        "model_sha256": model_sha256,
        "parameters": parameter_count,
        "workers_requested": workers,
        "worker_processes_observed": processes,
        "combats": len(episodes),
        "train_cells": len(cells),
        "by_cell": dict(sorted(cells.items())),
        "victories": victories,
        "player_decisions": player_decisions,
        "total_boundaries": total_boundaries,
        "elapsed_seconds": float(elapsed),
        "combats_per_second": len(episodes) / elapsed,
        "player_decisions_per_second": player_decisions / elapsed,
        "boundaries_per_second": total_boundaries / elapsed,
        "parameter_bytes_per_worker_lower_bound": 4 * parameter_count,
        "actor_inference_device": "cpu",
    }
