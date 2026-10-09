"""Fixed-seed public-only greedy evaluation of live self-play snapshots.

The monitor NEVER supplies teacher targets or alters the trainer's optimizer.
Its seeds are distinct from training and final held-out evaluation seeds.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sts2_ai.agents import NeuralGreedyAgent
from sts2_ai.emulator import FAIR_POLICY_ID, EmulatorBackend, InformationPolicy
from sts2_ai.evaluation import RunSummary, play_run
from sts2_ai.evaluation.boss_progress import summarize_boss_runs
from sts2_ai.evaluation.selfplay_metrics import summarize_completed_runs
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.training.selfplay import TrainingRound

SEED_PREFIX = "selfplay-monitor"


class FixedSeedMonitor:
    def __init__(
        self, backend: EmulatorBackend, *, environment: str,
        max_decisions: int, every: int, seeds: int, path: Path,
        resume: bool, progress: Callable[[str], None],
    ) -> None:
        self.backend = backend
        self.environment = environment
        self.max_decisions = max_decisions
        self.every = every
        self.seeds = seeds
        self.path = path
        self.progress = progress
        self.records: list[dict[str, Any]] = []
        if every <= 0 or seeds <= 0:
            raise ValueError("Fixed-seed monitoring requires positive cadence and seeds")
        if resume and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if (
                    record.get("seed_count") != seeds
                    or record.get("environment") != environment
                    or record.get("seed_prefix") != SEED_PREFIX
                ):
                    raise ValueError("Existing monitor file uses different fixed seeds")
                self.records.append(record)

    def __call__(self, row: TrainingRound, model: NeuralPolicyValueModel) -> None:
        if row.round_index != 0 and (row.round_index + 1) % self.every != 0:
            return
        number = row.round_index + 1
        self.progress(
            f"[monitor] round {number}: evaluating greedy "
            f"on {self.seeds} fixed unseen seeds"
        )
        agent = NeuralGreedyAgent(model, content_hash=model.model_id)
        runs: list[RunSummary] = []
        started = time.perf_counter()
        for index in range(self.seeds):
            runs.append(play_run(
                self.backend, agent,
                seed=f"{SEED_PREFIX}-{index}",
                policy=InformationPolicy(FAIR_POLICY_ID),
                max_decisions=self.max_decisions,
                environment=self.environment,
            ))
            if (index + 1) % min(self.seeds, 16) == 0:
                self.progress(
                    f"[monitor] round {number}: {index + 1}/{self.seeds} "
                    f"elapsed={time.perf_counter() - started:.1f}s"
                )

        summary = summarize_completed_runs(runs)
        boss_summary = summarize_boss_runs(runs)
        valid = [run for run in runs if not run.censored]
        floor16 = sum(
            (run.terminal_act or 1) > 1
            or ((run.terminal_act or 1) == 1 and (run.terminal_floor or 0) >= 16)
            for run in valid
        )
        cleared = sum(run.act1_cleared is True for run in valid)
        record: dict[str, Any] = {
            "schema": "sts2-fixed-seed-selfplay-monitor-v1",
            "training_round": number,
            "model_id": model.model_id,
            "environment": self.environment,
            "seed_prefix": SEED_PREFIX,
            "seed_count": self.seeds,
            "sampling_temperature": row.sampling_temperature,
            "greedy_completed_only": summary,
            "boss_health_completed_only": boss_summary,
            "act1_floor16_reached": floor16,
            "act1_clears": cleared,
            "per_seed": [
                {
                    "seed": run.seed,
                    "outcome": run.outcome,
                    "censored": run.censored,
                    "frontier_progress": run.frontier_progress,
                    "act": run.terminal_act,
                    "floor": run.terminal_floor,
                    "act1_cleared": run.act1_cleared,
                    "boss_progress": (
                        asdict(run.boss_progress) if run.boss_progress is not None
                        else None
                    ),
                }
                for run in runs
            ],
        }

        self.records = [
            item for item in self.records
            if item["training_round"] != number
        ] + [record]
        self.records.sort(key=lambda item: item["training_round"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        staged = self.path.with_name(self.path.name + ".pending")
        staged.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n"
                    for item in self.records),
            encoding="utf-8",
        )
        staged.replace(self.path)
        self.progress(
            f"[monitor] round {number} greedy | "
            f"mean_frontier={summary['mean_frontier_progress_completed_only']} "
            f"completed={summary['completed']}/{self.seeds} "
            f"wins={summary['wins']} floor16={floor16} act1_clears={cleared} "
            f"boss_entries={boss_summary['boss_entries']} "
            f"boss_defeats={boss_summary['boss_defeats']} "
            f"boss_damage={boss_summary['mean_boss_damage_fraction_on_defeat']} "
            f"near_kills={boss_summary['boss_near_kills_on_defeat_80pct']} | "
            f"saved={self.path}"
        )
