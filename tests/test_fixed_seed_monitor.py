"""Periodic monitoring is fixed-seed, public-only, and observer-only."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_neural_selfplay import ToyFullRunBackend
from test_neural_training_temperature import zero_model

from sts2_ai.evaluation import RunSummary
from sts2_ai.evaluation.fixed_seed_monitor import FixedSeedMonitor
from sts2_ai.training.selfplay import TrainingRound, train_selfplay


def _round(number: int) -> TrainingRound:
    return TrainingRound(
        round_index=number - 1, played=4, completed=4, censored=0,
        wins=0, update_steps=1, mean_loss=0.1, auxiliary_coefficient=0.4,
        decision_samples=8, mean_return=0.1, mean_progress=0.25,
    )


def test_fixed_seed_monitor_reuses_exact_seed_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = __import__("sts2_ai.evaluation.fixed_seed_monitor", fromlist=["play_run"])
    called: list[str] = []

    def fake_play_run(backend: object, agent: object, *, seed: str,
                      **kwargs: object) -> RunSummary:
        called.append(seed)
        return RunSummary(
            seed=seed, outcome="defeat", terminal_act=1, terminal_floor=6,
            decisions=5, emulator_transitions=5, wall_seconds=0.1,
            agent_compute_seconds=0.01, frontier_progress=6.0,
            frontier_enemy_hp=0, hp_trajectory=(70, 0), act1_cleared=False,
        )

    monkeypatch.setattr(imported, "play_run", fake_play_run)
    messages: list[str] = []
    path = tmp_path / "progress.monitor.jsonl"
    monitor = FixedSeedMonitor(
        ToyFullRunBackend(), environment="native-overgrowth", max_decisions=10,
        every=10, seeds=3, path=path, resume=False, progress=messages.append,
    )
    model = zero_model()
    monitor(_round(1), model)
    monitor(_round(2), model)  # Skipped.
    monitor(_round(10), model)
    assert called == [f"selfplay-monitor-{i}" for i in range(3)] * 2
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["training_round"] for r in saved] == [1, 10]
    assert all(r["greedy_completed_only"]["completed"] == 3 for r in saved)
    assert all(r["act1_clears"] == 0 for r in saved)
    assert all(len(r["per_seed"]) == 3 for r in saved)
    assert any("round 10 greedy" in item for item in messages)

    resumed = FixedSeedMonitor(
        ToyFullRunBackend(), environment="native-overgrowth", max_decisions=10,
        every=10, seeds=3, path=path, resume=True, progress=messages.append,
    )
    assert resumed.records == saved
    with pytest.raises(ValueError, match="different fixed seeds"):
        FixedSeedMonitor(
            ToyFullRunBackend(), environment="native-overgrowth",
            max_decisions=10, every=10, seeds=4, path=path,
            resume=True, progress=messages.append,
        )


def test_snapshot_monitor_observes_updated_model_without_affecting_training() -> None:
    pytest.importorskip("torch")
    settings = dict(rounds=2, episodes_per_round=4, dimension=16, hidden=4, seed=7)
    expected = train_selfplay(ToyFullRunBackend(), **settings)
    snapshots: list[tuple[int, dict[str, object]]] = []
    observed = train_selfplay(
        ToyFullRunBackend(), **settings,
        on_model_snapshot=lambda row, model: snapshots.append(
            (row.round_index, model.to_dict())
        ),
    )
    assert observed.rounds == expected.rounds
    assert observed.model.to_dict() == expected.model.to_dict()
    assert [i for i, _ in snapshots] == [0, 1]
    assert snapshots[1][1]["state_weight"] == expected.model.state_weight
