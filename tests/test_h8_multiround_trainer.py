"""Contracts for H8 as a genuine multi-round H7 training backend."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_h8_h7_policy_pilot as pilot  # noqa: E402
from train_h7_reference_combat import collect_h8, summarize  # noqa: E402
from sts2_ai.nature.h5_curriculum import (
    TRAIN_CELLS, cell_for_source, source_bank,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_backend="h8-native", actor_backend="shared",
        emulator_dll=Path("/tmp/h8.dll"),
        emulator_bindings=Path("/tmp/python-bindings"),
        workers=15, max_boundaries=1200,
        actor_device="cpu", actor_max_batch=16,
        actor_delay_ms=3.0, train_terminal_replay_every=2,
    )


def _mock_h8(*, status_override: str | None = None):
    calls = []

    def run(seeds, **kwargs):
        calls.append((seeds, kwargs))
        audit = kwargs["replay_indices"]
        capture = kwargs["capture_player_training"]
        rows = {}
        for index, seed in enumerate(seeds):
            cell = cell_for_source(seed)
            assert cell is not None
            rows[seed] = {
                "status": (status_override if index == 0 and status_override
                           else "terminal"),
                "cell": cell.key,
                "outcome": "victory",
                "exit_hp": 60,
                "terminal_enemy_hp": None,
                "score": 1.8,
                "replay_verified": index in audit,
                "reference_event_digest": "declared-event-law",
                "player_decisions": 1,
                "captured_player_decisions": int(capture),
                "player_steps": [{"phase": "player", "selected_id": "end_turn"}],
                "decision_sequence": [
                    {"phase": "player", "selected_id": "end_turn"},
                ],
                "training_actions": (
                    [{"selected_id": "end_turn"}] if capture else None
                ),
            }
        counts = {"terminal": len(seeds)} if status_override is None else {
            "terminal": len(seeds) - 1, status_override: 1,
        }
        return {
            "episodes": len(seeds), "statuses": counts,
            "trajectories": rows, "wall_seconds": 12.0,
            "workers": kwargs["workers"],
            "player_decisions": len(seeds),
            "all_boundaries": len(seeds),
            "terminal_combats_per_second": (len(seeds) - bool(status_override)) / 12,
            "actor_metrics": {"batches": 8, "requests": len(seeds)},
        }

    return run, calls


def test_h8_multiround_train_keeps_per_cell_replay_schedule(monkeypatch):
    seeds = source_bank("train", 32, study_seed=73127)
    mock, calls = _mock_h8()
    monkeypatch.setattr(pilot, "_cohort", mock)
    batch = collect_h8(
        seeds, spec=object(), args=_args(),
        reference_seed=23387, capture=True,
    )
    assert len(batch["episodes"]) == 32
    assert all(len(x) == 1 for x in batch["player_rollouts"])
    assert batch["terminal_replay_verified"] == len(TRAIN_CELLS)
    assert batch["terminal_replay_not_checked"] == len(TRAIN_CELLS)
    assert batch["process_count"] == 15
    assert batch["actor_metrics"]["server"]["batches"] == 8
    assert batch["terminal_replay_audit_seed_digest"] is not None
    assert summarize(batch["episodes"])["wins"] == 32
    assert calls[0][1]["chance_sampling"] == "event-law"
    assert calls[0][1]["audit_mechanics"] is False
    assert calls[0][1]["capture_player_training"] is True
    assert {ep.scenario_id for ep in batch["episodes"]} <= {
        c.scenario for c in TRAIN_CELLS}
    assert {ep.deck_preset_id for ep in batch["episodes"]} <= {
        c.deck for c in TRAIN_CELLS}
    for cell in TRAIN_CELLS:
        matching = [
            i for i, seed in enumerate(seeds)
            if cell_for_source(seed).key == cell.key
        ]
        assert sum(i in calls[0][1]["replay_indices"] for i in matching) == 1


def test_h8_selection_eval_audits_all_without_ppo_capture(monkeypatch):
    seeds = source_bank("selection", 16, study_seed=73127)
    mock, calls = _mock_h8()
    monkeypatch.setattr(pilot, "_cohort", mock)
    batch = collect_h8(
        seeds, spec=object(), args=_args(),
        reference_seed=700000003, capture=False,
    )
    assert batch["terminal_replay_verified"] == 16
    assert batch["terminal_replay_not_checked"] == 0
    assert all(x == () for x in batch["player_rollouts"])
    assert calls[0][1]["capture_player_training"] is False
    assert calls[0][1]["replay_indices"] == frozenset(range(16))


def test_h8_collector_rejects_censored_trajectories(monkeypatch):
    mock, _ = _mock_h8(status_override="max_boundaries")
    monkeypatch.setattr(pilot, "_cohort", mock)
    with pytest.raises(RuntimeError, match="CENSORED"):
        collect_h8(
            source_bank("train", 16, study_seed=73127),
            spec=object(), args=_args(),
            reference_seed=23387, capture=True,
        )


def test_h8_backend_requires_shared_actor():
    args = _args()
    args.actor_backend = "local"
    with pytest.raises(ValueError, match="shared actor"):
        collect_h8(
            source_bank("train", 16, study_seed=73127),
            spec=object(), args=args,
            reference_seed=23387, capture=True,
        )
