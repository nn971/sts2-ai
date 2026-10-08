"""Training progress must stream by default without changing on-policy learning."""
from __future__ import annotations

from pathlib import Path

import pytest
from test_neural_selfplay import ToyFullRunBackend

from sts2_ai.training.selfplay import TrainingRound, train_selfplay
from tools.train_selfplay import _TrainingProgress


def test_live_round_progress_is_flushed_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    progress = _TrainingProgress(3, 4)
    progress.start(0, 3, 0.12)
    progress.complete(TrainingRound(
        round_index=0,
        played=4,
        completed=3,
        censored=1,
        wins=1,
        update_steps=1,
        mean_loss=0.23,
        auxiliary_coefficient=0.4,
        decision_samples=24,
        mean_return=0.1,
        mean_progress=0.25,
        sampling_temperature=0.12,
    ))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[train] round 1/3 starting" in captured.err
    assert "temperature=0.12" in captured.err
    assert "[train] round 1/3 complete" in captured.err
    assert "completed=3/4 censored=1" in captured.err
    assert "wins=1 new_wins=1" in captured.err
    assert "progress=25.0%" in captured.err
    assert "decisions=24" in captured.err
    assert "eta=" in captured.err


def test_round_observers_do_not_change_training_or_resumability(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    starts: list[tuple[int, int, float]] = []
    completions: list[TrainingRound] = []
    config = dict(rounds=2, episodes_per_round=4, dimension=16, hidden=4, seed=7)
    baseline = train_selfplay(ToyFullRunBackend(), **config)
    checkpoint = tmp_path / "observed.pt"
    with_observer = train_selfplay(
        ToyFullRunBackend(), **config, checkpoint_path=checkpoint,
        on_round_start=lambda r, n, t: starts.append((r, n, t)),
        on_round_complete=completions.append,
    )
    assert starts == [(0, 2, 1.0), (1, 2, 1.0)]
    assert tuple(completions) == with_observer.rounds
    assert with_observer.rounds == baseline.rounds
    assert with_observer.model.to_dict() == baseline.model.to_dict()
    resumed = train_selfplay(
        ToyFullRunBackend(), **config, checkpoint_path=checkpoint, resume=True,
        on_round_start=lambda r, n, t: starts.append((r, n, t)),
    )
    assert resumed.model.to_dict() == baseline.model.to_dict()
    assert len(starts) == 2  # Nothing to report when no new round is due.
