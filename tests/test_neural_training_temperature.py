"""On-policy temperature scheduling, gradient correctness and warm-start safety."""
from __future__ import annotations

import random
from pathlib import Path

import pytest
from test_neural_selfplay import A, B, ToyFullRunBackend, zero_model

from sts2_ai.emulator import InformationPolicy
from sts2_ai.training.selfplay import (
    sample_public_action,
    temperature_for_round,
    train_selfplay,
)


def test_geometric_schedule_and_strict_validation() -> None:
    assert temperature_for_round(0, start=0.2, end=0.05, decay_rounds=10) == pytest.approx(0.2)
    assert temperature_for_round(5, start=0.2, end=0.05, decay_rounds=10) == pytest.approx(0.1)
    assert temperature_for_round(10, start=0.2, end=0.05, decay_rounds=10) == pytest.approx(0.05)
    assert temperature_for_round(50, start=0.2, end=0.05, decay_rounds=10) == pytest.approx(0.05)
    for start, end, duration in ((0, 1, 1), (1, -0.1, 1), (1, 1, 0), (float("nan"), 1, 1)):
        with pytest.raises(ValueError, match="temperature|Temperature"):
            temperature_for_round(0, start=start, end=end, decay_rounds=duration)
    with pytest.raises(ValueError, match="nonnegative"):
        temperature_for_round(-1, start=0.2, end=0.1, decay_rounds=1)


def test_temperature_one_preserves_original_categorical_draw() -> None:
    model = zero_model()
    backend = ToyFullRunBackend()
    handle = backend.reset("peek")
    obs = backend.observe(handle, InformationPolicy("prototype-fair-v0"))
    backend.release_many((handle,))
    for seed in (1, 7, 18):
        a = sample_public_action(model, obs, (A, B), rng=random.Random(seed))
        b = sample_public_action(
            model, obs, (A, B), rng=random.Random(seed), temperature=1.0
        )
        assert a == b
    with pytest.raises(ValueError, match="temperature"):
        sample_public_action(model, obs, (A, B), rng=random.Random(1), temperature=0.0)


def test_warm_started_on_policy_training_has_fresh_optimizer_and_annealed_temp(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    source = tmp_path / "source.json"
    zero_model().save(source)
    settings = dict(
        dimension=32, hidden=8, episodes_per_round=16,
        learning_rate=0.01, auxiliary_weight=0.0,
        entropy_weight=0.0, seed=13,
        sampling_temperature_start=0.2,
        sampling_temperature_end=0.05,
        temperature_decay_rounds=2,
        initialize_from_model=source,
        max_decisions=4,
    )
    checkpoint = tmp_path / "temperature.pt"
    original = train_selfplay(
        ToyFullRunBackend(), rounds=3, checkpoint_path=checkpoint, **settings
    )
    assert original.initial_model.to_dict() == zero_model().to_dict()
    assert [r.sampling_temperature for r in original.rounds] == pytest.approx(
        [0.2, 0.1, 0.05]
    )
    assert sum(r.update_steps for r in original.rounds) == 3
    assert original.model.model_id.startswith("selfplay-v4-")
    resumed = train_selfplay(
        ToyFullRunBackend(), rounds=3, checkpoint_path=checkpoint,
        resume=True, **settings
    )
    assert resumed.rounds == original.rounds
    assert resumed.model.to_dict() == original.model.to_dict()
    with pytest.raises(ValueError, match="configuration mismatch"):
        train_selfplay(
            ToyFullRunBackend(), rounds=4, checkpoint_path=checkpoint,
            resume=True, **(settings | {"sampling_temperature_end": 0.06})
        )
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="configuration mismatch"):
        train_selfplay(
            ToyFullRunBackend(), rounds=4, checkpoint_path=checkpoint,
            resume=True, **settings
        )


def test_wrong_warm_start_dimensions_fail_before_any_episode(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    model = tmp_path / "source.json"
    zero_model().save(model)
    backend = ToyFullRunBackend()
    with pytest.raises(ValueError, match="dimensions"):
        train_selfplay(
            backend, rounds=1, episodes_per_round=2,
            dimension=16, hidden=8, initialize_from_model=model,
        )
    assert not backend._states
