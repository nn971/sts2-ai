"""Deterministic local multi-seed Nature v15 study launch protocol."""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_nature_p0_replicas import make_commands, parse_seeds


def test_multiseed_launcher_is_symmetric_and_uses_independent_holdout_rng():
    seed_values = (29, 101, 211, 307, 401)
    jobs = make_commands(
        seeds=seed_values, rounds=100, episodes_per_round=128, workers=15,
        initial_model=Path("/initial.json"), train_corpus=Path("/train.jsonl"),
        evaluation_corpus=Path("/evaluation.jsonl"),
        emulator_dll=Path("/engine.dll"),
        emulator_bindings=Path("/bindings"), emulator_revision="csharp-sha",
        root=Path("/study"), temperature=0.85, learning_rate=0.0003,
        epochs=3, batch_size=32, sample_limit=2048,
    )
    assert len(jobs) == 15
    for seed, n, o, evaluation in zip(
        seed_values, jobs[::3], jobs[1::3], jobs[2::3], strict=True
    ):
        assert n["training_seed"] == o["training_seed"] == seed
        assert n["phase"] == o["phase"] == "train"
        assert n["arm"] == "nature"
        assert o["arm"] == "ordinary"
        assert "--nature-update" in n["argv"]
        assert "--nature-update" not in o["argv"]
        for key in ("--rounds", "--episodes-per-round", "--workers", "--seed",
                    "--initial-model", "--source-fixtures", "--epochs",
                    "--batch-size", "--sample-limit", "--temperature"):
            assert key in n["argv"] and key in o["argv"]
            assert n["argv"][n["argv"].index(key) + 1] == (
                o["argv"][o["argv"].index(key) + 1]
            )
        assert evaluation["phase"] == "holdout"
        assert evaluation["training_seed"] == seed
        assert evaluation["argv"][evaluation["argv"].index("--actor-seed") + 1] == (
            str(900000 + seed)
        )
    assert parse_seeds("29, 101,211") == (29, 101, 211)
    assert jobs == make_commands(
        seeds=seed_values, rounds=100, episodes_per_round=128, workers=15,
        initial_model=Path("/initial.json"), train_corpus=Path("/train.jsonl"),
        evaluation_corpus=Path("/evaluation.jsonl"),
        emulator_dll=Path("/engine.dll"),
        emulator_bindings=Path("/bindings"), emulator_revision="csharp-sha",
        root=Path("/study"), temperature=0.85, learning_rate=0.0003,
        epochs=3, batch_size=32, sample_limit=2048,
    )


@pytest.mark.parametrize("bad", ["", "29,29", "29,nan"])
def test_multiseed_planner_rejects_invalid_seed_sets(bad):
    with pytest.raises(ValueError):
        parse_seeds(bad)


def test_training_plan_rejects_nonpositive_budgets():
    with pytest.raises(ValueError, match="Positive training"):
        make_commands(
            seeds=(29,), rounds=0, episodes_per_round=128, workers=15,
            initial_model=Path("/initial.json"), train_corpus=Path("/train.jsonl"),
            evaluation_corpus=Path("/evaluation.jsonl"),
            emulator_dll=Path("/engine.dll"),
            emulator_bindings=Path("/bindings"), emulator_revision="csharp-sha",
            root=Path("/study"), temperature=0.85, learning_rate=0.0003,
            epochs=3, batch_size=32, sample_limit=2048,
        )
