"""H8/H7 policy pilot tests; no emulator required for contract tests."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from benchmark_h8_h7_policy_pilot import (  # noqa: E402
    _choose_weighted,
    _validate_pair,
)

from sts2_ai.nature.h5_curriculum import cell_for_source, source_bank


def test_h8_pilot_weighted_source_intent_rational_ticket():
    rnd = random.Random(17)
    options = ["A", "B", "C"]
    counts = {x: 0 for x in options}
    for _ in range(12_000):
        counts[_choose_weighted(rnd, options, [1, 3, 6])] += 1
    assert 1000 < counts["A"] < 1500
    assert 3300 < counts["B"] < 3900
    assert 6800 < counts["C"] < 7600


@pytest.mark.parametrize("bad", [[], [1], [0, 2], [-1, 2]])
def test_h8_pilot_invalid_source_intent_support_rejected(bad):
    with pytest.raises(ValueError):
        _choose_weighted(random.Random(1), ["A", "B"], bad)


def test_h8_pilot_uses_all_16_h7_source_cells():
    seeds = source_bank("train", 128, study_seed=73127)
    assert len(seeds) == 128
    assert len(set(seeds)) == len(seeds)
    assert len({cell_for_source(seed).key for seed in seeds}) == 16


def test_h8_pilot_rejects_same_choices_but_different_mechanics():
    row = {
        "cell": "mawler-a10::silent-defend-heavy-v1",
        "status": "terminal", "boundaries": 3,
        "player_decisions": 1, "nature_draws": 1, "intents": 1,
        "trace_digest": "same", "journal_digest": "same",
        "mechanics_sha256": "aaa",
        "outcome": "victory", "exit_hp": 66, "terminal_enemy_hp": None,
        "score": 1.9, "player_steps": [], "chance_steps": [],
        "decision_sequence": [], "reference_law_steps": [],
        "reference_event_digest": "identical",
        "captured_player_decisions": 0, "behavior_digest": None,
    }
    left = {"source_digest": "x", "trajectories": {"a": row}}
    _validate_pair(left, left)
    right = {
        "source_digest": "x",
        "trajectories": {"a": {**row, "mechanics_sha256": "bbb"}},
    }
    with pytest.raises(RuntimeError, match="mechanics_sha256"):
        _validate_pair(left, right)
