"""H6-A2: the macro objective must retain defeat progress after 16 clears."""
from __future__ import annotations

import pytest

from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH
from sts2_ai.training.selfplay import (
    Episode,
    _bounded_return,
    _curriculum_coefficient,
)


def _completed_run(outcome: str, floor: int) -> Episode:
    return Episode(
        seed=f"h6-a2-{outcome}-{floor}",
        decisions=(),
        outcome=outcome,
        act=1,
        floor=floor,
        hp_ratio=0.0,
        completed=True,
        environment=NATIVE_OVERGROWTH,
    )


def test_h6_a2_retains_defeat_progress_after_original_16_win_cutoff() -> None:
    # H6-A lost all floor/HP distinctions after the 16th cumulative clear.
    assert _curriculum_coefficient(0.4, 16, 16) == 0.0

    # H6-A2 keeps the same shaping formula and only stretches annealing.
    coefficient = _curriculum_coefficient(0.4, 16, 256)
    assert coefficient == pytest.approx(0.375)
    assert 0 < _bounded_return(_completed_run("defeat", 4), coefficient)
    assert (
        _bounded_return(_completed_run("defeat", 15), coefficient)
        > _bounded_return(_completed_run("defeat", 4), coefficient)
    )
    assert _bounded_return(_completed_run("victory", 16), coefficient) == 1.0


@pytest.mark.parametrize("cumulative_wins", [32, 64, 128, 192, 255])
def test_h6_a2_sparse_clear_rate_keeps_progress_reward(
    cumulative_wins: int,
) -> None:
    coefficient = _curriculum_coefficient(0.4, cumulative_wins, 256)
    assert 0 < coefficient < 0.4
    defeat_return = _bounded_return(_completed_run("defeat", 12), coefficient)
    assert 0 < defeat_return < 1
    assert _bounded_return(_completed_run("victory", 16), coefficient) == 1


def test_h6_a2_anneals_to_zero_only_after_256_cumulative_clears() -> None:
    assert _curriculum_coefficient(0.4, 256, 256) == 0
    assert _curriculum_coefficient(0.4, 512, 256) == 0
