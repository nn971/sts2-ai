"""H14 macro-boundary temporal-difference returns and non-leaking episode cuts."""
from __future__ import annotations

import pytest

from sts2_ai.training.h14_macro_gae import macro_gae


def test_losing_runs_have_distinct_advantages_from_actual_floor_progress() -> None:
    early = macro_gae([0, 0], [0.0, 0.0], final_floor=1,
                      act1_cleared=False, gamma=1.0, gae_lambda=1.0)
    late = macro_gae([0, 4], [0.0, 0.0], final_floor=10,
                     act1_cleared=False, gamma=1.0, gae_lambda=1.0)
    assert early[0].value_target == pytest.approx(-0.98)
    assert late[0].value_target == pytest.approx(-0.80)
    assert late[1].reward == pytest.approx(6 * 0.02 - 1.0)
    assert all(row.advantage == pytest.approx(row.value_target)
               for row in early + late)


def test_macro_gae_uses_successor_macro_value_and_resets_at_terminal() -> None:
    targets = macro_gae(
        [0, 3, 3], [0.2, 0.4, -0.5], final_floor=5,
        act1_cleared=False, gamma=0.9, gae_lambda=0.5,
    )
    # Immediate floor credit goes to the macro move that reached floor 3.
    assert targets[0].reward == pytest.approx(0.06)
    assert targets[1].reward == 0.0
    assert targets[2].reward == pytest.approx(-0.96)
    last_delta = -0.96 - (-0.5)  # no terminal bootstrap
    mid_delta = 0 + 0.9 * (-0.5) - 0.4
    first_delta = 0.06 + 0.9 * 0.4 - 0.2
    assert targets[2].advantage == pytest.approx(last_delta)
    assert targets[1].advantage == pytest.approx(mid_delta + 0.9 * 0.5 * last_delta)
    assert targets[0].advantage == pytest.approx(
        first_delta + 0.9 * 0.5 * targets[1].advantage,
    )
    assert targets[1].value_target == pytest.approx(targets[1].advantage + 0.4)


def test_act_one_clear_terminal_reward_has_no_act_two_bootstrap() -> None:
    targets = macro_gae(
        [12, 16], [0.1, 0.3], final_floor=16,
        act1_cleared=True, gamma=1.0, gae_lambda=1.0,
    )
    assert targets[-1].reward == 1.0
    assert targets[0].value_target == pytest.approx(1.08)


@pytest.mark.parametrize("floors,final", [
    ([0, 6, 3], 7), ([1], 0), ([0, 17], 17),
])
def test_malformed_or_backward_floor_progress_fails(floors, final) -> None:
    with pytest.raises(ValueError, match="monotone"):
        macro_gae(floors, [0.0] * len(floors),
                  final_floor=final, act1_cleared=False)


def test_malformed_behavior_values_or_unsupported_success_boundary_fail() -> None:
    with pytest.raises(ValueError, match="finite"):
        macro_gae([0], [float("nan")], final_floor=1, act1_cleared=False)
    with pytest.raises(ValueError, match="complete Act-1"):
        macro_gae([0], [0.0], final_floor=15, act1_cleared=True)
