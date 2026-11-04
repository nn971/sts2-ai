"""H21 scalar projection must equal the inherited 16-field numeric encoder."""
from __future__ import annotations

import math

import pytest

from sts2_ai.models.shared_game_objects import (
    NUMERIC_FIELDS,
    _numbers,
    _single_numeric,
)


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize(
    "value",
    [None, False, True, 0, 1, -1, 2.5, -3.125, 10_000,
     "1", "", float("inf"), float("-inf"), float("nan")],
)
def test_h21_fast_single_numeric_is_exact(field, value):
    assert _single_numeric(field, value) == _numbers({field: value})


def test_present_zero_and_missing_value_remain_distinct():
    values, masks = _single_numeric("stacks", 0)
    missing, absent = _single_numeric("stacks", None)
    assert values == missing
    assert masks != absent
    assert sum(masks) == 1
    assert sum(absent) == 0
    assert math.isfinite(sum(values))


def test_fast_projection_rejects_unknown_numeric_field():
    with pytest.raises(KeyError):
        _single_numeric("not_a_public_numeric_field", 3)
