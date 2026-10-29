"""Regression: H15 imports must not transitively require a completed search package."""
from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "from sts2_ai.search.combat_puct import CombatPuct; "
        "from sts2_ai.search import UctMcts, StochasticPuct",
        "from sts2_ai.training.bounded_targets import OnlineBoundedMoments; "
        "from sts2_ai.search.combat_puct import CombatPuct",
        "from sts2_ai.agents import OracleMctsAgent; "
        "from sts2_ai.search import UctMcts",
    ],
)
def test_fresh_process_import_order(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
