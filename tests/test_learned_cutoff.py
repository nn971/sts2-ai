from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sts2_ai.agents import HeuristicAgent
from sts2_ai.emulator import InformationPolicy, Observation
from sts2_ai.models import HashedLinearPolicyValueModel
from sts2_ai.search import LearnedCutoffValue, UctMcts, sts2_value
from sts2_ai.testing.mock_backend import MockLinearBackend


def _observation(payload: dict[str, object]) -> Observation:
    return Observation(
        policy_id="fair-test",
        payload_json=json.dumps(payload, sort_keys=True),
        observation_hash="test-observation",
    )


def test_learned_cutoff_preserves_actual_terminal_outcomes(tmp_path: Path) -> None:
    model = HashedLinearPolicyValueModel.zeros(128)
    model.value_weights = [10.0] * model.dimension
    path = tmp_path / "model.json"
    model.save(path)
    cutoff = LearnedCutoffValue.load(path)

    assert cutoff(_observation({"terminal_outcome": "victory", "hp": 1})) == 1.0
    assert cutoff(_observation({"terminal_outcome": "defeat", "hp": 70})) == -1.0
    with pytest.raises(ValueError, match="Unsupported terminal"):
        cutoff(_observation({"terminal_outcome": "draw"}))


def test_nonterminal_cutoff_uses_model_value_only(tmp_path: Path) -> None:
    model = HashedLinearPolicyValueModel.zeros(128)
    path = tmp_path / "model.json"
    model.save(path)
    cutoff = LearnedCutoffValue.load(path)

    observation = _observation({
        "act": 1, "floor": 2, "hp": 60, "max_hp": 70
    })
    assert cutoff(observation) == pytest.approx(
        model.evaluate(observation, ()).value
    )
    assert cutoff(observation) == 0.0
    assert sts2_value(observation) != 0.0


def test_weight_file_fingerprints_distinct_searches(tmp_path: Path) -> None:
    model = HashedLinearPolicyValueModel.zeros(128, model_id="reused-name")
    path = tmp_path / "weights.json"
    model.save(path)
    first = LearnedCutoffValue.load(path)
    model.value_weights[1] = 0.5
    model.save(path)
    second = LearnedCutoffValue.load(path)

    assert first.value_id != second.value_id
    backend = MockLinearBackend(terminal_at=10)
    policy = InformationPolicy("fair-test")
    common = {
        "policy": policy,
        "rollout_policy": HeuristicAgent(),
        "rollout_depth": 2,
        "seed": 2,
    }
    baseline = UctMcts(backend, **common)
    left = UctMcts(backend, value_fn=first, **common)
    right = UctMcts(backend, value_fn=second, **common)
    assert baseline.search_version != left.search_version
    assert left.search_version != right.search_version
    assert f"|value={left._value_fn.value_id}" in left.search_version


def test_invalid_model_values_do_not_enter_uct(tmp_path: Path) -> None:
    model = HashedLinearPolicyValueModel.zeros(64)
    model.value_weights[0] = math.nan
    path = tmp_path / "bad.json"
    model.save(path)
    cutoff = LearnedCutoffValue.load(path)
    with pytest.raises(ValueError, match="invalid value"):
        cutoff(_observation({"phase": 3, "hp": 50}))
