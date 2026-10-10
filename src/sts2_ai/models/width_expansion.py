"""Function-preserving hidden-width expansion of portable v8 phase-split models.

Every hidden coordinate is replicated. Incoming weights, biases, and
per-instance interaction coefficients are copied; each replicated output
coefficient is split across its clones. Unequal splits preserve all initial
predictions yet break clone-gradient symmetry once training resumes.

This is an explicit NEW optimizer experiment, not a checkpoint migration.
"""
from __future__ import annotations

import copy
from typing import Any

from sts2_ai.models.neural import (
    NEURAL_FORMAT,
    TACTICAL_INSTANCES_FORMAT,
    NeuralPolicyValueModel,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel


def _split_coefficients(source: list[float], copies: int, *, reverse: bool) -> list[float]:
    values: list[float] = []
    for index, value in enumerate(source):
        # Unequal, deterministic allocations; the sum is exactly 1 in
        # real arithmetic, and the two clones have unequal policy gradients.
        if copies == 2:
            weights = [0.375, 0.625] if index % 2 == 0 else [0.625, 0.375]
            if reverse:
                weights.reverse()
        else:
            raw = list(range(1, copies + 1))
            if (index % 2 == 1) != reverse:
                raw.reverse()
            total = sum(raw)
            weights = [w / total for w in raw]
        values.extend(value * weight for weight in weights)
    return values


def widen_neural(model: NeuralPolicyValueModel, target_hidden: int) -> NeuralPolicyValueModel:
    old = model.hidden
    if target_hidden <= old or target_hidden % old:
        raise ValueError("Widening requires an integer multiple of the original width")
    copies = target_hidden // old
    raw: dict[str, Any] = copy.deepcopy(model.to_dict())
    raw["hidden"] = target_hidden
    raw["model_id"] = f"{model.model_id}-width{target_hidden}"
    for field in ("state_weight", "action_weight", "enemy_weight"):
        if raw.get(field) is not None:
            raw[field] = [
                copy.deepcopy(row)
                for row in raw[field] for _ in range(copies)
            ]
    for field in (
        "state_bias", "action_bias", "enemy_bias",
        "enemy_context_weight", "enemy_target_weight",
    ):
        if raw.get(field) is not None:
            raw[field] = [
                value for value in raw[field] for _ in range(copies)
            ]
    raw["policy_weight"] = _split_coefficients(
        model.policy_weight, copies, reverse=False,
    )
    raw["value_weight"] = _split_coefficients(
        model.value_weight, copies, reverse=True,
    )
    return NeuralPolicyValueModel.from_dict(raw)


def widen_phase_split(
    model: PhaseSplitNeuralModel, target_hidden: int,
) -> PhaseSplitNeuralModel:
    if (
        model.strategy.format_id != NEURAL_FORMAT
        or model.combat.format_id != TACTICAL_INSTANCES_FORMAT
    ):
        raise ValueError("Capacity widening expects semantic strategy and v8 instance combat")
    return PhaseSplitNeuralModel(
        strategy=widen_neural(model.strategy, target_hidden),
        combat=widen_neural(model.combat, target_hidden),
        model_id=f"{model.model_id}-width{target_hidden}",
        combat_value_objective=model.combat_value_objective,
    )
