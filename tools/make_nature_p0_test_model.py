#!/usr/bin/env python3
"""Emit a deliberately UNTRAINED portable phase-split policy for CI protocol tests.

Do not use this test artifact as a trained checkpoint or player strength
comparison. Any real experiment must supply a frozen trained export.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sts2_ai.models.neural import NEURAL_FORMAT, TACTICAL_FORMAT, NeuralPolicyValueModel
from sts2_ai.models.phase_split import PhaseSplitNeuralModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def zero_head(format_id: str) -> NeuralPolicyValueModel:
        return NeuralPolicyValueModel.from_dict({
            "format": format_id,
            "model_id": "test-only-zero-head",
            "dimension": 32, "hidden": 8,
            "state_weight": [[0.0] * 32 for _ in range(8)],
            "state_bias": [0.0] * 8,
            "action_weight": [[0.0] * 32 for _ in range(8)],
            "action_bias": [0.0] * 8,
            "policy_weight": [0.0] * 8,
            "policy_bias": 0.0,
            "value_weight": [0.0] * 8,
            "value_bias": 0.0,
        })

    PhaseSplitNeuralModel(
        strategy=zero_head(NEURAL_FORMAT),
        combat=zero_head(TACTICAL_FORMAT),
        model_id="test-only-untrained-zero-policy",
    ).save(args.output)
    print(f"[nature-p0-test-model] untrained_only=True file={args.output}", flush=True)


if __name__ == "__main__":
    main()
