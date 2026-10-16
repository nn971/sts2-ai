#!/usr/bin/env python3
"""Generate a TEST-ONLY v15 pooled model for real emulator PPO contract CI.

Untrained legacy heads and nonzero tiny residual gates force the portable
public-object encoder to execute in external workers, preventing the
zero-gate fast path from hiding a fair-observation compatibility bug.
This model is NOT a trained player checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sts2_ai.models.object_embeddings import ObjectVocabulary
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.pooled_objects import PooledObjectTorch
from sts2_ai.training.pooled_ppo_v15 import initialize_combat_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from sts2_ai.models.neural import (
        NEURAL_FORMAT,
        TACTICAL_FORMAT,
        NeuralPolicyValueModel,
    )

    def head(fmt: str) -> NeuralPolicyValueModel:
        return NeuralPolicyValueModel.from_dict({
            "format": fmt, "model_id": "test-only-zero-legacy",
            "dimension": 32, "hidden": 8,
            "state_weight": [[0.] * 32 for _ in range(8)],
            "state_bias": [0.] * 8,
            "action_weight": [[0.] * 32 for _ in range(8)],
            "action_bias": [0.] * 8,
            "policy_weight": [0.] * 8,
            "policy_bias": 0.,
            "value_weight": [0.] * 8,
            "value_bias": 0.,
        })

    base = PhaseSplitNeuralModel(
        head(NEURAL_FORMAT), head(TACTICAL_FORMAT),
        "test-only-untrained-v15-base",
        combat_value_objective="hp_preservation",
    )
    import torch

    torch.manual_seed(29)
    vocabulary = ObjectVocabulary.from_object_sets(())
    trainable = PooledObjectTorch(vocabulary, width=16)
    # Actively exercise the portable object encoder in every combat; its
    # zero-gate short circuit would otherwise bypass P0 public projection.
    with torch.no_grad():
        trainable.module.policy_gate.fill_(0.025)
        trainable.module.value_gate.fill_(0.015)
    exported = PhaseSplitNeuralModel(
        base.strategy, trainable.export(base.combat),
        "test-only-untrained-pooled-v15",
        combat_value_objective="hp_preservation",
    )
    # Verify the warm-start / exported weight shape contract used in training.
    verify = PooledObjectTorch(vocabulary, width=16)
    initialize_combat_head(verify, base, exported)
    exported.save(args.output)
    print(
        f"[nature-ppo-test-only] untrained=True model={args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
