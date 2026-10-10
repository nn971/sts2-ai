"""Differentiable v8 enemy-set encoder and exact target-pointer policy.

The pure Python inference implementation in models/neural.py implements the
identical operations, with no Torch dependency at deployment.
"""
from __future__ import annotations

from typing import Any, Sequence

from sts2_ai.emulator import LegalAction
from sts2_ai.models.enemy_instances import (
    action_target_instance,
    enemy_instance_vectors,
    instance_action_features,
    instance_global_features,
)
from sts2_ai.training.neural import _dense
from sts2_ai.training.enemy_attention import contextualize_torch


def instance_forward(
    state: dict[str, Any], actions: Sequence[LegalAction],
    params: dict[str, Any], dimension: int, torch: Any,
) -> tuple[Any, Any]:
    enemy_features = enemy_instance_vectors(state, dimension)
    ids = list(enemy_features)
    linear = torch.nn.functional.linear
    relu = torch.nn.functional.relu
    if ids:
        enemy_input = torch.tensor([
            _dense(enemy_features[eid], dimension) for eid in ids
        ], dtype=torch.float32)
        enemy_hidden = relu(linear(
            enemy_input, params["enemy_weight"], params["enemy_bias"],
        ))
        enemy_hidden = contextualize_torch(
            enemy_hidden.unsqueeze(0),
            torch.ones((1, len(ids)), dtype=torch.bool), params, torch,
        ).squeeze(0)
        pooled = enemy_hidden.mean(dim=0)
        by_id = {eid: i for i, eid in enumerate(ids)}
    else:
        enemy_hidden = torch.zeros(
            (0, params["enemy_bias"].shape[0]), dtype=torch.float32,
        )
        pooled = torch.zeros_like(params["enemy_bias"])
        by_id = {}

    state_x = torch.tensor(
        _dense(instance_global_features(state, dimension), dimension),
        dtype=torch.float32,
    )
    hidden = relu(
        linear(state_x, params["state_weight"], params["state_bias"])
        + pooled * params["enemy_context_weight"]
    )
    value = torch.tanh(
        torch.dot(hidden, params["value_weight"]) + params["value_bias"]
    )
    if not actions:
        return value, torch.empty((0,), dtype=torch.float32)
    action_x = torch.tensor([
        _dense(instance_action_features(
            state, action.kind, action.payload_json, dimension,
        ), dimension)
        for action in actions
    ], dtype=torch.float32)
    targets = [
        action_target_instance(action.payload_json, enemy_features)
        for action in actions
    ]
    zero_target = torch.zeros_like(hidden)
    target_x = torch.stack([
        enemy_hidden[by_id[target]] if target is not None else zero_target
        for target in targets
    ])
    action_hidden = relu(
        hidden.unsqueeze(0)
        + linear(action_x, params["action_weight"], params["action_bias"])
        + target_x * params["enemy_target_weight"]
    )
    logits = torch.mv(action_hidden, params["policy_weight"]) + params["policy_bias"]
    return value, logits


def add_instance_parameters(
    params: dict[str, Any], dimension: int, hidden: int, torch: Any,
) -> None:
    """Create new learnable tensors; never retrofit v6/v7 optimizer state."""
    import math

    scale = math.sqrt(2.0 / (dimension + hidden))
    params.update({
        "enemy_weight": (torch.randn(hidden, dimension) * scale).requires_grad_(),
        "enemy_bias": torch.zeros(hidden, requires_grad=True),
        "enemy_context_weight": torch.full(
            (hidden,), 0.25, requires_grad=True,
        ),
        "enemy_target_weight": torch.full(
            (hidden,), 0.5, requires_grad=True,
        ),
    })
