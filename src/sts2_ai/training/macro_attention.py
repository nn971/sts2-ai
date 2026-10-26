"""Differentiable four-head H6-C macro-object attention.

Parameter names deliberately match the existing v13 instance-attention
kernel so the batched PPO forward path, masks and portable inference can
reuse the same algebra without touching the frozen combat model.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sts2_ai.emulator import LegalAction
from sts2_ai.models.macro_objects import (
    macro_action_target,
    macro_deep_action_target,
    macro_deep_object_vectors,
    macro_attention_action_features,
    macro_global_features,
    macro_object_vectors,
    macro_semantic_action_features,
)
from sts2_ai.training.enemy_attention import (
    contextualize_deep_torch, contextualize_torch,
)
from sts2_ai.training.neural import _dense


def macro_attention_forward(
    state: dict[str, Any], actions: Sequence[LegalAction],
    params: dict[str, Any], dimension: int, torch: Any,
    *, semantic_actions: bool = False, deep_actions: bool = False,
) -> tuple[Any, Any]:
    objects = (macro_deep_object_vectors if deep_actions else macro_object_vectors)(
        state, actions, dimension,
    )
    ids = list(objects)
    linear = torch.nn.functional.linear
    relu = torch.nn.functional.relu
    if ids:
        objects_x = torch.tensor(
            [_dense(objects[key], dimension) for key in ids], dtype=torch.float32,
        )
        embeddings = relu(linear(
            objects_x, params["enemy_weight"], params["enemy_bias"],
        ))
        embeddings = contextualize_torch(
            embeddings.unsqueeze(0),
            torch.ones((1, len(ids)), dtype=torch.bool), params, torch,
        ).squeeze(0)
        if deep_actions:
            embeddings = contextualize_deep_torch(
                embeddings.unsqueeze(0),
                torch.ones((1, len(ids)), dtype=torch.bool),
                params, torch,
            ).squeeze(0)
        pooled = embeddings.mean(dim=0)
        index = {key: i for i, key in enumerate(ids)}
    else:
        embeddings = torch.zeros(
            (0, params["enemy_bias"].shape[0]), dtype=torch.float32,
        )
        pooled = torch.zeros_like(params["enemy_bias"])
        index = {}
    state_x = torch.tensor(
        _dense(macro_global_features(state, dimension), dimension), dtype=torch.float32,
    )
    hidden = relu(
        linear(state_x, params["state_weight"], params["state_bias"])
        + pooled * params["enemy_context_weight"]
    )
    value = torch.tanh(torch.dot(
        hidden, params["value_weight"],
    ) + params["value_bias"])
    if not actions:
        return value, torch.empty((0,), dtype=torch.float32)
    actions_x = torch.tensor([
        _dense((macro_semantic_action_features
                if semantic_actions or deep_actions else macro_attention_action_features)(
            state, action.kind, action.payload_json, dimension,
        ), dimension) for action in actions
    ], dtype=torch.float32)
    zero = torch.zeros_like(hidden)
    targets = [
        (macro_deep_action_target if deep_actions else macro_action_target)(
            state, action, objects,
        ) for action in actions
    ]
    target_x = torch.stack([
        embeddings[index[target]] if target is not None else zero
        for target in targets
    ])
    action_hidden = relu(
        hidden.unsqueeze(0)
        + linear(actions_x, params["action_weight"], params["action_bias"])
        + target_x * params["enemy_target_weight"]
    )
    logits = torch.mv(action_hidden, params["policy_weight"]) + params["policy_bias"]
    return value, logits
