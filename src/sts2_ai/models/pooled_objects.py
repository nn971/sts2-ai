"""v15c: trainable pooled-object residual, with dependency-free portable inference.

The old combat policy is frozen as the base. An independently trained residual
uses the instance-aware v15 embedding, simple mean pooling and direct
source/target lookups. No attention is used: this is the v15d control.

Torch is imported lazily for training only; deployed JSON inference is pure
Python. Training the residual is opt-in, not wired into legacy PPO yet.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.models.protocol import PolicyValueEstimate

from .object_embeddings import ObjectVocabulary, TypedObjectEncoder
from .object_instances import (
    ObjectRef,
    ObjectSet,
    extract_actionable_combat_objects,
    legal_action_object_refs,
)

POOLED_OBJECT_FORMAT = "sts2-neural-policy-value-v15c-pooled-object-residual"


def _mean(vectors: Sequence[list[float]], width: int) -> list[float]:
    if not vectors:
        return [0.0] * width
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(width)]


def _linear(weights: list[list[float]], bias: list[float], x: list[float]) -> list[float]:
    return [
        b + sum(w * v for w, v in zip(row, x, strict=True))
        for row, b in zip(weights, bias, strict=True)
    ]


def _silu(x: float) -> float:
    return x / (1.0 + math.exp(-x)) if x > -40.0 else 0.0


def _tensor_payload(weights: dict[str, Any]) -> dict[str, Any]:
    """Export portable floats; forbid NaNs/Infs in any nested tensor."""
    def visit(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [visit(x) for x in value]
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite v15c model weight")
        return number
    return {key: visit(value) for key, value in weights.items()}


class PortableTypedEncoder:
    """Exact algebra of v15b's embedding module, without Torch dependency."""

    def __init__(self, vocabulary: ObjectVocabulary, width: int,
                 params: dict[str, Any]) -> None:
        self.vocabulary = vocabulary
        self.width = width
        self.params = params
        self.maps = {
            name: {item: index for index, item in enumerate(getattr(vocabulary, name))}
            for name in ("kinds", "identities", "roles", "categories", "numeric_fields")
        }

    def _lookup(self, table: str, key: str) -> list[float]:
        return self.params[table + ".weight"][self.maps[table].get(key, 0)]

    def _numerical(self, kind_idx: int, field: str, value: float) -> list[float]:
        field_embed = self._lookup("numeric_fields", field)
        input_vec = field_embed + [
            math.copysign(math.log1p(abs(value)), value) / 4.0
        ]
        prefix = "numeric_layers." + str(kind_idx)
        u = _linear(
            self.params[prefix + ".0.weight"],
            self.params[prefix + ".0.bias"], input_vec,
        )
        return _linear(
            self.params[prefix + ".2.weight"],
            self.params[prefix + ".2.bias"],
            [_silu(v) for v in u],
        )

    def encode(self, objects: ObjectSet) -> list[list[float]]:
        rows: list[list[float]] = []
        for token in objects.tokens:
            kind = self.maps["kinds"].get(token.kind, 0)
            sections = [
                self._lookup("kinds", token.kind),
                self._lookup("identities", token.kind + ":" + token.type_id),
                self._lookup("roles", token.kind + ":" + token.role),
            ]
            if token.categorical:
                sections.append(_mean([
                    self._lookup("categories", token.kind + ":" + tag)
                    for tag in token.categorical
                ], self.width))
            if token.numeric:
                sections.append(_mean([
                    self._numerical(kind, token.kind + ":" + field, value)
                    for field, value in token.numeric
                ], self.width))
            x = [sum(section[i] for section in sections) for i in range(self.width)]
            avg = sum(x) / self.width
            var = sum((v - avg) ** 2 for v in x) / self.width
            inv = 1.0 / math.sqrt(var + 1.0e-5)
            weight, bias = self.params["norm.weight"], self.params["norm.bias"]
            rows.append([(x[i] - avg) * inv * weight[i] + bias[i]
                         for i in range(self.width)])
        return rows


class PooledObjectTorch:
    """Differentiable v15c head over a *frozen* legacy combat model.

    This control has no object self-attention or cross-attention. A legal
    action receives the mean embedding of objects it explicitly references.
    """

    def __init__(self, vocabulary: ObjectVocabulary, *, width: int = 64) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.width = width
        self.vocabulary = vocabulary
        self.embedding = TypedObjectEncoder(vocabulary, width=width)

        class Residual(nn.Module):
            def __init__(self, encoder: Any, n: int) -> None:
                super().__init__()
                self.encoder = encoder
                self.state_proj = nn.Linear(n, n)
                self.action_proj = nn.Linear(n, n)
                self.policy_readout = nn.Linear(n, 1)
                self.value_readout = nn.Linear(n, 1)
                self.policy_gate = nn.Parameter(torch.zeros(()))
                self.value_gate = nn.Parameter(torch.zeros(()))

        self.module = Residual(self.embedding.module, width)

    def forward_prepared(
        self, objects: ObjectSet,
        action_refs: Sequence[tuple[ObjectRef, ...]],
        baseline: PolicyValueEstimate,
    ) -> tuple[Any, Any]:
        """Trainable forward with frozen base outputs and parsed references.

        Cache the public object extraction and base predictions once per
        rollout cohort; *recompute* embeddings every optimizer minibatch.
        """
        torch = self.torch
        embedded = self.embedding.encode(objects)
        context = embedded.mean(dim=0)
        shared = self.module.state_proj(context)
        indices = objects.index()
        value0 = torch.tensor(
            baseline.value, dtype=embedded.dtype, device=embedded.device,
        )
        logits0 = torch.tensor(
            baseline.action_logits, dtype=embedded.dtype, device=embedded.device,
        )
        value_delta = self.module.value_readout(torch.tanh(shared)).squeeze()
        value = (
            value0 + torch.tanh(self.module.value_gate) * torch.tanh(value_delta)
            * (1.0 - value0.abs())
        )
        if not action_refs:
            return value, logits0
        if len(action_refs) != len(baseline.action_logits):
            raise ValueError("v15c cached action references/logits mismatch")
        action_vectors = [
            embedded[[indices[ref] for ref in refs]].mean(dim=0)
            if refs else torch.zeros_like(context)
            for refs in action_refs
        ]
        action_x = torch.stack(action_vectors, dim=0)
        updates = self.module.policy_readout(
            torch.tanh(shared.unsqueeze(0) + self.module.action_proj(action_x))
        ).squeeze(-1)
        return value, logits0 + self.module.policy_gate * updates

    def forward(
        self, base: NeuralPolicyValueModel, observation: Observation,
        actions: Sequence[LegalAction],
    ) -> tuple[Any, Any]:
        """Compute differentiable residuals; base outputs stay frozen."""
        state = json.loads(observation.payload_json)
        if not isinstance(state, dict):
            raise ValueError("Invalid public combat observation")
        objects = extract_actionable_combat_objects(
            state, [(a.kind, a.payload_json) for a in actions],
        )
        action_refs = tuple(
            legal_action_object_refs(action.kind, action.payload_json, objects)
            for action in actions
        )
        baseline = base.evaluate(observation, actions)
        return self.forward_prepared(objects, action_refs, baseline)

    def export(self, base: NeuralPolicyValueModel) -> PortablePooledObjectModel:
        weights = {
            name: tensor.detach().cpu().tolist()
            for name, tensor in self.module.state_dict().items()
        }
        return PortablePooledObjectModel.from_dict({
            "format": POOLED_OBJECT_FORMAT,
            "model_id": base.model_id + "-pooled-v15c",
            "base": base.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "width": self.width,
            "weights": _tensor_payload(weights),
        })


@dataclass(slots=True)
class PortablePooledObjectModel:
    """Versioned combat head that can be used by phase-split inference."""

    model_id: str
    base: NeuralPolicyValueModel
    vocabulary: ObjectVocabulary
    width: int
    weights: dict[str, Any]
    format_id: str = POOLED_OBJECT_FORMAT

    @property
    def dimension(self) -> int:
        return self.base.dimension

    @property
    def hidden(self) -> int:
        return self.base.hidden

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PortablePooledObjectModel:
        if raw.get("format") != POOLED_OBJECT_FORMAT:
            raise ValueError("Unknown v15c pooled-object format")
        if (type(raw.get("width")) is not int or raw["width"] <= 0
                or not isinstance(raw.get("base"), dict)
                or not isinstance(raw.get("weights"), dict)
                or not isinstance(raw.get("vocabulary"), dict)
                or not isinstance(raw.get("model_id"), str)
                or not raw["model_id"]):
            raise ValueError("Invalid v15c checkpoint metadata")
        model = cls(
            model_id=raw["model_id"],
            base=NeuralPolicyValueModel.from_dict(raw["base"]),
            vocabulary=ObjectVocabulary.from_dict(raw["vocabulary"]),
            width=raw["width"],
            weights=_tensor_payload(raw["weights"]),
        )
        model._validate_weights()
        return model

    def _validate_weights(self) -> None:
        """Reject incomplete/malformed tensors before evaluating a checkpoint."""
        w = self.width
        sizes = {
            "encoder.kinds.weight": (len(self.vocabulary.kinds), w),
            "encoder.identities.weight": (len(self.vocabulary.identities), w),
            "encoder.roles.weight": (len(self.vocabulary.roles), w),
            "encoder.categories.weight": (len(self.vocabulary.categories), w),
            "encoder.numeric_fields.weight": (len(self.vocabulary.numeric_fields), w),
            "encoder.norm.weight": (w,),
            "encoder.norm.bias": (w,),
            "state_proj.weight": (w, w),
            "state_proj.bias": (w,),
            "action_proj.weight": (w, w),
            "action_proj.bias": (w,),
            "policy_readout.weight": (1, w),
            "policy_readout.bias": (1,),
            "value_readout.weight": (1, w),
            "value_readout.bias": (1,),
            "policy_gate": (),
            "value_gate": (),
        }
        for i in range(len(self.vocabulary.kinds)):
            p = "encoder.numeric_layers." + str(i)
            sizes.update({
                p + ".0.weight": (w, w + 1), p + ".0.bias": (w,),
                p + ".2.weight": (w, w), p + ".2.bias": (w,),
            })
        if set(self.weights) != set(sizes):
            raise ValueError("Incomplete v15c tensor set")
        def check(value: Any, shape: tuple[int, ...], name: str) -> None:
            if not shape:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("Invalid v15c scalar: " + name)
                return
            if not isinstance(value, list) or len(value) != shape[0]:
                raise ValueError("Incorrect v15c tensor shape: " + name)
            for entry in value:
                check(entry, shape[1:], name)

        for name, shape in sizes.items():
            check(self.weights[name], shape, name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format_id, "model_id": self.model_id,
            "base": self.base.to_dict(), "vocabulary": self.vocabulary.to_dict(),
            "width": self.width, "weights": self.weights,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True) + "\n")

    def evaluate(
        self, observation: Observation, legal_actions: Sequence[LegalAction],
    ) -> PolicyValueEstimate:
        baseline = self.base.evaluate(observation, legal_actions)
        gates = self.weights["policy_gate"], self.weights["value_gate"]
        if gates == (0.0, 0.0):
            # Exact warm-start preservation, and no need to encode the state.
            return baseline

        state = json.loads(observation.payload_json)
        if not isinstance(state, dict):
            raise ValueError("Invalid public combat observation")
        objects = extract_actionable_combat_objects(
            state, [(a.kind, a.payload_json) for a in legal_actions],
        )
        params = {name.removeprefix("encoder."): value
                  for name, value in self.weights.items() if name.startswith("encoder.")}
        rows = PortableTypedEncoder(self.vocabulary, self.width, params).encode(objects)
        shared = _linear(
            self.weights["state_proj.weight"], self.weights["state_proj.bias"],
            _mean(rows, self.width),
        )
        value_delta = _linear(
            self.weights["value_readout.weight"], self.weights["value_readout.bias"],
            [math.tanh(x) for x in shared],
        )[0]
        value = (baseline.value + math.tanh(gates[1]) * math.tanh(value_delta)
                 * (1.0 - abs(baseline.value)))
        index = objects.index()
        logits = []
        for action, base_logit in zip(legal_actions, baseline.action_logits, strict=True):
            refs = legal_action_object_refs(action.kind, action.payload_json, objects)
            chosen = _mean([rows[index[ref]] for ref in refs], self.width)
            action_projection = _linear(
                self.weights["action_proj.weight"], self.weights["action_proj.bias"],
                chosen,
            )
            merged = [math.tanh(a + b)
                      for a, b in zip(shared, action_projection, strict=True)]
            delta = _linear(
                self.weights["policy_readout.weight"],
                self.weights["policy_readout.bias"], merged,
            )[0]
            logits.append(base_logit + gates[0] * delta)
        return PolicyValueEstimate(tuple(logits), value)
