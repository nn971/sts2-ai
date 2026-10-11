"""v15b: typed trainable embeddings for *public* mutable object instances.

Prototype module; no existing v8/v13 checkpoint or PPO path is modified.
Vocabularies are explicit and immutable after creation. Missing/OOV labels
have reserved embeddings rather than being silently dropped.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .object_instances import ObjectRef, ObjectSet

EMBEDDING_SCHEMA = "sts2-typed-object-embeddings-v15b"
UNKNOWN = "<UNK>"


@dataclass(frozen=True, slots=True)
class ObjectVocabulary:
    kinds: tuple[str, ...]
    identities: tuple[str, ...]
    roles: tuple[str, ...]
    categories: tuple[str, ...]
    numeric_fields: tuple[str, ...]

    @staticmethod
    def from_object_sets(samples: Sequence[ObjectSet]) -> ObjectVocabulary:
        tokens = [token for sample in samples for token in sample.tokens]
        def entries(values: Any) -> tuple[str, ...]:
            return (UNKNOWN, *sorted(set(values) - {UNKNOWN}))
        return ObjectVocabulary(
            kinds=entries(token.kind for token in tokens),
            identities=entries(f"{token.kind}:{token.type_id}" for token in tokens),
            roles=entries(f"{token.kind}:{token.role}" for token in tokens),
            categories=entries(f"{token.kind}:{tag}" for token in tokens for tag in token.categorical),
            numeric_fields=entries(f"{token.kind}:{name}" for token in tokens for name, _ in token.numeric),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EMBEDDING_SCHEMA,
            "kinds": list(self.kinds), "identities": list(self.identities),
            "roles": list(self.roles), "categories": list(self.categories),
            "numeric_fields": list(self.numeric_fields),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ObjectVocabulary:
        if raw.get("schema") != EMBEDDING_SCHEMA:
            raise ValueError("Unknown object embedding vocabulary schema")
        fields: dict[str, tuple[str, ...]] = {}
        for name in ("kinds", "identities", "roles", "categories", "numeric_fields"):
            value = raw.get(name)
            if (not isinstance(value, list) or not value
                    or value[0] != UNKNOWN or any(not isinstance(x, str) for x in value)
                    or len(set(value)) != len(value)):
                raise ValueError(f"Invalid object vocabulary: {name}")
            fields[name] = tuple(value)
        return cls(**fields)


class TypedObjectEncoder:
    """Lazily requires Torch; the main package keeps dependency-free inference.

    `module` is a regular torch.nn.Module suitable for optimizer.parameters().
    Objects are encoded independently, with no hidden instance-ID embeddings.
    Attention/action scoring belong to the subsequent v15c implementation.
    """

    def __init__(self, vocabulary: ObjectVocabulary, *, width: int = 64) -> None:
        if width <= 0:
            raise ValueError("Embedding width must be positive")
        import torch
        from torch import nn

        self.vocabulary = vocabulary
        self.width = width
        self._maps = {
            name: {item: index for index, item in enumerate(getattr(vocabulary, name))}
            for name in ("kinds", "identities", "roles", "categories", "numeric_fields")
        }

        class Network(nn.Module):
            def __init__(self, vocab: ObjectVocabulary, d: int) -> None:
                super().__init__()
                self.kinds = nn.Embedding(len(vocab.kinds), d)
                self.identities = nn.Embedding(len(vocab.identities), d)
                self.roles = nn.Embedding(len(vocab.roles), d)
                self.categories = nn.Embedding(len(vocab.categories), d)
                self.numeric_fields = nn.Embedding(len(vocab.numeric_fields), d)
                self.numeric_layers = nn.ModuleList([
                    nn.Sequential(nn.Linear(d + 1, d), nn.SiLU(), nn.Linear(d, d))
                    for _ in vocab.kinds
                ])
                self.norm = nn.LayerNorm(d)

        self.module = Network(vocabulary, width)
        self.torch = torch

    def _id(self, name: str, value: str) -> int:
        return self._maps[name].get(value, 0)

    def encode(self, objects: ObjectSet) -> Any:
        """Return [N,width] tensor; explicit refs stay out of differentiable data."""
        torch = self.torch
        device = self.module.kinds.weight.device
        rows: list[Any] = []
        for token in objects.tokens:
            kind_id = self._id("kinds", token.kind)
            def index(value: int) -> Any:
                return torch.tensor(value, dtype=torch.long, device=device)
            x = (
                self.module.kinds(index(kind_id))
                + self.module.identities(index(self._id("identities", f"{token.kind}:{token.type_id}")))
                + self.module.roles(index(self._id("roles", f"{token.kind}:{token.role}")))
            )
            if token.categorical:
                idx = torch.tensor([
                    self._id("categories", f"{token.kind}:{label}")
                    for label in token.categorical
                ], dtype=torch.long, device=device)
                x = x + self.module.categories(idx).mean(dim=0)
            if token.numeric:
                ids = torch.tensor([
                    self._id("numeric_fields", f"{token.kind}:{field}")
                    for field, _ in token.numeric
                ], dtype=torch.long, device=device)
                values = torch.tensor([
                    math.copysign(math.log1p(abs(value)), value) / 4.0
                    for _, value in token.numeric
                ], dtype=self.module.kinds.weight.dtype, device=device).unsqueeze(-1)
                numeric = torch.cat((self.module.numeric_fields(ids), values), dim=-1)
                x = x + self.module.numeric_layers[kind_id](numeric).mean(dim=0)
            rows.append(self.module.norm(x))
        if not rows:
            return self.module.kinds.weight.new_zeros((0, self.width))
        return torch.stack(rows, dim=0)

    @staticmethod
    def reference_rows(objects: ObjectSet, refs: Sequence[ObjectRef]) -> tuple[int, ...]:
        index = objects.index()
        return tuple(index[ref] for ref in refs)
