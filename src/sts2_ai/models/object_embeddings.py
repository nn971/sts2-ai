"""v15b: typed trainable embeddings for *public* mutable object instances.

Prototype module; no existing v8/v13 checkpoint or PPO path is modified.
Vocabularies are explicit and immutable after creation. Missing/OOV labels
have reserved embeddings rather than being silently dropped.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
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
    def from_object_sets(
        samples: Sequence[ObjectSet], *,
        manifest: Mapping[str, Any] | None = None,
    ) -> ObjectVocabulary:
        """Build a frozen vocabulary from public observations + declared catalog.

        The bridge's capability manifest contains only implemented content
        identifiers, not engine state or private RNG. It prevents unseen
        catalog cards/enemies/relics/potions/powers from all sharing <UNK>.
        Dynamic categorical tags and numeric fields still come from
        *public* samples; their OOV rate is separately reportable.
        """
        tokens = [token for sample in samples for token in sample.tokens]

        def entries(values: Any) -> tuple[str, ...]:
            return (UNKNOWN, *sorted(set(values) - {UNKNOWN}))

        catalog: set[str] = {"global:player"}
        roles: set[str] = {
            "global:combat",
            "card:hand", "card:draw_pile", "card:discard_pile",
            "card:exhaust_pile", "card:play_pile",
            "enemy:living", "relic:owned", "potion:held",
            "power:active", "status:active",
        }
        if manifest is not None:
            for kind, field in (
                ("card", "cardIds"), ("enemy", "enemyIds"),
                ("relic", "relicIds"), ("potion", "potionIds"),
                ("power", "powerIds"),
            ):
                values = manifest.get(field)
                if (not isinstance(values, list)
                        or any(not isinstance(v, str) or not v for v in values)
                        or len(set(values)) != len(values)):
                    raise ValueError(f"Invalid emulator manifest {field}")
                catalog.update(f"{kind}:{value}" for value in values)
        return ObjectVocabulary(
            kinds=entries(
                [token.kind for token in tokens]
                + ["global", "card", "enemy", "relic", "potion", "power", "status"]
            ),
            identities=entries(
                catalog | {f"{token.kind}:{token.type_id}" for token in tokens}
            ),
            roles=entries(
                roles | {f"{token.kind}:{token.role}" for token in tokens}
            ),
            categories=entries(
                f"{token.kind}:{tag}"
                for token in tokens for tag in token.categorical
            ),
            numeric_fields=entries(
                f"{token.kind}:{name}"
                for token in tokens for name, _ in token.numeric
            ),
        )

    def oov_report(self, samples: Sequence[ObjectSet]) -> dict[str, Any]:
        """Audit OOV use; no raw instance IDs are included in the report."""
        known = {
            name: set(getattr(self, name))
            for name in ("kinds", "identities", "roles", "categories", "numeric_fields")
        }
        total = {name: 0 for name in known}
        missing: dict[str, dict[str, int]] = {name: {} for name in known}

        def observe(namespace: str, key: str) -> None:
            total[namespace] += 1
            if key not in known[namespace]:
                items = missing[namespace]
                items[key] = items.get(key, 0) + 1

        for sample in samples:
            for token in sample.tokens:
                observe("kinds", token.kind)
                observe("identities", f"{token.kind}:{token.type_id}")
                observe("roles", f"{token.kind}:{token.role}")
                for tag in token.categorical:
                    observe("categories", f"{token.kind}:{tag}")
                for field, _ in token.numeric:
                    observe("numeric_fields", f"{token.kind}:{field}")

        return {
            name: {
                "total": total[name],
                "unknown": sum(missing[name].values()),
                "unknown_fraction": (
                    sum(missing[name].values()) / total[name]
                    if total[name] else 0.0
                ),
                "examples": sorted(
                    missing[name], key=lambda key: (-missing[name][key], key),
                )[:20],
            }
            for name in known
        }

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



@dataclass(frozen=True, slots=True)
class CompiledObjectFeatures:
    """Public, weight-independent token indices; safe across PPO epochs.

    Instance references, numeric instance IDs, and private emulator state
    are not represented. Numeric *values* are frozen only until the public
    observation changes; PPO replays immutable captured decisions.
    """

    count: int
    kinds: tuple[int, ...]
    identities: tuple[int, ...]
    roles: tuple[int, ...]
    category_rows: tuple[int, ...]
    category_ids: tuple[int, ...]
    category_counts: tuple[int, ...]
    numeric_groups: tuple[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[float, ...]], ...
    ]
    numeric_counts: tuple[int, ...]



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


    def prepare(self, objects: ObjectSet) -> CompiledObjectFeatures:
        """Compile visible categorical/numeric labels once, not neural values."""
        kind_ids = []
        identities = []
        roles = []
        category_rows = []
        category_ids = []
        category_counts = []
        numeric_rows: dict[int, list[int]] = {}
        numeric_ids: dict[int, list[int]] = {}
        numeric_values: dict[int, list[float]] = {}
        numeric_counts = []
        for row, token in enumerate(objects.tokens):
            kind = self._id("kinds", token.kind)
            kind_ids.append(kind)
            identities.append(self._id("identities", f"{token.kind}:{token.type_id}"))
            roles.append(self._id("roles", f"{token.kind}:{token.role}"))
            category_counts.append(max(1, len(token.categorical)))
            numeric_counts.append(max(1, len(token.numeric)))
            for tag in token.categorical:
                category_rows.append(row)
                category_ids.append(
                    self._id("categories", f"{token.kind}:{tag}")
                )
            for field, value in token.numeric:
                numeric_rows.setdefault(kind, []).append(row)
                numeric_ids.setdefault(kind, []).append(
                    self._id("numeric_fields", f"{token.kind}:{field}")
                )
                numeric_values.setdefault(kind, []).append(
                    math.copysign(math.log1p(abs(value)), value) / 4.0
                )
        return CompiledObjectFeatures(
            count=len(objects.tokens),
            kinds=tuple(kind_ids),
            identities=tuple(identities),
            roles=tuple(roles),
            category_rows=tuple(category_rows),
            category_ids=tuple(category_ids),
            category_counts=tuple(category_counts),
            numeric_groups=tuple(
                (
                    kind, tuple(numeric_rows[kind]), tuple(numeric_ids[kind]),
                    tuple(numeric_values[kind]),
                )
                for kind in sorted(numeric_rows)
            ),
            numeric_counts=tuple(numeric_counts),
        )

    def encode_many_compiled(
        self, prepared: Sequence[CompiledObjectFeatures],
    ) -> tuple[Any, ...]:
        """Encode pre-indexed public tokens with the *current* model weights.

        All Torch embeddings, projections, norms and gradients are freshly
        evaluated on each call. No stale tensor, autograd graph or private
        state is retained by the cache.
        """
        torch = self.torch
        device = self.module.kinds.weight.device
        dtype = self.module.kinds.weight.dtype
        lengths = [row.count for row in prepared]
        total = sum(lengths)
        if total == 0:
            return tuple(
                self.module.kinds.weight.new_zeros((length, self.width))
                for length in lengths
            )

        def ids(values: Sequence[int]) -> Any:
            return torch.tensor(values, dtype=torch.long, device=device)

        kinds = [v for row in prepared for v in row.kinds]
        identities = [v for row in prepared for v in row.identities]
        roles = [v for row in prepared for v in row.roles]
        x = (
            self.module.kinds(ids(kinds))
            + self.module.identities(ids(identities))
            + self.module.roles(ids(roles))
        )
        cat_rows: list[int] = []
        cat_ids: list[int] = []
        cat_counts: list[int] = []
        offset = 0
        for row in prepared:
            cat_rows.extend(offset + local for local in row.category_rows)
            cat_ids.extend(row.category_ids)
            cat_counts.extend(row.category_counts)
            offset += row.count
        if cat_rows:
            vectors = self.module.categories(ids(cat_ids))
            sums = torch.zeros_like(x).index_add(0, ids(cat_rows), vectors)
            counts = torch.tensor(
                cat_counts, device=device, dtype=dtype,
            ).unsqueeze(-1)
            x = x + sums / counts

        numeric_by_kind: dict[
            int, tuple[list[int], list[int], list[float]]
        ] = {}
        num_counts: list[int] = []
        offset = 0
        for row in prepared:
            num_counts.extend(row.numeric_counts)
            for kind, positions, fields, values in row.numeric_groups:
                bucket = numeric_by_kind.setdefault(kind, ([], [], []))
                bucket[0].extend(offset + local for local in positions)
                bucket[1].extend(fields)
                bucket[2].extend(values)
            offset += row.count
        if numeric_by_kind:
            sums = torch.zeros_like(x)
            for kind in sorted(numeric_by_kind):
                positions, fields, values = numeric_by_kind[kind]
                field_vectors = self.module.numeric_fields(ids(fields))
                scalar = torch.tensor(
                    values, device=device, dtype=dtype,
                ).unsqueeze(-1)
                projected = self.module.numeric_layers[kind](
                    torch.cat((field_vectors, scalar), dim=-1)
                )
                sums = sums.index_add(0, ids(positions), projected)
            counts = torch.tensor(
                num_counts, device=device, dtype=dtype,
            ).unsqueeze(-1)
            x = x + sums / counts
        result = self.module.norm(x)
        return tuple(torch.split(result, lengths, dim=0))

    def encode_many(self, batches: Sequence[ObjectSet]) -> tuple[Any, ...]:
        """Encode all visible tokens in one differentiable vectorized pass.

        The previous implementation called Torch operations separately for
        every token and each numeric field. Flattening across decisions
        reduces Python/Torch dispatch, while segmented reductions preserve
        *exact* token multiplicity and all v15 embedding semantics.

        No instance/reference values are converted to neural inputs. This
        also preserves empty batch members and per-decision boundaries.
        """
        torch = self.torch
        device = self.module.kinds.weight.device
        dtype = self.module.kinds.weight.dtype
        lengths = [len(batch.tokens) for batch in batches]
        tokens = [token for batch in batches for token in batch.tokens]
        n = len(tokens)
        if not n:
            return tuple(
                self.module.kinds.weight.new_zeros((length, self.width))
                for length in lengths
            )

        def indices(values: list[int]) -> Any:
            return torch.tensor(values, device=device, dtype=torch.long)

        kind_ids = [self._id("kinds", token.kind) for token in tokens]
        identity_ids = [
            self._id("identities", f"{token.kind}:{token.type_id}")
            for token in tokens
        ]
        role_ids = [
            self._id("roles", f"{token.kind}:{token.role}")
            for token in tokens
        ]
        x = (
            self.module.kinds(indices(kind_ids))
            + self.module.identities(indices(identity_ids))
            + self.module.roles(indices(role_ids))
        )

        category_rows: list[int] = []
        category_ids: list[int] = []
        for row, token in enumerate(tokens):
            for tag in token.categorical:
                category_rows.append(row)
                category_ids.append(self._id("categories", f"{token.kind}:{tag}"))
        if category_rows:
            categories = self.module.categories(indices(category_ids))
            sums = torch.zeros_like(x).index_add(
                0, indices(category_rows), categories,
            )
            counts = torch.tensor(
                [max(1, len(token.categorical)) for token in tokens],
                dtype=dtype, device=device,
            ).unsqueeze(-1)
            x = x + sums / counts

        # Numeric layers are kind-specific. Batch all scalar fields of the
        # same kind, then scatter the per-field outputs back to their tokens.
        numeric_rows: dict[int, list[int]] = {}
        numeric_ids: dict[int, list[int]] = {}
        numeric_values: dict[int, list[float]] = {}
        for row, (kind, token) in enumerate(zip(kind_ids, tokens, strict=True)):
            for field, value in token.numeric:
                numeric_rows.setdefault(kind, []).append(row)
                numeric_ids.setdefault(kind, []).append(
                    self._id("numeric_fields", f"{token.kind}:{field}")
                )
                numeric_values.setdefault(kind, []).append(
                    math.copysign(math.log1p(abs(value)), value) / 4.0
                )
        if numeric_rows:
            sums = torch.zeros_like(x)
            for kind in sorted(numeric_rows):
                fields = self.module.numeric_fields(indices(numeric_ids[kind]))
                values = torch.tensor(
                    numeric_values[kind], dtype=dtype, device=device,
                ).unsqueeze(-1)
                projected = self.module.numeric_layers[kind](
                    torch.cat((fields, values), dim=-1)
                )
                sums = sums.index_add(0, indices(numeric_rows[kind]), projected)
            counts = torch.tensor(
                [max(1, len(token.numeric)) for token in tokens],
                dtype=dtype, device=device,
            ).unsqueeze(-1)
            x = x + sums / counts

        embeddings = self.module.norm(x)
        return tuple(torch.split(embeddings, lengths, dim=0))

    def encode(self, objects: ObjectSet) -> Any:
        """Return [N,width]; refs and their numeric IDs remain outside Torch."""
        return self.encode_many((objects,))[0]

    @staticmethod
    def reference_rows(objects: ObjectSet, refs: Sequence[ObjectRef]) -> tuple[int, ...]:
        index = objects.index()
        return tuple(index[ref] for ref in refs)
