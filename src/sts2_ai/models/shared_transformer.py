"""H7 opt-in trainable shared Transformer for public game-state object tokens.

Torch-only research module, deliberately separate from legacy portable JSON
combat and H6 macro checkpoints. Output is per-legal-action logits and a
phase-specific value. No teacher/pretrained weights are loaded implicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .shared_game_objects import (
    ACTION_BUCKETS,
    IDENTITY_BUCKETS,
    KINDS,
    NUMERIC_FIELDS,
    SHARED_OBJECT_SCHEMA,
    ZONES,
    EncodedDecision,
)

H7_MODEL_FORMAT = "sts2-h7-shared-object-transformer-v1"


@dataclass(frozen=True)
class H7Config:
    width: int = 384
    layers: int = 6
    heads: int = 8
    feedforward: int = 1536
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (self.width <= 0 or self.layers <= 0 or self.heads <= 0
                or self.feedforward <= 0 or self.width % self.heads
                or not 0 <= self.dropout < 1):
            raise ValueError("Invalid H7 Transformer configuration")


@dataclass(frozen=True)
class H7Batch:
    kinds: Tensor
    identities: Tensor
    zones: Tensor
    numeric: Tensor
    observed: Tensor
    owners: Tensor
    object_mask: Tensor
    action_kinds: Tensor
    action_semantics: Tensor
    sources: Tensor
    targets: Tensor
    action_mask: Tensor
    phases: Tensor

    def to(self, device: torch.device | str) -> H7Batch:
        return H7Batch(**{name: tensor.to(device) for name, tensor in vars_as_dict(self).items()})


def vars_as_dict(batch: H7Batch) -> dict[str, Tensor]:
    return {name: getattr(batch, name) for name in H7Batch.__dataclass_fields__}


def collate_h7(decisions: list[EncodedDecision]) -> H7Batch:
    if not decisions or any(item.schema != SHARED_OBJECT_SCHEMA for item in decisions):
        raise ValueError("Need nonempty same-version H7 decisions")
    if any(not item.objects or not item.actions for item in decisions):
        raise ValueError("H7 decisions require objects and legal actions")
    batch = len(decisions)
    max_obj = max(len(item.objects) for item in decisions)
    max_actions = max(len(item.actions) for item in decisions)
    nfeatures = len(NUMERIC_FIELDS)
    kinds = torch.zeros(batch, max_obj, dtype=torch.long)
    identities = torch.zeros_like(kinds)
    zones = torch.zeros_like(kinds)
    numeric = torch.zeros(batch, max_obj, nfeatures, dtype=torch.float32)
    observed = torch.zeros_like(numeric)
    owners = torch.full_like(kinds, -1)
    object_mask = torch.zeros(batch, max_obj, dtype=torch.bool)
    action_kinds = torch.zeros(batch, max_actions, dtype=torch.long)
    action_semantics = torch.zeros_like(action_kinds)
    sources = torch.full_like(action_kinds, -1)
    targets = torch.full_like(action_kinds, -1)
    action_mask = torch.zeros(batch, max_actions, dtype=torch.bool)
    phases = torch.zeros(batch, dtype=torch.long)
    for row, decision in enumerate(decisions):
        phases[row] = 0 if decision.phase == "combat" else 1 if decision.phase == "macro" else -1
        if phases[row] < 0:
            raise ValueError("Unrecognized H7 decision phase")
        for pos, token in enumerate(decision.objects):
            kinds[row, pos] = token.kind
            identities[row, pos] = token.identity
            zones[row, pos] = token.zone
            numeric[row, pos] = torch.tensor(token.numeric)
            observed[row, pos] = torch.tensor(token.observed)
            if token.owner >= len(decision.objects):
                raise ValueError("Dangling public object owner")
            owners[row, pos] = token.owner
        object_mask[row, :len(decision.objects)] = True
        for pos, action in enumerate(decision.actions):
            for ptr in (action.source, action.target):
                if ptr >= len(decision.objects) or ptr < -1:
                    raise ValueError("Dangling action pointer")
            action_kinds[row, pos] = action.kind
            action_semantics[row, pos] = action.semantic
            sources[row, pos] = action.source
            targets[row, pos] = action.target
        action_mask[row, :len(decision.actions)] = True
    return H7Batch(
        kinds, identities, zones, numeric, observed, owners, object_mask,
        action_kinds, action_semantics, sources, targets, action_mask, phases,
    )


class SharedObjectTransformer(nn.Module):
    """A single backbone with independent combat/macro action and value heads."""

    def __init__(self, config: H7Config | None = None) -> None:
        super().__init__()
        self.config = config or H7Config()
        dim = self.config.width
        self.kinds = nn.Embedding(len(KINDS), dim)
        self.identities = nn.Embedding(IDENTITY_BUCKETS, dim, padding_idx=0)
        self.zones = nn.Embedding(len(ZONES), dim)
        self.numerics = nn.Linear(2 * len(NUMERIC_FIELDS), dim)
        self.input_norm = nn.LayerNorm(dim)
        self.owner_projection = nn.Linear(dim, dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=self.config.heads,
            dim_feedforward=self.config.feedforward, dropout=self.config.dropout,
            activation="gelu", norm_first=True, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, self.config.layers, enable_nested_tensor=False,
        )
        self.context_norm = nn.LayerNorm(dim)
        self.action_kinds = nn.Embedding(ACTION_BUCKETS, dim, padding_idx=0)
        self.action_source = nn.Linear(dim, dim, bias=False)
        self.action_target = nn.Linear(dim, dim, bias=False)
        self.action_attention = nn.MultiheadAttention(
            dim, self.config.heads, dropout=self.config.dropout, batch_first=True,
        )
        self.action_norm = nn.LayerNorm(dim)
        self.policy_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            for _ in range(2)
        ])
        self.value_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
            for _ in range(2)
        ])

    @staticmethod
    def _point(objects: Tensor, indices: Tensor) -> Tensor:
        width = objects.shape[-1]
        gathered = objects.gather(
            1, indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, width),
        )
        return gathered * (indices >= 0).unsqueeze(-1)

    def forward(self, batch: H7Batch) -> tuple[Tensor, Tensor]:
        if not batch.object_mask[:, 0].all() or not batch.action_mask.any(dim=1).all():
            raise ValueError("H7 requires global token and at least one valid action")
        embedding = (
            self.kinds(batch.kinds) + self.identities(batch.identities)
            + self.zones(batch.zones)
            + self.numerics(torch.cat([batch.numeric, batch.observed], dim=-1))
        )
        embedding = self.input_norm(embedding)
        embedding = embedding + self.owner_projection(self._point(embedding, batch.owners))
        context = self.context_norm(self.encoder(
            embedding, src_key_padding_mask=~batch.object_mask,
        ))
        context = context.masked_fill(~batch.object_mask.unsqueeze(-1), 0.0)
        queries = (
            self.action_kinds(batch.action_kinds)
            + self.identities(batch.action_semantics)
            + self.action_source(self._point(context, batch.sources))
            + self.action_target(self._point(context, batch.targets))
        )
        attended, _ = self.action_attention(
            queries, context, context, key_padding_mask=~batch.object_mask,
            need_weights=False,
        )
        attended = self.action_norm(queries + attended)
        combat_logits = self.policy_heads[0](attended).squeeze(-1)
        macro_logits = self.policy_heads[1](attended).squeeze(-1)
        phase = (batch.phases == 0).unsqueeze(1)
        logits = torch.where(phase, combat_logits, macro_logits)
        value0 = torch.tanh(self.value_heads[0](context[:, 0]).squeeze(-1))
        value1 = torch.tanh(self.value_heads[1](context[:, 0]).squeeze(-1))
        values = torch.where(batch.phases == 0, value0, value1)
        logits = logits.masked_fill(~batch.action_mask, torch.finfo(logits.dtype).min)
        return logits, values

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": H7_MODEL_FORMAT,
            "object_schema": SHARED_OBJECT_SCHEMA,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(cls, path: Path, *, map_location: Any = "cpu") -> SharedObjectTransformer:
        data = torch.load(path, map_location=map_location, weights_only=True)
        if (data.get("format") != H7_MODEL_FORMAT
                or data.get("object_schema") != SHARED_OBJECT_SCHEMA):
            raise ValueError("Unsupported H7 model/codec version")
        model = cls(H7Config(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return model
