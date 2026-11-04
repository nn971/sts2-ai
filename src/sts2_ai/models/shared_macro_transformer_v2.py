"""H21 grounded macro model, with opt-in six-type edge vocabulary."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sts2_ai.models.shared_game_objects import EncodedDecision
from sts2_ai.models.shared_map_transformer_v2 import H20MapGraphTransformer
from sts2_ai.models.shared_transformer import H7Batch, H7Config, H20Batch, collate_h7, vars_as_dict
from sts2_ai.training.h21_macro_offer_codec import (
    H21_CODEC,
    H21_OBJECT_SCHEMA,
    H21_RELATION_TYPES,
)
from sts2_ai.training.h21_stable_identity import (
    H21_IDENTITY_COMPONENT_BUCKETS,
    H21_IDENTITY_COMPONENTS,
)

H21_MODEL_FORMAT = "sts2-h21-grounded-macro-transformer-v4"


class H21StableEmbedding(nn.Module):
    """Four hashed embedding streams representing a stable 48-bit key.

    The codec checks for collisions in the *effective* 48-bit key. There
    are no floating-point conversions or raw instance-ID embeddings.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.tables = nn.ModuleList([
            nn.Embedding(H21_IDENTITY_COMPONENT_BUCKETS, width, padding_idx=0)
            for _ in range(H21_IDENTITY_COMPONENTS)
        ])

    def forward(self, ids: Tensor) -> Tensor:
        if ids.dtype != torch.long:
            raise ValueError("H21 requires 64-bit integer public concept IDs")
        bucket = H21_IDENTITY_COMPONENT_BUCKETS
        result = self.tables[0](ids % bucket)
        for stream in range(1, H21_IDENTITY_COMPONENTS):
            result = result + self.tables[stream](
                (ids // (bucket ** stream)) % bucket)
        return result


def collate_h21(decisions: list[EncodedDecision]) -> H7Batch:
    batch = collate_h7(decisions, expected_schema=H21_OBJECT_SCHEMA)
    n = len(decisions)
    max_edges = max(1, *(len(item.relations) for item in decisions))
    types = torch.zeros(n, max_edges, dtype=torch.long)
    sources = torch.full_like(types, -1)
    targets = torch.full_like(types, -1)
    mask = torch.zeros(n, max_edges, dtype=torch.bool)
    for row, decision in enumerate(decisions):
        for col, edge in enumerate(decision.relations):
            if (not 0 <= edge.relation_type < H21_RELATION_TYPES
                    or not 0 <= edge.source < len(decision.objects)
                    or not 0 <= edge.target < len(decision.objects)):
                raise ValueError("Invalid or dangling public H21 relation")
            types[row, col] = edge.relation_type
            sources[row, col] = edge.source
            targets[row, col] = edge.target
            mask[row, col] = True
    return H20Batch(
        **vars_as_dict(batch),
        relation_types=types, relation_sources=sources,
        relation_targets=targets, relation_mask=mask,
    )


class H21GroundedMacroTransformer(H20MapGraphTransformer):
    """Preserve H20 map messages while adding macro option ownership edges."""

    def __init__(self, config: H7Config | None = None) -> None:
        super().__init__(config)
        self.relation_types = nn.Embedding(H21_RELATION_TYPES, self.config.width)
        self.identities = H21StableEmbedding(self.config.width)
        self.action_kinds = H21StableEmbedding(self.config.width)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": H21_MODEL_FORMAT,
            "object_schema": H21_OBJECT_SCHEMA,
            "codec": H21_CODEC,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(
        cls, path: Path, *, map_location: Any = "cpu",
    ) -> H21GroundedMacroTransformer:
        data = torch.load(path, map_location=map_location, weights_only=True)
        if (data.get("format") != H21_MODEL_FORMAT
                or data.get("object_schema") != H21_OBJECT_SCHEMA
                or data.get("codec") != H21_CODEC):
            raise ValueError("Unsupported H21 macro-offer model/codec version")
        model = cls(H7Config(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return model
