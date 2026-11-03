"""H20 sparse map-edge message passing and strictly versioned checkpoint."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sts2_ai.models.shared_game_objects import EncodedDecision
from sts2_ai.models.shared_owner_transformer_v2 import H19OwnerTransformer
from sts2_ai.models.shared_transformer import H7Batch, H7Config, H20Batch, collate_h7, vars_as_dict
from sts2_ai.training.h20_map_graph_codec import (
    H20_MAP_CODEC,
    H20_OBJECT_SCHEMA,
    H20_RELATION_TYPES,
)

H20_MODEL_FORMAT = "sts2-h20-relational-map-transformer-v1"


def collate_h20(decisions: list[EncodedDecision]) -> H7Batch:
    batch = collate_h7(decisions, expected_schema=H20_OBJECT_SCHEMA)
    n = len(decisions)
    max_edges = max(1, *(len(item.relations) for item in decisions))
    types = torch.zeros(n, max_edges, dtype=torch.long)
    sources = torch.full_like(types, -1)
    targets = torch.full_like(types, -1)
    mask = torch.zeros(n, max_edges, dtype=torch.bool)
    for row, decision in enumerate(decisions):
        for j, edge in enumerate(decision.relations):
            if (not 0 <= edge.relation_type < H20_RELATION_TYPES
                    or not 0 <= edge.source < len(decision.objects)
                    or not 0 <= edge.target < len(decision.objects)):
                raise ValueError("Dangling or invalid H20 public map edge")
            types[row, j] = edge.relation_type
            sources[row, j] = edge.source
            targets[row, j] = edge.target
            mask[row, j] = True
    return H20Batch(
        **vars_as_dict(batch),
        relation_types=types,
        relation_sources=sources,
        relation_targets=targets,
        relation_mask=mask,
    )


class H20MapGraphTransformer(H19OwnerTransformer):
    """Message passing over visible DAG edges before shared object attention.

    Each NEXT edge sends downstream room features toward its predecessor.
    Current/reachable edges send selected/current room context toward the
    global object. Three rounds distinguish local branching structures without
    expanding combinatorial path choices or embedding raw NodeIds.
    """

    def __init__(self, config: H7Config | None = None) -> None:
        super().__init__(config)
        width = self.config.width
        self.relation_types = nn.Embedding(H20_RELATION_TYPES, width)
        self.relation_target = nn.Linear(width, width, bias=False)
        self.relation_output = nn.Linear(width, width, bias=False)
        self.relation_norm = nn.LayerNorm(width)

    def _object_embeddings(self, batch: H7Batch) -> Tensor:
        embedding = super()._object_embeddings(batch)
        if (batch.relation_types is None or batch.relation_sources is None
                or batch.relation_targets is None or batch.relation_mask is None):
            raise ValueError("H20 model requires typed public map relations")
        sources = batch.relation_sources
        targets = batch.relation_targets
        mask = batch.relation_mask
        dim = embedding.shape[-1]
        valid = mask.unsqueeze(-1)
        for _ in range(3):
            target_features = self._point(embedding, targets)
            messages = self.relation_target(target_features) + self.relation_types(
                batch.relation_types)
            messages = messages * valid
            accum = torch.zeros_like(embedding).scatter_add(
                1, sources.clamp(min=0).unsqueeze(-1).expand(-1, -1, dim),
                messages)
            counts = torch.zeros_like(embedding[..., :1]).scatter_add(
                1, sources.clamp(min=0).unsqueeze(-1),
                valid.to(embedding.dtype))
            update = self.relation_output(accum / counts.clamp(min=1))
            # Padding remains masked in the shared Transformer.
            embedding = self.relation_norm(embedding + update)
        return embedding

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": H20_MODEL_FORMAT,
            "object_schema": H20_OBJECT_SCHEMA,
            "codec": H20_MAP_CODEC,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(
        cls, path: Path, *, map_location: Any = "cpu",
    ) -> H20MapGraphTransformer:
        data = torch.load(path, map_location=map_location, weights_only=True)
        if (data.get("format") != H20_MODEL_FORMAT
                or data.get("object_schema") != H20_OBJECT_SCHEMA
                or data.get("codec") != H20_MAP_CODEC):
            raise ValueError("Unsupported H20 map-graph model/codec version")
        model = cls(H7Config(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return model
