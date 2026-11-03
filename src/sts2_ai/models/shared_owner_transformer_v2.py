"""Isolated H19 owner-attribute vocabulary and checkpoint format.

The original H7/H16 shared model remains byte-compatible with its historical
weights; owner-linked attributes use one additional learnable token kind and
a strictly versioned checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sts2_ai.models.shared_game_objects import (
    OWNER_ATTRIBUTE_KIND_V2,
    EncodedDecision,
)
from sts2_ai.models.shared_transformer import (
    H7Batch,
    H7Config,
    SharedObjectTransformer,
    collate_h7,
)
from sts2_ai.training.h19_resource_codec import (
    H19_OBJECT_SCHEMA,
    H19_RESOURCE_CODEC,
)

H19_MODEL_FORMAT = "sts2-h19-owner-shared-transformer-v1"


def collate_h19(decisions: list[EncodedDecision]) -> H7Batch:
    return collate_h7(decisions, expected_schema=H19_OBJECT_SCHEMA)


class H19OwnerTransformer(SharedObjectTransformer):
    """V2 kind vocabulary, with unchanged H7 numeric and relation channels."""

    def __init__(self, config: H7Config | None = None) -> None:
        super().__init__(config)
        self.kinds = nn.Embedding(OWNER_ATTRIBUTE_KIND_V2 + 1, self.config.width)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": H19_MODEL_FORMAT,
            "object_schema": H19_OBJECT_SCHEMA,
            "codec": H19_RESOURCE_CODEC,
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(
        cls, path: Path, *, map_location: Any = "cpu",
    ) -> H19OwnerTransformer:
        data = torch.load(path, map_location=map_location, weights_only=True)
        if (data.get("format") != H19_MODEL_FORMAT
                or data.get("object_schema") != H19_OBJECT_SCHEMA
                or data.get("codec") != H19_RESOURCE_CODEC):
            raise ValueError("Unsupported H19 owner-resource model/codec version")
        model = cls(H7Config(**data["config"]))
        model.load_state_dict(data["state_dict"], strict=True)
        return model
