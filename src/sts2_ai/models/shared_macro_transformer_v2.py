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


def collate_h21_packed(decisions: list[EncodedDecision]) -> H20Batch:
    """Opt-in P2 candidate: build each CPU tensor in a single conversion.

    This preserves the exact H21 v4 object/action/relation layout of
    collate_h21. Keep the reference path as the production default
    until paired, exact-tensor CPU and actor timing accepts the candidate.
    """
    from sts2_ai.models.shared_game_objects import NUMERIC_FIELDS

    if not decisions or any(item.schema != H21_OBJECT_SCHEMA for item in decisions):
        raise ValueError("Need nonempty same-version H7 decisions")
    if any(not item.objects or not item.actions for item in decisions):
        raise ValueError("H7 decisions require objects and legal actions")

    max_objects = max(len(item.objects) for item in decisions)
    max_actions = max(len(item.actions) for item in decisions)
    max_members = max(
        1, max((len(action.members) for item in decisions
                for action in item.actions), default=0),
    )
    max_edges = max(1, *(len(item.relations) for item in decisions))
    nfeatures = len(NUMERIC_FIELDS)

    names = (
        "kinds", "identities", "zones", "numeric", "observed", "owners",
        "object_mask", "action_kinds", "action_semantics", "sources",
        "targets", "action_mask", "phases", "action_members",
        "relation_types", "relation_sources", "relation_targets", "relation_mask",
    )
    rows: dict[str, list[Any]] = {name: [] for name in names}
    for item in decisions:
        objects, actions, relations = item.objects, item.actions, item.relations
        num_objects = len(objects)
        object_padding = max_objects - num_objects
        action_padding = max_actions - len(actions)
        edge_padding = max_edges - len(relations)

        phase = 0 if item.phase == "combat" else 1 if item.phase == "macro" else -1
        if phase < 0:
            raise ValueError("Unrecognized H7 decision phase")
        rows["phases"].append(phase)

        for token in objects:
            if token.owner >= num_objects:
                raise ValueError("Dangling public object owner")
        rows["kinds"].append(
            [t.kind for t in objects] + [0] * object_padding)
        rows["identities"].append(
            [t.identity for t in objects] + [0] * object_padding)
        rows["zones"].append(
            [t.zone for t in objects] + [0] * object_padding)
        rows["numeric"].append(
            [t.numeric for t in objects] + [(0.0,) * nfeatures] * object_padding)
        rows["observed"].append(
            [t.observed for t in objects] + [(0,) * nfeatures] * object_padding)
        rows["owners"].append(
            [t.owner for t in objects] + [-1] * object_padding)
        rows["object_mask"].append(
            [True] * num_objects + [False] * object_padding)

        for action in actions:
            for ptr in (action.source, action.target):
                if ptr >= num_objects or ptr < -1:
                    raise ValueError("Dangling action pointer")
            for member in action.members:
                if member < 0 or member >= num_objects:
                    raise ValueError("Dangling selected-card pointer")
        rows["action_kinds"].append(
            [a.kind for a in actions] + [0] * action_padding)
        rows["action_semantics"].append(
            [a.semantic for a in actions] + [0] * action_padding)
        rows["sources"].append(
            [a.source for a in actions] + [-1] * action_padding)
        rows["targets"].append(
            [a.target for a in actions] + [-1] * action_padding)
        rows["action_mask"].append(
            [True] * len(actions) + [False] * action_padding)
        rows["action_members"].append(
            [list(a.members) + [-1] * (max_members - len(a.members))
             for a in actions]
            + [[-1] * max_members for _ in range(action_padding)])

        for edge in relations:
            if (not 0 <= edge.relation_type < H21_RELATION_TYPES
                    or not 0 <= edge.source < num_objects
                    or not 0 <= edge.target < num_objects):
                raise ValueError("Invalid or dangling public H21 relation")
        rows["relation_types"].append(
            [e.relation_type for e in relations] + [0] * edge_padding)
        rows["relation_sources"].append(
            [e.source for e in relations] + [-1] * edge_padding)
        rows["relation_targets"].append(
            [e.target for e in relations] + [-1] * edge_padding)
        rows["relation_mask"].append(
            [True] * len(relations) + [False] * edge_padding)

    float_fields = {"numeric", "observed"}
    boolean_fields = {"object_mask", "action_mask", "relation_mask"}
    tensors = {
        name: torch.tensor(
            rows[name],
            dtype=(torch.float32 if name in float_fields
                   else torch.bool if name in boolean_fields else torch.long),
        )
        for name in names
    }
    return H20Batch(**tensors)


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
