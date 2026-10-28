"""Opt-in packed collation for inference-only shared H7/H8 actors.

This encodes the exact same public object/action tensors as collate_h7, but
avoids constructing two tiny Torch tensors *per object* and consolidates
the host-to-device transfers into seven packed buffers. It does not mutate
source EncodedDecision tokens, change action order, or touch model weights.

Use the reference collate_h7 in PPO/backprop; this path is only for the
frozen remote actor service and must pass field-by-field parity.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .shared_game_objects import NUMERIC_FIELDS, SHARED_OBJECT_SCHEMA, EncodedDecision
from .shared_transformer import H7Batch

# The only fields packed in an interleaved last dimension. Numeric fields,
# presence masks, action masks and phases already have contiguous storage.
H7_PACKED_METADATA_FIELDS = (
    "kinds", "identities", "zones", "owners",
    "action_kinds", "action_semantics", "sources", "targets",
)


def contiguous_actor_metadata(batch: H7Batch) -> H7Batch:
    """Materialize only the eight strided metadata views.

    Called *after* the seven packed buffers have been transferred to the
    target device. This deliberately leaves host packing, source input, model
    weights, batch scheduling, tensor values and other strides unchanged,
    isolating the cost or benefit of contiguous metadata for inference.
    """
    return H7Batch(**{
        name: (getattr(batch, name).contiguous()
               if name in H7_PACKED_METADATA_FIELDS else getattr(batch, name))
        for name in H7Batch.__dataclass_fields__
    })


@dataclass(frozen=True)
class H7PackedHost:
    """The seven CPU buffers stay interleaved until a batched device transfer."""

    object_meta: torch.Tensor
    action_meta: torch.Tensor
    numeric: torch.Tensor
    observed: torch.Tensor
    object_mask: torch.Tensor
    action_mask: torch.Tensor
    phases: torch.Tensor

    def to(
        self, device: str | torch.device, *,
        contiguous_metadata: bool = False,
    ) -> H7Batch:
        obj = self.object_meta.to(device)
        act = self.action_meta.to(device)
        numeric = self.numeric.to(device)
        observed = self.observed.to(device)
        object_mask = self.object_mask.to(device)
        action_mask = self.action_mask.to(device)
        phases = self.phases.to(device)
        result = H7Batch(
            kinds=obj[..., 0],
            identities=obj[..., 1],
            zones=obj[..., 2],
            numeric=numeric,
            observed=observed,
            owners=obj[..., 3],
            object_mask=object_mask,
            action_kinds=act[..., 0],
            action_semantics=act[..., 1],
            sources=act[..., 2],
            targets=act[..., 3],
            action_mask=action_mask,
            phases=phases,
        )
        return contiguous_actor_metadata(result) if contiguous_metadata else result


def prepare_h7_actor_packed(decisions: list[EncodedDecision]) -> H7PackedHost:
    """Build packed CPU tensors, independently of a GPU execution stream."""
    if not decisions or any(d.schema != SHARED_OBJECT_SCHEMA for d in decisions):
        raise ValueError("Need nonempty same-version H7 decisions")
    if any(not d.objects or not d.actions for d in decisions):
        raise ValueError("H7 decisions require objects and legal actions")
    if any(d.objects[0].kind != 0 for d in decisions):
        raise ValueError("H7 requires the global state token at position zero")

    nobjects = max(len(d.objects) for d in decisions)
    nactions = max(len(d.actions) for d in decisions)
    nfeatures = len(NUMERIC_FIELDS)
    zero_features = (0.0,) * nfeatures
    zero_observed = (0,) * nfeatures
    # Packed layout: (kind, identity, zone, owner).
    object_meta: list[list[tuple[int, int, int, int]]] = []
    object_numeric: list[list[tuple[float, ...]]] = []
    object_observed: list[list[tuple[int, ...]]] = []
    object_masks: list[list[bool]] = []
    # Packed layout: (action kind, semantic, source, target).
    action_meta: list[list[tuple[int, int, int, int]]] = []
    action_masks: list[list[bool]] = []
    phases: list[int] = []
    for d in decisions:
        nobj = len(d.objects)
        nact = len(d.actions)
        phase = {"combat": 0, "macro": 1}.get(d.phase, -1)
        if phase < 0:
            raise ValueError("Unrecognized H7 decision phase")
        phases.append(phase)
        meta_row = []
        numeric_row = []
        observed_row = []
        for token in d.objects:
            if token.owner >= nobj:
                raise ValueError("Dangling public object owner")
            if (len(token.numeric) != nfeatures or
                    len(token.observed) != nfeatures):
                raise ValueError("Wrong H7 numeric field count")
            meta_row.append((token.kind, token.identity, token.zone, token.owner))
            numeric_row.append(token.numeric)
            observed_row.append(token.observed)
        padding = nobjects - nobj
        meta_row.extend([(0, 0, 0, -1)] * padding)
        numeric_row.extend([zero_features] * padding)
        observed_row.extend([zero_observed] * padding)
        object_meta.append(meta_row)
        object_numeric.append(numeric_row)
        object_observed.append(observed_row)
        object_masks.append([True] * nobj + [False] * padding)
        action_row = []
        for action in d.actions:
            for ptr in (action.source, action.target):
                if ptr >= nobj or ptr < -1:
                    raise ValueError("Dangling action pointer")
            action_row.append((
                action.kind, action.semantic, action.source, action.target))
        pad_actions = nactions - nact
        action_row.extend([(0, 0, -1, -1)] * pad_actions)
        action_meta.append(action_row)
        action_masks.append([True] * nact + [False] * pad_actions)

    # One copy each for object/action metadata (rather than many small
    # independent .to(device) calls), plus two float arrays and masks.
    return H7PackedHost(
        object_meta=torch.tensor(object_meta, dtype=torch.long),
        action_meta=torch.tensor(action_meta, dtype=torch.long),
        numeric=torch.tensor(object_numeric, dtype=torch.float32),
        observed=torch.tensor(object_observed, dtype=torch.float32),
        object_mask=torch.tensor(object_masks, dtype=torch.bool),
        action_mask=torch.tensor(action_masks, dtype=torch.bool),
        phases=torch.tensor(phases, dtype=torch.long),
    )


def collate_h7_actor_packed(
    decisions: list[EncodedDecision], device: str | torch.device = "cpu",
    *, contiguous_metadata: bool = False,
) -> H7Batch:
    return prepare_h7_actor_packed(decisions).to(
        device, contiguous_metadata=contiguous_metadata)
