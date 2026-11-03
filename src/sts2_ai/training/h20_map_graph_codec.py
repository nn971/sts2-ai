"""H20 map-node graph and grounded route-choice semantic codec.

Map NodeIds are ephemeral reference keys. Learned node embeddings use room
type, floor and act attributes, while exact NextNodeIds become typed sparse
edges. Every choose_map_node action points at its true map object.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from sts2_ai.models.shared_game_objects import (
    IDENTITY_BUCKETS,
    KINDS,
    OWNER_ATTRIBUTE_KIND_V2,
    ZONES,
    ActionToken,
    EncodedDecision,
    ObjectToken,
    RelationToken,
    _bucket,
    _numbers,
)
from sts2_ai.training.h19_resource_codec import encode_h19_resource_decision

H20_OBJECT_SCHEMA = "sts2-h20-public-relational-map-v1"
H20_MAP_CODEC = "h20-map-graph-actions-v1"
NEXT_EDGE = 0
CURRENT_EDGE = 1
REACHABLE_EDGE = 2
H20_RELATION_TYPES = 3
NODE_KIND = KINDS.index("room")
MAP_ZONE = ZONES.index("map")


def encode_h20_map_decision(
    state: dict[str, Any], actions: Sequence[Any], *,
    public_schema_id: str,
    allow_affliction_source_relations: bool = False,
) -> EncodedDecision:
    base = encode_h19_resource_decision(
        state, actions, public_schema_id=public_schema_id,
        allow_affliction_source_relations=allow_affliction_source_relations,
    )
    nodes = state.get("map")
    if not isinstance(nodes, list):
        raise ValueError("H20 requires a visible map node list")
    node_positions = [idx for idx, obj in enumerate(base.objects)
                      if obj.kind == NODE_KIND and obj.zone == MAP_ZONE]
    if len(node_positions) != len(nodes):
        raise ValueError("H20 map object count differs from public map")

    objects = list(base.objects)
    node_by_id: dict[str, int] = {}
    for idx, node in zip(node_positions, nodes, strict=True):
        if (not isinstance(node, dict)
                or not isinstance(node.get("node_id"), str)
                or not isinstance(node.get("next_node_ids"), list)
                or type(node.get("floor")) is not int
                or type(node.get("act")) is not int
                or not isinstance(node.get("room_type"), (int, str))):
            raise ValueError("Malformed visible H20 map node")
        node_id = node["node_id"]
        if node_id in node_by_id:
            raise ValueError("Duplicate public NodeId")
        node_by_id[node_id] = idx
        # Override H7's string-only _entry_id: native room types may be enums.
        # Raw NodeIds never appear in either the learned identity or numerics.
        numeric, observed = _numbers({"floor": node["floor"]})
        objects[idx] = ObjectToken(
            NODE_KIND,
            _bucket("room_type:" + str(node["room_type"]), IDENTITY_BUCKETS),
            MAP_ZONE, numeric, observed,
        )
        for field, value in (("act", node["act"]),):
            num, mask = _numbers({"stacks": value})
            objects.append(ObjectToken(
                OWNER_ATTRIBUTE_KIND_V2,
                _bucket("map_node." + field, IDENTITY_BUCKETS),
                MAP_ZONE, num, mask, owner=idx,
            ))
    relations = []
    for node in nodes:
        source = node_by_id[node["node_id"]]
        next_ids = node["next_node_ids"]
        if len(next_ids) != len(set(next_ids)):
            raise ValueError("Duplicate NextNodeIds on one map node")
        for next_id in next_ids:
            if not isinstance(next_id, str) or next_id not in node_by_id:
                raise ValueError("Map NextNodeIds references a missing visible node")
            relations.append(RelationToken(NEXT_EDGE, source, node_by_id[next_id]))

    current = state.get("current_map_node_id")
    if current is not None:
        if not isinstance(current, str) or current not in node_by_id:
            raise ValueError("Unknown public current_map_node_id")
        relations.append(RelationToken(CURRENT_EDGE, 0, node_by_id[current]))
    current_idx = -1 if current is None else node_by_id[current]

    action_tokens: list[ActionToken] = []
    for action, old in zip(actions, base.actions, strict=True):
        if action.kind != "choose_map_node":
            action_tokens.append(old)
            continue
        try:
            payload = json.loads(action.payload_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed map action payload") from exc
        node_id = (payload.get("NodeId", payload.get("node_id"))
                   if isinstance(payload, dict) else None)
        if not isinstance(node_id, str) or node_id not in node_by_id:
            raise ValueError("Unbound choose_map_node target")
        target = node_by_id[node_id]
        # The target's structured object/edges now supply the semantics.
        # Payload hash is no longer the sole action feature.
        action_tokens.append(replace(
            old,
            semantic=_bucket("choose_map_node", IDENTITY_BUCKETS),
            source=current_idx, target=target,
        ))
        relations.append(RelationToken(REACHABLE_EDGE, 0, target))
    return EncodedDecision(
        H20_OBJECT_SCHEMA, base.phase, tuple(objects),
        tuple(action_tokens), tuple(relations),
    )
