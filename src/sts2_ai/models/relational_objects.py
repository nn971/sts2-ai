"""v15d: ownership-aware two-layer multihead object/action attention.

The frozen v8/v13 combat head is unchanged. Both gates start at zero, so
initial policy/value predictions are *exactly* the legacy model's outputs.

Two four-head object-attention blocks have explicit directed owner/effect
relation biases. Action cross-attention retains source/target/selection roles,
and count features avoid mean-pooling's complete loss of multiplicity.
Pure-Python portable inference implements the same operators as Torch.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.models.object_embeddings import ObjectVocabulary, TypedObjectEncoder
from sts2_ai.models.object_instances import (
    ActionBinding,
    ObjectRef,
    ObjectSet,
    extract_combat_objects,
    legal_action_object_bindings,
)
from sts2_ai.models.pooled_objects import (
    PortableTypedEncoder,
    _linear,
    _mean,
    _silu,
    _tensor_payload,
)
from sts2_ai.models.protocol import PolicyValueEstimate

RELATIONAL_OBJECT_FORMAT = "sts2-neural-policy-value-v15d-relational-object-residual"
ACTION_KINDS = ("<UNK>", "play_card", "use_potion", "select_cards", "end_turn")
ACTION_ROLES = ("source_card", "source_potion", "target_enemy", "selected_card")
HEADS = 4
LAYERS = 2
RELATIONS = 5  # unrelated, self, effect->owner, owner->effect, siblings


def _relations(objects: ObjectSet) -> list[list[int]]:
    """Directed relation IDs; integer instance values are never learned."""
    tokens = objects.tokens
    out = [[0] * len(tokens) for _ in tokens]
    for i, src in enumerate(tokens):
        for j, dst in enumerate(tokens):
            if i == j:
                relation = 1
            elif src.owner is not None and src.owner == dst.ref:
                relation = 2
            elif dst.owner is not None and dst.owner == src.ref:
                relation = 3
            elif src.owner is not None and src.owner == dst.owner:
                relation = 4
            else:
                relation = 0
            out[i][j] = relation
    return out


def _action_tokens(
    actions: Sequence[LegalAction], objects: ObjectSet,
) -> tuple[tuple[str, tuple[ActionBinding, ...]], ...]:
    return tuple(
        (a.kind, legal_action_object_bindings(a.kind, a.payload_json, objects))
        for a in actions
    )


def _softmax(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("Attention requires at least one key")
    m = max(values)
    ex = [math.exp(x - m) for x in values]
    total = sum(ex)
    return [v / total for v in ex]


def _norm(x: list[float], scale: list[float], shift: list[float]) -> list[float]:
    mu = sum(x) / len(x)
    inv = 1.0 / math.sqrt(sum((v - mu) ** 2 for v in x) / len(x) + 1e-5)
    return [(v - mu) * inv * scale[i] + shift[i] for i, v in enumerate(x)]


def _plus(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


class RelationalObjectTorch:
    """Differentiable relation-aware residual with portable JSON export."""

    def __init__(self, vocabulary: ObjectVocabulary, *, width: int = 64) -> None:
        if width <= 0 or width % HEADS:
            raise ValueError("v15d width must be positive and divisible by 4")
        import torch
        from torch import nn

        self.torch = torch
        self.width = width
        self.vocabulary = vocabulary
        self.embedding = TypedObjectEncoder(vocabulary, width=width)

        class Block(nn.Module):
            def __init__(self, d: int) -> None:
                super().__init__()
                self.q = nn.Linear(d, d)
                self.k = nn.Linear(d, d)
                self.v = nn.Linear(d, d)
                self.out = nn.Linear(d, d)
                self.bias = nn.Parameter(torch.zeros(HEADS, RELATIONS))
                self.norm1 = nn.LayerNorm(d)
                self.ff1 = nn.Linear(d, d)
                self.ff2 = nn.Linear(d, d)
                self.norm2 = nn.LayerNorm(d)

            def forward(self, x: Any, rel: Any) -> Any:
                n, d = x.shape
                width = d // HEADS
                q = self.q(x).reshape(n, HEADS, width).transpose(0, 1)
                k = self.k(x).reshape(n, HEADS, width).transpose(0, 1)
                v = self.v(x).reshape(n, HEADS, width).transpose(0, 1)
                scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(width)
                scores = scores + self.bias[:, rel]
                attention = scores.softmax(dim=-1)
                a = torch.matmul(attention, v)
                a = a.transpose(0, 1).reshape(n, d)
                y = self.norm1(x + self.out(a))
                return self.norm2(y + self.ff2(torch.nn.functional.silu(self.ff1(y))))

        class Network(nn.Module):
            def __init__(self, encoder: Any, d: int) -> None:
                super().__init__()
                self.encoder = encoder
                self.blocks = nn.ModuleList([Block(d) for _ in range(LAYERS)])
                self.global_count = nn.Parameter(torch.zeros(d))
                self.state_proj = nn.Linear(d, d)
                self.action_kinds = nn.Embedding(len(ACTION_KINDS), d)
                self.action_roles = nn.Embedding(len(ACTION_ROLES), d)
                self.action_count = nn.Parameter(torch.zeros(d))
                self.cross_q = nn.Linear(d, d)
                self.cross_k = nn.Linear(d, d)
                self.cross_v = nn.Linear(d, d)
                self.cross_out = nn.Linear(d, d)
                self.empty_action = nn.Parameter(torch.zeros(d))
                self.policy_readout = nn.Linear(d, 1)
                self.value_readout = nn.Linear(d, 1)
                self.policy_gate = nn.Parameter(torch.zeros(()))
                self.value_gate = nn.Parameter(torch.zeros(()))

        self.module = Network(self.embedding.module, width)

    def forward_prepared(
        self, objects: ObjectSet,
        actions: Sequence[tuple[str, tuple[ActionBinding, ...]]],
        baseline: PolicyValueEstimate,
    ) -> tuple[Any, Any]:
        torch = self.torch
        x = self.embedding.encode(objects)
        device, dtype = x.device, x.dtype
        n, d = x.shape
        relation = torch.tensor(_relations(objects), dtype=torch.long, device=device)
        for block in self.module.blocks:
            x = block(x, relation)
        global_pos = objects.index().get(ObjectRef("global", "player"))
        if global_pos is None:
            raise ValueError("Missing player/global token")
        context = x.mean(dim=0) + x[global_pos]
        context = context + math.log1p(n) * self.module.global_count
        shared = torch.tanh(self.module.state_proj(context))
        initial_value = torch.tensor(baseline.value, dtype=dtype, device=device)
        correction = torch.tanh(self.module.value_readout(shared).squeeze())
        value = initial_value + (
            torch.tanh(self.module.value_gate) * correction
            * (1.0 - initial_value.abs())
        )
        initial_logits = torch.tensor(baseline.action_logits, dtype=dtype, device=device)
        if not actions:
            return value, initial_logits
        if len(actions) != len(baseline.action_logits):
            raise ValueError("v15d actions and baseline logits differ in length")
        index = objects.index()
        action_rows: list[Any] = []
        for kind, bindings in actions:
            kind_idx = ACTION_KINDS.index(kind) if kind in ACTION_KINDS else 0
            kind_embedding = self.module.action_kinds(
                torch.tensor(kind_idx, dtype=torch.long, device=device)
            )
            if bindings:
                keys = []
                for binding in bindings:
                    role = ACTION_ROLES.index(binding.role)
                    keys.append(x[index[binding.ref]] + self.module.action_roles.weight[role])
                subset = torch.stack(keys)
                hd = d // HEADS
                q = self.module.cross_q(shared + kind_embedding).reshape(HEADS, hd)
                k = self.module.cross_k(subset).reshape(-1, HEADS, hd).transpose(0, 1)
                v = self.module.cross_v(subset).reshape(-1, HEADS, hd).transpose(0, 1)
                weights = (
                    torch.einsum("hd,hmd->hm", q, k) / math.sqrt(hd)
                ).softmax(dim=-1)
                attended = torch.einsum("hm,hmd->hd", weights, v).reshape(d)
                cross = self.module.cross_out(attended)
            else:
                cross = self.module.empty_action
            merged = torch.tanh(
                shared + kind_embedding + cross
                + math.log1p(len(bindings)) * self.module.action_count
            )
            action_rows.append(self.module.policy_readout(merged).squeeze())
        delta = torch.stack(action_rows)
        return value, initial_logits + self.module.policy_gate * delta

    def forward_batch_prepared(
        self,
        objects: Sequence[ObjectSet],
        actions: Sequence[Sequence[tuple[str, tuple[ActionBinding, ...]]]],
        baselines: Sequence[PolicyValueEstimate],
        *,
        stage_hook: Callable[[str], None] | None = None,
    ) -> tuple[Any, tuple[Any, ...]]:
        """Batched padded object and action attention, without state leakage.

        This intentionally shares the v15d module parameters with the
        per-decision reference path. Padding is excluded from *key* softmax,
        player/global pooling and action cross-attention. Caller retains
        individual legal-action menus and their original ordering.
        """
        torch = self.torch
        b = len(objects)
        if len(actions) != b or len(baselines) != b:
            raise ValueError("v15f batch objects/actions/baselines mismatch")
        if not b:
            return (
                self.module.policy_gate.new_zeros((0,)),
                (),
            )
        sizes = [len(obj.tokens) for obj in objects]
        if any(n < 1 for n in sizes):
            raise ValueError("v15f requires at least the global object")
        embeddings = self.embedding.encode_many(objects)
        if stage_hook is not None:
            stage_hook("object_embedding")
        device, dtype = embeddings[0].device, embeddings[0].dtype
        nmax, d = max(sizes), self.width
        ids = torch.arange(nmax, device=device)
        valid = ids.unsqueeze(0) < torch.tensor(
            sizes, device=device, dtype=torch.long,
        ).unsqueeze(1)
        x = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        relation = torch.zeros((b, nmax, nmax), device=device, dtype=torch.long)
        global_indices: list[int] = []
        indices = []
        for i, sample in enumerate(objects):
            index = sample.index()
            player = index.get(ObjectRef("global", "player"))
            if player is None:
                raise ValueError("Missing player/global token")
            indices.append(index)
            global_indices.append(player)
            n = sizes[i]
            relation[i, :n, :n] = torch.tensor(
                _relations(sample), device=device, dtype=torch.long,
            )
        if stage_hook is not None:
            stage_hook("relation_setup")
        hd = d // HEADS
        for block in self.module.blocks:
            q = block.q(x).reshape(b, nmax, HEADS, hd).transpose(1, 2)
            k = block.k(x).reshape(b, nmax, HEADS, hd).transpose(1, 2)
            v = block.v(x).reshape(b, nmax, HEADS, hd).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
            bias = block.bias[:, relation].permute(1, 0, 2, 3)
            scores = (scores + bias).masked_fill(
                ~valid[:, None, None, :], torch.finfo(dtype).min,
            )
            weights = scores.softmax(dim=-1)
            attended = torch.matmul(weights, v).transpose(1, 2)
            attended = attended.reshape(b, nmax, d)
            y = block.norm1(x + block.out(attended))
            x = block.norm2(
                y + block.ff2(torch.nn.functional.silu(block.ff1(y)))
            )
        if stage_hook is not None:
            stage_hook("self_attention")
        lengths = torch.tensor(sizes, device=device, dtype=dtype)
        context = (
            (x * valid.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
            + x[torch.arange(b, device=device), torch.tensor(
                global_indices, device=device, dtype=torch.long,
            )]
        )
        context = context + lengths.log1p().unsqueeze(-1) * self.module.global_count
        shared = torch.tanh(self.module.state_proj(context))
        base_values = torch.tensor(
            [row.value for row in baselines], device=device, dtype=dtype,
        )
        correction = torch.tanh(
            self.module.value_readout(shared).squeeze(-1)
        )
        values = (
            base_values + torch.tanh(self.module.value_gate) * correction
            * (1.0 - base_values.abs())
        )
        if stage_hook is not None:
            stage_hook("value_head")
        owners: list[int] = []
        kind_ids: list[int] = []
        action_counts: list[int] = []
        edges: list[list[tuple[int, int]]] = []
        logits0: list[float] = []
        action_lengths: list[int] = []
        for i, (menu, baseline) in enumerate(zip(actions, baselines, strict=True)):
            if len(menu) != len(baseline.action_logits):
                raise ValueError("v15f action/logit menu mismatch")
            action_lengths.append(len(menu))
            for (kind, bindings), baseline_logit in zip(
                menu, baseline.action_logits, strict=True,
            ):
                owners.append(i)
                kind_ids.append(ACTION_KINDS.index(kind) if kind in ACTION_KINDS else 0)
                action_counts.append(len(bindings))
                edges.append([
                    (indices[i][binding.ref], ACTION_ROLES.index(binding.role))
                    for binding in bindings
                ])
                logits0.append(baseline_logit)
        a = len(owners)
        if stage_hook is not None:
            stage_hook("action_metadata")
        if not a:
            return values, tuple(values.new_zeros((0,)) for _ in range(b))
        owner_tensor = torch.tensor(owners, dtype=torch.long, device=device)
        kinds = self.module.action_kinds(torch.tensor(
            kind_ids, dtype=torch.long, device=device,
        ))
        local_shared = shared[owner_tensor]
        max_edges = max(action_counts)
        if max_edges:
            edge_indices = torch.tensor([
                [index for index, _ in row] + [0] * (max_edges - len(row))
                for row in edges
            ], dtype=torch.long, device=device)
            edge_roles = torch.tensor([
                [role for _, role in row] + [0] * (max_edges - len(row))
                for row in edges
            ], dtype=torch.long, device=device)
            edge_valid = (
                torch.arange(max_edges, device=device).unsqueeze(0)
                < torch.tensor(action_counts, device=device).unsqueeze(1)
            )
            subset = (
                x[owner_tensor[:, None], edge_indices]
                + self.module.action_roles(edge_roles)
            )
            queries = self.module.cross_q(
                local_shared + kinds
            ).reshape(a, HEADS, hd)
            keys = self.module.cross_k(subset).reshape(
                a, max_edges, HEADS, hd,
            ).transpose(1, 2)
            payload = self.module.cross_v(subset).reshape(
                a, max_edges, HEADS, hd,
            ).transpose(1, 2)
            scores = torch.einsum("ahd,ahmd->ahm", queries, keys) / math.sqrt(hd)
            scores = scores.masked_fill(
                ~edge_valid[:, None, :], torch.finfo(dtype).min,
            )
            attention = scores.softmax(dim=-1)
            attended = torch.einsum(
                "ahm,ahmd->ahd", attention, payload,
            ).reshape(a, d)
            cross = self.module.cross_out(attended)
            cross = torch.where(
                torch.tensor(
                    [count > 0 for count in action_counts], device=device,
                ).unsqueeze(-1),
                cross, self.module.empty_action.unsqueeze(0),
            )
        else:
            cross = self.module.empty_action.unsqueeze(0).expand(a, -1)
        if stage_hook is not None:
            stage_hook("action_cross_attention")
        counts = torch.tensor(action_counts, dtype=dtype, device=device)
        merged = torch.tanh(
            local_shared + kinds + cross
            + counts.log1p().unsqueeze(-1) * self.module.action_count
        )
        updates = self.module.policy_readout(merged).squeeze(-1)
        base_logits = torch.tensor(logits0, device=device, dtype=dtype)
        flat_logits = base_logits + self.module.policy_gate * updates
        if stage_hook is not None:
            stage_hook("policy_head")
        return values, flat_logits.split(action_lengths)

    def forward(
        self, base: NeuralPolicyValueModel, observation: Observation,
        actions: Sequence[LegalAction],
    ) -> tuple[Any, Any]:
        state = json.loads(observation.payload_json)
        if not isinstance(state, dict):
            raise ValueError("Invalid public combat observation")
        objects = extract_combat_objects(state)
        return self.forward_prepared(
            objects, _action_tokens(actions, objects),
            base.evaluate(observation, actions),
        )

    def export(self, base: NeuralPolicyValueModel) -> PortableRelationalObjectModel:
        return PortableRelationalObjectModel.from_dict({
            "format": RELATIONAL_OBJECT_FORMAT,
            "model_id": base.model_id + "-relational-v15d",
            "base": base.to_dict(), "vocabulary": self.vocabulary.to_dict(),
            "width": self.width,
            "weights": _tensor_payload({
                name: tensor.detach().cpu().tolist()
                for name, tensor in self.module.state_dict().items()
            }),
        })


@dataclass(slots=True)
class PortableRelationalObjectModel:
    model_id: str
    base: NeuralPolicyValueModel
    vocabulary: ObjectVocabulary
    width: int
    weights: dict[str, Any]
    format_id: str = RELATIONAL_OBJECT_FORMAT

    @property
    def dimension(self) -> int:
        return self.base.dimension

    @property
    def hidden(self) -> int:
        return self.base.hidden

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PortableRelationalObjectModel:
        if raw.get("format") != RELATIONAL_OBJECT_FORMAT:
            raise ValueError("Unknown relational-object checkpoint format")
        if (not isinstance(raw.get("width"), int) or raw["width"] <= 0
                or raw["width"] % HEADS or not isinstance(raw.get("model_id"), str)
                or not raw["model_id"] or not isinstance(raw.get("base"), dict)
                or not isinstance(raw.get("vocabulary"), dict)
                or not isinstance(raw.get("weights"), dict)):
            raise ValueError("Invalid relational-object checkpoint metadata")
        obj = cls(
            raw["model_id"], NeuralPolicyValueModel.from_dict(raw["base"]),
            ObjectVocabulary.from_dict(raw["vocabulary"]), raw["width"],
            _tensor_payload(raw["weights"]),
        )
        obj._validate()
        return obj

    def _validate(self) -> None:
        d = self.width
        shapes: dict[str, tuple[int, ...]] = {
            "encoder.kinds.weight": (len(self.vocabulary.kinds), d),
            "encoder.identities.weight": (len(self.vocabulary.identities), d),
            "encoder.roles.weight": (len(self.vocabulary.roles), d),
            "encoder.categories.weight": (len(self.vocabulary.categories), d),
            "encoder.numeric_fields.weight": (len(self.vocabulary.numeric_fields), d),
            "encoder.norm.weight": (d,),
            "encoder.norm.bias": (d,),
            "global_count": (d,),
            "state_proj.weight": (d, d),
            "state_proj.bias": (d,),
            "action_kinds.weight": (len(ACTION_KINDS), d),
            "action_roles.weight": (len(ACTION_ROLES), d),
            "action_count": (d,),
            "empty_action": (d,),
            "policy_readout.weight": (1, d),
            "policy_readout.bias": (1,),
            "value_readout.weight": (1, d),
            "value_readout.bias": (1,),
            "policy_gate": (),
            "value_gate": (),
        }
        for name in ("cross_q", "cross_k", "cross_v", "cross_out"):
            shapes[name + ".weight"] = (d, d)
            shapes[name + ".bias"] = (d,)
        for i in range(len(self.vocabulary.kinds)):
            name = "encoder.numeric_layers." + str(i)
            shapes[name + ".0.weight"] = (d, d + 1)
            shapes[name + ".0.bias"] = (d,)
            shapes[name + ".2.weight"] = (d, d)
            shapes[name + ".2.bias"] = (d,)
        for i in range(LAYERS):
            name = "blocks." + str(i)
            for sub in ("q", "k", "v", "out", "ff1", "ff2"):
                shapes[name + "." + sub + ".weight"] = (d, d)
                shapes[name + "." + sub + ".bias"] = (d,)
            shapes[name + ".bias"] = (HEADS, RELATIONS)
            for norm in ("norm1", "norm2"):
                shapes[name + "." + norm + ".weight"] = (d,)
                shapes[name + "." + norm + ".bias"] = (d,)
        if set(self.weights) != set(shapes):
            raise ValueError("Incomplete v15d checkpoint tensor set")

        def check(value: Any, shape: tuple[int, ...], key: str) -> None:
            if not shape:
                if not isinstance(value, float) or not math.isfinite(value):
                    raise ValueError("Invalid v15d scalar: " + key)
                return
            if not isinstance(value, list) or len(value) != shape[0]:
                raise ValueError("Invalid v15d tensor shape: " + key)
            for child in value:
                check(child, shape[1:], key)

        for key, shape in shapes.items():
            check(self.weights[key], shape, key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format_id, "model_id": self.model_id,
            "base": self.base.to_dict(), "vocabulary": self.vocabulary.to_dict(),
            "width": self.width, "weights": self.weights,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), sort_keys=True) + "\n")

    def _project(self, prefix: str, x: list[float]) -> list[float]:
        return _linear(
            self.weights[prefix + ".weight"], self.weights[prefix + ".bias"], x,
        )

    def _encode(self, objects: ObjectSet) -> list[list[float]]:
        w, d = self.weights, self.width
        enc = {key.removeprefix("encoder."): value for key, value in w.items()
               if key.startswith("encoder.")}
        x = PortableTypedEncoder(self.vocabulary, d, enc).encode(objects)
        rel = _relations(objects)
        hd = d // HEADS
        for level in range(LAYERS):
            p = "blocks." + str(level)
            q, k, v = [
                [self._project(p + "." + component, row) for row in x]
                for component in ("q", "k", "v")
            ]
            updates: list[list[float]] = []
            for i, source in enumerate(x):
                attended: list[float] = []
                for head in range(HEADS):
                    span = slice(head * hd, (head + 1) * hd)
                    scores = [
                        sum(a * b for a, b in zip(q[i][span], k[j][span], strict=True))
                        / math.sqrt(hd) + w[p + ".bias"][head][rel[i][j]]
                        for j in range(len(x))
                    ]
                    probability = _softmax(scores)
                    attended.extend(
                        sum(probability[j] * v[j][z] for j in range(len(x)))
                        for z in range(head * hd, (head + 1) * hd)
                    )
                y = _norm(
                    _plus(source, self._project(p + ".out", attended)),
                    w[p + ".norm1.weight"], w[p + ".norm1.bias"],
                )
                ff = self._project(p + ".ff1", y)
                y = _norm(
                    _plus(y, self._project(p + ".ff2", [_silu(a) for a in ff])),
                    w[p + ".norm2.weight"], w[p + ".norm2.bias"],
                )
                updates.append(y)
            x = updates
        return x

    def evaluate(
        self, observation: Observation, legal_actions: Sequence[LegalAction],
    ) -> PolicyValueEstimate:
        baseline = self.base.evaluate(observation, legal_actions)
        w = self.weights
        if w["policy_gate"] == 0.0 and w["value_gate"] == 0.0:
            return baseline
        state = json.loads(observation.payload_json)
        if not isinstance(state, dict):
            raise ValueError("Invalid public combat observation")
        objects = extract_combat_objects(state)
        actions = _action_tokens(legal_actions, objects)
        x = self._encode(objects)
        d, n = self.width, len(x)
        idx = objects.index()
        player = idx.get(ObjectRef("global", "player"))
        if player is None:
            raise ValueError("Missing player/global token")
        context = _plus(_mean(x, d), x[player])
        context = _plus(context, [math.log1p(n) * value for value in w["global_count"]])
        shared = [math.tanh(value) for value in self._project("state_proj", context)]
        value_delta = math.tanh(self._project("value_readout", shared)[0])
        value = (
            baseline.value + math.tanh(w["value_gate"]) * value_delta
            * (1.0 - abs(baseline.value))
        )
        logits: list[float] = []
        hd = d // HEADS
        for (kind, bindings), base_logit in zip(
            actions, baseline.action_logits, strict=True,
        ):
            kind_idx = (
                ACTION_KINDS.index(kind) if kind in ACTION_KINDS else 0
            )
            kind_vec = w["action_kinds.weight"][kind_idx]
            if bindings:
                subset = [
                    _plus(x[idx[binding.ref]],
                          w["action_roles.weight"][ACTION_ROLES.index(binding.role)])
                    for binding in bindings
                ]
                q = self._project("cross_q", _plus(shared, kind_vec))
                keys = [self._project("cross_k", row) for row in subset]
                values = [self._project("cross_v", row) for row in subset]
                attended = []
                for head in range(HEADS):
                    span = slice(head * hd, (head + 1) * hd)
                    scores = [
                        sum(a * b for a, b in zip(q[span], key[span], strict=True))
                        / math.sqrt(hd)
                        for key in keys
                    ]
                    prob = _softmax(scores)
                    attended.extend(
                        sum(prob[j] * values[j][z] for j in range(len(subset)))
                        for z in range(head * hd, (head + 1) * hd)
                    )
                cross = self._project("cross_out", attended)
            else:
                cross = w["empty_action"]
            merged = [
                math.tanh(a + b + c + math.log1p(len(bindings)) * count)
                for a, b, c, count in zip(
                    shared, kind_vec, cross, w["action_count"], strict=True,
                )
            ]
            logits.append(
                base_logit + w["policy_gate"]
                * self._project("policy_readout", merged)[0]
            )
        return PolicyValueEstimate(tuple(logits), value)
