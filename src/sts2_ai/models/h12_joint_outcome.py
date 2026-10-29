"""H12 policy-conditioned finite-mixture model of joint public combat outcomes.

The mixture index is shared by survival, exit HP, named held-potion
retention and persistent-resource change heads. This represents their
correlation without treating independent marginal predictions as a joint
distribution. Newly generated *identities* are kept in the observation
corpus and are explicitly outside this first generative model.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HASH_DIM = 512
DENSE_DIM = 18
FEATURE_DIM = HASH_DIM + DENSE_DIM
POTION_BUCKETS = 4096
MAX_POTIONS = 12
MIXTURE_COMPONENTS = 8
MODEL_SCHEMA = "sts2-h12-joint-outcome-mixture-v1"


def _digest(text: str, mod: int) -> int:
    return int.from_bytes(hashlib.blake2b(
        text.encode("utf-8"), digest_size=8,
    ).digest(), "little") % mod


def _ids(rows: Any, key: str) -> list[str]:
    if not isinstance(rows, list):
        return []
    return [row[key] for row in rows
            if isinstance(row, dict) and isinstance(row.get(key), str)]


def public_features(state: dict[str, Any]) -> Tensor:
    """Stable features from allowlisted H12 public_resources frames only."""
    values = torch.zeros(FEATURE_DIM, dtype=torch.float32)

    def number(key: str, scale: float) -> float:
        value = state.get(key)
        return max(-4., min(4., float(value) / scale)) if type(value) in (int, float) else 0.

    deck = state.get("deck")
    deck = deck if isinstance(deck, list) else []
    relics = state.get("relics")
    relics = relics if isinstance(relics, list) else []
    potions = state.get("potions")
    potions = potions if isinstance(potions, list) else []
    combat = state.get("combat")
    combat = combat if isinstance(combat, dict) else {}
    enemies = combat.get("enemies")
    enemies = enemies if isinstance(enemies, list) else []
    hand = combat.get("hand")
    hand = hand if isinstance(hand, list) else []
    hp = number("hp", 100.)
    max_hp = number("max_hp", 100.)
    dense = [
        hp, max_hp, hp / max(max_hp, 0.001),
        number("gold", 200.), number("floor", 16.),
        number("act", 3.),
        len(deck) / 30., len(relics) / 10.,
        len(potions) / 5., len(enemies) / 4.,
        len(hand) / 10., sum(
            max(0, int(e.get("hp", 0))) for e in enemies
            if isinstance(e, dict) and type(e.get("hp")) is int
        ) / 200.,
        float(combat.get("turn", 0)) / 10.
            if type(combat.get("turn")) is int else 0.,
        float(combat.get("energy", 0)) / 5.
            if type(combat.get("energy")) is int else 0.,
        float(combat.get("player_block", 0)) / 40.
            if type(combat.get("player_block")) is int else 0.,
        float(combat.get("draw_pile_count", 0)) / 30.
            if type(combat.get("draw_pile_count")) is int else 0.,
        sum(type(e.get("intent_damage")) is int for e in enemies
            if isinstance(e, dict)) / 4.,
        sum(int(e.get("intent_damage") or 0)
            * max(1, int(e.get("intent_hits") or 1))
            for e in enemies if isinstance(e, dict)
            and type(e.get("intent_damage")) is int) / 100.,
    ]
    values[HASH_DIM:] = torch.tensor(dense, dtype=torch.float32)
    terms: list[str] = []
    for card in deck:
        if isinstance(card, dict) and isinstance(card.get("card_id"), str):
            terms.append(f"deck:{card['card_id']}:u{card.get('upgrade_level', 0)}")
    for card in hand:
        if isinstance(card, dict) and isinstance(card.get("card_id"), str):
            terms.append(f"hand:{card['card_id']}:u{card.get('upgrade_level', 0)}")
    terms.extend(f"relic:{r}" for r in relics if isinstance(r, str))
    terms.extend(f"potion:{r}" for r in _ids(potions, "potion_id"))
    for room_key, count in (state.get("public_map_future_room_counts") or {}).items():
        if isinstance(room_key, str) and type(count) is int:
            terms.extend([f"visible-map:{room_key}"] * min(8, max(0, count)))
    for enemy in enemies:
        if not isinstance(enemy, dict):
            continue
        eid = enemy.get("enemy_id")
        if isinstance(eid, str):
            terms.append(f"enemy:{eid}")
            move = enemy.get("move_id")
            if isinstance(move, str):
                terms.append(f"intent:{eid}:{move}")
        for power in enemy.get("powers", []):
            if isinstance(power, dict) and isinstance(power.get("power_id"), str):
                terms.append(f"enemy-power:{power['power_id']}")
    for power in combat.get("player_powers", []):
        if isinstance(power, dict) and isinstance(power.get("power_id"), str):
            terms.append(f"player-power:{power['power_id']}")
    for term in terms:
        values[_digest(term, HASH_DIM)] += 1.0
    values[:HASH_DIM] /= math.sqrt(max(1, len(terms)))
    return values


def _potion_slots(entry: dict[str, Any], after: dict[str, Any]
                  ) -> tuple[Tensor, Tensor, Tensor]:
    before_names = _ids(entry.get("potions"), "potion_id")
    after_counts = Counter(_ids(after.get("potions"), "potion_id"))
    if len(before_names) > MAX_POTIONS:
        raise ValueError("H12 potion capacity exceeded; no silent truncation")
    ids = torch.zeros(MAX_POTIONS, dtype=torch.long)
    mask = torch.zeros(MAX_POTIONS, dtype=torch.float32)
    retained = torch.zeros(MAX_POTIONS, dtype=torch.float32)
    for i, name in enumerate(before_names):
        ids[i] = 1 + _digest("potion:" + name, POTION_BUCKETS - 1)
        mask[i] = 1.
        if after_counts[name] > 0:
            retained[i] = 1.
            after_counts[name] -= 1
    return ids, mask, retained


def training_example(combat: dict[str, Any], *,
                     resource_exit: dict[str, Any] | None = None
                     ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    entry = combat["entry"]
    # Survival and HP always describe the immediate combat exit. Other
    # resources may be measured after the frozen policy settles room rewards.
    after = combat["exit"]
    resources = resource_exit if resource_exit is not None else after
    if not isinstance(entry.get("combat"), dict):
        raise ValueError("H12 outcome training requires the combat-entry observation")
    max_hp = entry.get("max_hp")
    exit_hp = after.get("hp")
    if type(max_hp) is not int or max_hp <= 0 or type(exit_hp) is not int:
        raise ValueError("Combat outcome has no usable public HP")
    survived = combat.get("survived")
    if type(survived) is not bool:
        raise ValueError("Combat lacks an observed survival label")
    if survived != (exit_hp > 0):
        raise ValueError("Combat survival and exit HP disagree")
    pid, mask, retained = _potion_slots(entry, resources)
    gold_change = resources["gold"] - entry["gold"]
    outcome = torch.tensor([
        float(survived),
        max(0., min(1., exit_hp / max_hp)),
        math.tanh(gold_change / 100.),
        float(entry["deck"] != resources["deck"]),
        float(entry["relics"] != resources["relics"]),
        float(bool((Counter(_ids(resources.get("potions"), "potion_id"))
                    - Counter(_ids(entry.get("potions"), "potion_id"))))),
    ], dtype=torch.float32)
    return public_features(entry), pid, mask, retained, outcome


class H12JointOutcomeMixture(nn.Module):
    def __init__(self, components: int = MIXTURE_COMPONENTS) -> None:
        super().__init__()
        if components < 2:
            raise ValueError("H12 requires a nondegenerate outcome mixture")
        self.components = components
        self.encoder = nn.Sequential(
            nn.Linear(FEATURE_DIM, 384), nn.GELU(),
            nn.LayerNorm(384), nn.Linear(384, 256), nn.GELU(),
        )
        self.mixture = nn.Linear(256, components)
        # survival, exit-HP mean/scale, gold-delta mean, deck-change,
        # relic-change, new-potion-event
        self.outcome_head = nn.Linear(256, components * 7)
        self.potion_embedding = nn.Embedding(POTION_BUCKETS, 64, padding_idx=0)
        self.potion_head = nn.Sequential(
            nn.Linear(256 + 64, 128), nn.GELU(),
            nn.Linear(128, components),
        )

    def forward(self, features: Tensor, potion_ids: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:
        context = self.encoder(features)
        mixture = self.mixture(context).log_softmax(dim=-1)
        outcomes = self.outcome_head(context).view(
            -1, self.components, 7,
        )
        potions = self.potion_embedding(potion_ids)
        joint = torch.cat([
            context[:, None, :].expand(-1, potion_ids.shape[1], -1), potions,
        ], dim=-1)
        retention = self.potion_head(joint).transpose(1, 2)
        return mixture, outcomes, retention

    @staticmethod
    def _log_bernoulli(logits: Tensor, target: Tensor) -> Tensor:
        return -F.binary_cross_entropy_with_logits(
            logits, target.expand_as(logits), reduction="none",
        )

    def log_probability(self, features: Tensor, potion_ids: Tensor,
                        potion_mask: Tensor, potion_retained: Tensor,
                        outcome: Tensor) -> Tensor:
        weights, params, retention = self(features, potion_ids)
        survived = self._log_bernoulli(params[:, :, 0], outcome[:, 0:1])
        hp_mu = torch.sigmoid(params[:, :, 1])
        hp_sigma = F.softplus(params[:, :, 2]) * 0.2 + 0.035
        hp = torch.distributions.Normal(hp_mu, hp_sigma).log_prob(
            outcome[:, 1:2],
        )
        gold_mu = torch.tanh(params[:, :, 3])
        gold = torch.distributions.Normal(gold_mu, 0.18).log_prob(
            outcome[:, 2:3],
        )
        deck = self._log_bernoulli(params[:, :, 4], outcome[:, 3:4])
        relic = self._log_bernoulli(params[:, :, 5], outcome[:, 4:5])
        gained = self._log_bernoulli(params[:, :, 6], outcome[:, 5:6])
        potion_logp = self._log_bernoulli(
            retention, potion_retained[:, None, :],
        )
        potion_logp = (potion_logp * potion_mask[:, None, :]).sum(dim=-1)
        return torch.logsumexp(
            weights + survived + hp + gold + deck + relic + gained + potion_logp,
            dim=-1,
        )

    @torch.no_grad()
    def predict(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Expose a shared-latent distribution, rather than independent heads."""
        self.eval()
        before = entry.get("potions") or []
        # No exit observation at inference; the names alone define its queries.
        ids, mask, _ = _potion_slots(entry, entry)
        device = next(self.parameters()).device
        weights, params, retention = self(
            public_features(entry).to(device)[None, :],
            ids.to(device)[None, :],
        )
        component_probs = weights[0].exp()
        survived = torch.sigmoid(params[0, :, 0])
        hp = torch.sigmoid(params[0, :, 1])
        named = torch.sigmoid(retention[0])
        result = {
            "schema": MODEL_SCHEMA,
            "mixture_weights": component_probs.cpu().tolist(),
            "survival_by_component": survived.cpu().tolist(),
            "exit_hp_fraction_by_component": hp.cpu().tolist(),
            "survival_probability": float((component_probs * survived).sum()),
            "expected_exit_hp_fraction": float((component_probs * hp).sum()),
            "expected_gold_delta_tanh": float((
                component_probs * params[0, :, 3].tanh()
            ).sum()),
            "deck_change_probability": float((
                component_probs * torch.sigmoid(params[0, :, 4])
            ).sum()),
            "relic_change_probability": float((
                component_probs * torch.sigmoid(params[0, :, 5])
            ).sum()),
            "new_potion_event_probability": float((
                component_probs * torch.sigmoid(params[0, :, 6])
            ).sum()),
            "named_potion_retention": [
                {"potion_id": item["potion_id"],
                 "slot": item["slot"],
                 "probability": float((component_probs * named[:, i]).sum())}
                for i, item in enumerate(before) if i < MAX_POTIONS
            ],
        }
        return result


class H12MacroContinuation(nn.Module):
    """Public-state continuation estimate under the pinned rollout policy."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_DIM, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x).squeeze(-1)
