"""H24 experimental hard resource-monotone continuation-value head.

For fixed resource-neutral context z, all coefficients on certified HP, gold,
potion-presence and relic-presence coordinates are nonnegative. The encoder
must separately certify that counterfactual states share exactly the same z.
This module is a tested primitive; H22 trainer wiring is not yet implemented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sts2_ai.training.h24_applicability import ResourceConstraintSelection


@dataclass(frozen=True)
class SafeResourceVocabulary:
    """Audited beneficial item identities; unknown items stay in context."""

    potion_ids: tuple[str, ...] = ()
    relic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for group in (self.potion_ids, self.relic_ids):
            if len(set(group)) != len(group) or any(not item for item in group):
                raise ValueError("Safe resource identities must be unique and nonempty")


class MonotoneResourceLogit(nn.Module):
    """Nondecreasing logit in certified resources, conditional on invariant z.

    L(z,x) = base(z) + softplus(w_h(z))*hp/100
                     + softplus(w_g(z))*log(1+gold)/log(501)
                     + sum_p softplus(w_p(z))*count_p
                     + sum_r softplus(w_r(z))*count_r.

    Sigmoid(L) inherits this order. Resource-dependent context would invalidate
    the guarantee, so the eventual trainer must use a neutralized encoder.
    """

    def __init__(
        self, context_width: int, *, potions: int = 0, relics: int = 0,
        hp: bool = True, gold: bool = True,
    ) -> None:
        super().__init__()
        if (context_width <= 0 or min(potions, relics) < 0
                or not (hp or gold or potions or relics)):
            raise ValueError("Invalid monotone head dimensions")
        self.context_width = context_width
        self.potions = potions
        self.relics = relics
        self.hp = hp
        self.gold = gold
        self.base = nn.Linear(context_width, 1)
        self.hp_weight = nn.Linear(context_width, 1) if hp else None
        self.gold_weight = nn.Linear(context_width, 1) if gold else None
        self.potion_weights = nn.Linear(context_width, potions) if potions else None
        self.relic_weights = nn.Linear(context_width, relics) if relics else None
        for layer in (self.hp_weight, self.gold_weight,
                      self.potion_weights, self.relic_weights):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, -4.0)

    @staticmethod
    def _column(
        value: Tensor | None, batch: int, width: int,
        device: torch.device, label: str,
    ) -> Tensor:
        if value is None:
            if width:
                raise ValueError(f"Missing certified {label} coordinates")
            return torch.empty((batch, 0), device=device)
        if value.shape != (batch, width) or value.device != device:
            raise ValueError(f"Invalid certified {label} coordinates")
        if not bool((torch.isfinite(value) & (value >= 0)).all()):
            raise ValueError(f"Invalid certified {label} values")
        return value

    def forward(
        self, neutral_context: Tensor, *, hp: Tensor | None = None,
        gold: Tensor | None = None, potions: Tensor | None = None,
        relics: Tensor | None = None,
    ) -> Tensor:
        if (neutral_context.ndim != 2
                or neutral_context.shape[1] != self.context_width):
            raise ValueError("Invalid resource-neutral context width")
        if not bool(torch.isfinite(neutral_context).all()):
            raise ValueError("Nonfinite resource-neutral context")
        batch = neutral_context.shape[0]
        device = neutral_context.device
        result = self.base(neutral_context).squeeze(-1)
        if self.hp_weight is not None:
            x = self._column(hp, batch, 1, device, "HP")
            result = result + (
                F.softplus(self.hp_weight(neutral_context)) * x / 100
            ).sum(-1)
        elif hp is not None:
            raise ValueError("HP is not configured")
        if self.gold_weight is not None:
            x = self._column(gold, batch, 1, device, "gold")
            result = result + (
                F.softplus(self.gold_weight(neutral_context))
                * torch.log1p(x) / math.log(501)
            ).sum(-1)
        elif gold is not None:
            raise ValueError("Gold is not configured")
        if self.potion_weights is not None:
            x = self._column(potions, batch, self.potions, device, "potion")
            result = result + (
                F.softplus(self.potion_weights(neutral_context)) * x
            ).sum(-1)
        elif potions is not None and potions.numel():
            raise ValueError("Potions are not configured")
        if self.relic_weights is not None:
            x = self._column(relics, batch, self.relics, device, "relic")
            result = result + (
                F.softplus(self.relic_weights(neutral_context)) * x
            ).sum(-1)
        elif relics is not None and relics.numel():
            raise ValueError("Relics are not configured")
        return result

def build_h24_monotone_head(
    context_width: int, *,
    selection: ResourceConstraintSelection | None = None,
    vocabulary: SafeResourceVocabulary | None = None,
) -> MonotoneResourceLogit:
    """Build only selected monotone resource coordinates.

    A vocabulary is an input list, not a mechanics-safety certificate.
    The eventual value encoder/trainer must also verify eligibility.
    By default HP and gold are present; potion coordinates require named
    items, and relic coordinates require explicit opt-in *and* named items.
    """
    chosen = selection if selection is not None else ResourceConstraintSelection()
    words = vocabulary if vocabulary is not None else SafeResourceVocabulary()
    if not isinstance(chosen, ResourceConstraintSelection):
        raise TypeError("Expected H24 resource constraint selection")
    if not isinstance(words, SafeResourceVocabulary):
        raise TypeError("Expected H24 certified-resource vocabulary")
    if words.relic_ids and not chosen.relic:
        raise ValueError("Relic monotonicity is disabled by default; opt in explicitly")
    if words.potion_ids and not chosen.potion:
        raise ValueError("Potion monotonicity is disabled by selection")
    return MonotoneResourceLogit(
        context_width, hp=chosen.hp, gold=chosen.gold,
        potions=len(words.potion_ids) if chosen.potion else 0,
        relics=len(words.relic_ids) if chosen.relic else 0,
    )
