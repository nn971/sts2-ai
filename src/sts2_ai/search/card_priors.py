"""Optional human-statistics priors for *offered* card-reward actions only.

No statistics are bundled or fetched automatically. Import a deliberately
versioned, authorized aggregate snapshot and keep its biases/provenance visible.
These priors are exploratory hints, never game-mechanics probabilities.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_ai.emulator import LegalAction, Observation
from sts2_ai.models.hashed_linear import semantic_action_label, state_dict

CARD_PRIOR_FORMAT = "sts2-offer-pick-prior-v1"
_REWARD_KINDS = frozenset(("take_reward_card", "skip_reward_card"))


@dataclass(frozen=True, slots=True)
class CardPickCount:
    card_id: str
    offered: int
    picked: int

    def __post_init__(self) -> None:
        if not self.card_id or self.card_id == "__skip__":
            raise ValueError("Card ID must be a genuine nonempty game identifier")
        if type(self.offered) is not int or type(self.picked) is not int:
            raise ValueError("Counts must be integers")
        if self.offered <= 0 or not 0 <= self.picked <= self.offered:
            raise ValueError("Require 0 <= picked <= offered and offered > 0")


@dataclass(frozen=True, slots=True)
class CardPriorContext:
    game_build: str
    character: str
    act: int
    ascension: int


@dataclass(frozen=True, slots=True)
class CardPriorDataset:
    source: str
    source_url: str
    collected_at: str
    game_build: str
    character: str
    act: int
    ascension_min: int
    ascension_max: int
    methodology: str
    cards: tuple[CardPickCount, ...]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> CardPriorDataset:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Card-prior snapshot must be a JSON object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CardPriorDataset:
        if raw.get("format") != CARD_PRIOR_FORMAT:
            raise ValueError("Unrecognized card-prior snapshot format")
        meta = raw.get("provenance")
        rows = raw.get("cards")
        if not isinstance(meta, dict) or not isinstance(rows, list):
            raise ValueError("Require provenance object and cards array")
        fields = ("source", "source_url", "collected_at", "game_build",
                  "character", "methodology")
        for field in fields:
            if not isinstance(meta.get(field), str) or not meta[field].strip():
                raise ValueError(f"Provenance field {field!r} must be nonempty")
        for field in ("act", "ascension_min", "ascension_max"):
            if type(meta.get(field)) is not int:
                raise ValueError(f"Provenance field {field!r} must be integer")
        if meta["act"] < 1 or meta["ascension_min"] < 0:
            raise ValueError("Invalid Act/ascension interval")
        if meta["ascension_max"] < meta["ascension_min"]:
            raise ValueError("Ascension range is inverted")
        if not rows:
            raise ValueError("Card-prior snapshot needs observations")
        cards: list[CardPickCount] = []
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError("Card-count row must be an object")
            card_id = item.get("card_id")
            if not isinstance(card_id, str):
                raise ValueError("Card identifier must be a string")
            offered = item.get("offered")
            picked = item.get("picked")
            if (
                isinstance(offered, bool) or not isinstance(offered, int)
                or isinstance(picked, bool) or not isinstance(picked, int)
            ):
                raise ValueError("Card offered/picked counts must be integers")
            cards.append(CardPickCount(card_id, offered, picked))
        if len({c.card_id for c in cards}) != len(cards):
            raise ValueError("Duplicate card identifier in aggregate snapshot")
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(
            source=meta["source"], source_url=meta["source_url"],
            collected_at=meta["collected_at"], game_build=meta["game_build"],
            character=meta["character"], act=meta["act"],
            ascension_min=meta["ascension_min"], ascension_max=meta["ascension_max"],
            methodology=meta["methodology"], cards=tuple(cards),
            fingerprint=fingerprint,
        )

    def matches(self, context: CardPriorContext) -> bool:
        return (
            self.game_build == context.game_build
            and self.character == context.character
            and self.act == context.act
            and self.ascension_min <= context.ascension <= self.ascension_max
        )


@dataclass(frozen=True, slots=True)
class BlendedPrior:
    probabilities: tuple[float, ...]
    matched_cards: int
    used_external_data: bool
    dataset_fingerprint: str | None


def _softmax(logits: tuple[float, ...]) -> tuple[float, ...]:
    if not logits or any(not math.isfinite(x) for x in logits):
        raise ValueError("Logits must be finite and nonempty")
    shift = max(logits)
    scaled = [math.exp(x - shift) for x in logits]
    total = math.fsum(scaled)
    return tuple(v / total for v in scaled)


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1.0 - p))


def blended_card_reward_prior(
    observation: Observation,
    actions: tuple[LegalAction, ...],
    learned_logits: tuple[float, ...],
    *,
    context: CardPriorContext,
    dataset: CardPriorDataset | None,
    human_weight: float = 0.0,
    smoothing_exposures: float = 20.0,
    score_scale: float = 1.0,
) -> BlendedPrior:
    """Blend a learned masked softmax with a contextual, shrinkage-smoothed hint.

    Missing cards and Skip get neutral relative logits. A snapshot is active
    only when all actions constitute a card-reward choice and metadata matches
    the declared experimental context *and* visible Act.
    """
    if len(actions) != len(learned_logits) or not actions:
        raise ValueError("Need one learned logit per legal action")
    if not math.isfinite(human_weight) or not 0.0 <= human_weight <= 1.0:
        raise ValueError("human_weight must lie in [0, 1]")
    if not math.isfinite(smoothing_exposures) or smoothing_exposures <= 0:
        raise ValueError("smoothing_exposures must be positive and finite")
    if not math.isfinite(score_scale) or score_scale < 0:
        raise ValueError("score_scale must be nonnegative and finite")
    learned = _softmax(learned_logits)
    state = state_dict(observation.payload_json)
    reward_only = (
        set(a.kind for a in actions) <= _REWARD_KINDS
        and any(a.kind == "take_reward_card" for a in actions)
    )
    if (
        dataset is None or human_weight == 0.0
        or not reward_only or not dataset.matches(context)
        or state.get("act") != context.act
    ):
        return BlendedPrior(learned, 0, False, None)

    total_offers = sum(c.offered for c in dataset.cards)
    base_rate = sum(c.picked for c in dataset.cards) / total_offers
    baseline = _logit(base_rate)
    by_card = {c.card_id: c for c in dataset.cards}
    scores: list[float] = []
    matched = 0
    for action in actions:
        semantic = semantic_action_label(state, action.kind, action.payload_json)
        prefix = "take_reward_card:"
        card = by_card.get(semantic[len(prefix):]) if semantic.startswith(prefix) else None
        if card is None:
            scores.append(0.0)
            continue
        matched += 1
        adjusted = ((card.picked + smoothing_exposures * base_rate)
                    / (card.offered + smoothing_exposures))
        scores.append(score_scale * (_logit(adjusted) - baseline))
    if matched == 0:
        return BlendedPrior(learned, 0, False, None)
    human = _softmax(tuple(scores))
    mixed = tuple(
        (1.0 - human_weight) * model_p + human_weight * human_p
        for model_p, human_p in zip(learned, human, strict=True)
    )
    return BlendedPrior(mixed, matched, True, dataset.fingerprint)
