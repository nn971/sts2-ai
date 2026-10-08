"""Conditional next-card laws from *certified* public combat-pile knowledge.

Idealized exchangeable-shuffle model, NOT an assertion that STS2's concrete
seeded PRNG produces mathematically uniform permutations. The actual native
shuffler and hidden correlations need separate parity checks.

A caller must independently certify that every card remaining in the hidden
pile is accounted for. Unknown generated cards, card ordering effects and
unseen changes to draw-pile membership invalidate that certificate.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .chance import KnownDeckDrawLaw
from .protocol import Observation

DRAW_BELIEF_VERSION = "certified-exchangeable-combat-draw-v1"


class UncertifiedDrawPile(ValueError):
    """Public information does not justify the asserted hidden pile contents."""


def _count_tokens(tokens: Sequence[str]) -> KnownDeckDrawLaw:
    return KnownDeckDrawLaw.from_counts(Counter(tokens))


def _remove_one(counts: tuple[tuple[str, int], ...], card: str) -> tuple[tuple[str, int], ...]:
    values = dict(counts)
    if values.get(card, 0) <= 0:
        raise UncertifiedDrawPile(f"Revealed card {card!r} was absent from hidden pile")
    values[card] -= 1
    return KnownDeckDrawLaw.from_counts(values).counts


@dataclass(frozen=True, slots=True)
class OrderedDrawOutcome:
    cards: tuple[str, ...]
    probability: float


@dataclass(frozen=True, slots=True)
class DrawBelief:
    """Hidden exchangeable multiset plus optional *publicly known* top-card prefix.

    The `known_top` tuple is in draw order. Those cards have already been
    removed from `hidden_counts`. No RNG seed or exact hidden pile order is
    available to this object.
    """

    hidden_counts: tuple[tuple[str, int], ...]
    known_top: tuple[str, ...] = ()

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> DrawBelief:
        return cls(KnownDeckDrawLaw.from_counts(counts).counts)

    @property
    def unknown_size(self) -> int:
        return sum(value for _, value in self.hidden_counts)

    @property
    def remaining(self) -> int:
        return self.unknown_size + len(self.known_top)

    def __post_init__(self) -> None:
        if self.hidden_counts != KnownDeckDrawLaw.from_counts(dict(self.hidden_counts)).counts:
            raise ValueError("Hidden pile counts must be normalized and nonnegative")
        if any(not isinstance(card, str) or not card for card in self.known_top):
            raise ValueError("Known top-card tokens must be nonempty strings")

    def certify_known_top(self, cards: Sequence[str]) -> DrawBelief:
        """Transfer already-unordered cards into the publicly observed top prefix.

        Only call when game rules/public effects certify the listed ordered top
        cards. The underlying probabilities of that observation are handled by
        filtering below; this method itself does not supply evidence.
        """
        hidden = self.hidden_counts
        for card in cards:
            hidden = _remove_one(hidden, card)
        return DrawBelief(hidden, self.known_top + tuple(cards))

    def observe_next(self, card: str) -> tuple[DrawBelief, float]:
        """Condition on a card-type reveal; return posterior and likelihood."""
        if self.known_top:
            if card != self.known_top[0]:
                raise UncertifiedDrawPile(
                    "Observed draw contradicts the publicly known top card"
                )
            return DrawBelief(self.hidden_counts, self.known_top[1:]), 1.0
        if self.unknown_size == 0:
            raise UncertifiedDrawPile("Cannot reveal a card from an empty pile")
        available = dict(self.hidden_counts).get(card, 0)
        if not available:
            raise UncertifiedDrawPile("Revealed card was not in the hidden multiset")
        return DrawBelief(_remove_one(self.hidden_counts, card)), available / self.unknown_size

    def ordered_distribution(
        self, draws: int, *, max_outcomes: int = 4096
    ) -> tuple[OrderedDrawOutcome, ...]:
        if isinstance(draws, bool) or not isinstance(draws, int) or not 0 <= draws <= self.remaining:
            raise ValueError("Draws must lie between zero and remaining cards")
        if isinstance(max_outcomes, bool) or max_outcomes <= 0:
            raise ValueError("max_outcomes must be positive")
        outcomes: list[OrderedDrawOutcome] = []

        def expand(
            belief: DrawBelief, prefix: tuple[str, ...], p: float, remaining: int
        ) -> None:
            if remaining == 0:
                if len(outcomes) >= max_outcomes:
                    raise ValueError("Ordered draw support exceeds max_outcomes")
                outcomes.append(OrderedDrawOutcome(prefix, p))
                return
            if belief.known_top:
                top = belief.known_top[0]
                expand(DrawBelief(belief.hidden_counts, belief.known_top[1:]),
                       prefix + (top,), p, remaining - 1)
                return
            total = belief.unknown_size
            for card, count in belief.hidden_counts:
                next_belief = DrawBelief(_remove_one(belief.hidden_counts, card))
                expand(next_belief, prefix + (card,), p * count / total, remaining - 1)

        expand(self, (), 1.0, draws)
        return tuple(outcomes)

    def sample_ordered(
        self, draws: int, *, search_rng: random.Random
    ) -> tuple[str, ...]:
        """Sample a legal ordered future without using the live game's RNG."""
        if isinstance(draws, bool) or not isinstance(draws, int) or not 0 <= draws <= self.remaining:
            raise ValueError("Draws must lie between zero and remaining cards")
        fixed = self.known_top[:draws]
        remaining = draws - len(fixed)
        if remaining == 0:
            return fixed
        law = KnownDeckDrawLaw(self.hidden_counts)
        return fixed + law.sample_ordered(remaining, rng=search_rng)

    def after_reshuffle(self, known_discard: Mapping[str, int]) -> DrawBelief:
        """A reshuffle is legal only when the draw pile is publicly empty."""
        if self.remaining:
            raise UncertifiedDrawPile("Cannot reshuffle before draw pile is empty")
        return DrawBelief.from_counts(known_discard)


def _variant_token(card: object) -> str:
    if not isinstance(card, dict):
        raise UncertifiedDrawPile("Expected visible card objects in public piles")
    card_id = card.get("card_id")
    level = card.get("upgrade_level")
    if not isinstance(card_id, str) or not card_id:
        raise UncertifiedDrawPile("Public card has no stable card ID")
    if isinstance(level, bool) or not isinstance(level, int) or level < 0:
        raise UncertifiedDrawPile("Public card has an invalid upgrade level")
    # Deck/combat-local mutable/enchantment state is not certified by this v1
    # inventory; even apparently empty state may conceal mechanistic effects.
    state = card.get("state")
    if state is not None and state != {}:
        raise UncertifiedDrawPile("Card state requires a richer variant descriptor")
    if card.get("is_temporary") is True:
        raise UncertifiedDrawPile("Temporary/generated cards invalidate the initial deck")
    return json.dumps((card_id, level), separators=(",", ":"))


def certified_opening_draw_belief(
    observation: Observation, *, opening_frame_certified: bool = False
) -> DrawBelief:
    """Derive an unordered remaining pile only at an externally certified opening.

    The caller MUST have verified that this is the fresh *post-opening-draw*
    combat frame with no additional draw-pile edits, card creation, selection
    continuations, fixed top cards or hidden additions. This is deliberately
    not inferred solely from `turn == 1`: card plays can already occur then.

    All visible deck cards, hand, discard and exhaust are counted by card ID +
    upgrade level, never by ephemeral instance ID. Strict size equality avoids
    silently guessing unseen generated cards.
    """
    if not opening_frame_certified:
        raise UncertifiedDrawPile(
            "Cannot infer hidden pile from a combat screenshot without an "
            "audited opening-frame/no-deck-mutation certificate"
        )
    try:
        state = json.loads(observation.payload_json)
    except json.JSONDecodeError as exc:
        raise UncertifiedDrawPile("Malformed public observation") from exc
    if not isinstance(state, dict):
        raise UncertifiedDrawPile("Public observation must be a JSON object")
    deck = state.get("deck")
    combat = state.get("combat")
    if not isinstance(deck, list) or not isinstance(combat, dict):
        raise UncertifiedDrawPile("Expected combat and persistent deck")
    if combat.get("turn") != 1 or combat.get("pending_choice_id") is not None:
        raise UncertifiedDrawPile("Not a certified opening combat frame")
    hand = combat.get("hand")
    discard = combat.get("discard_pile")
    exhausted = combat.get("exhaust_pile")
    pile_size = combat.get("draw_pile_count")
    if not isinstance(hand, list) or discard != [] or exhausted != []:
        raise UncertifiedDrawPile("Initial-frame audit requires empty discard/exhaust")
    if isinstance(pile_size, bool) or not isinstance(pile_size, int) or pile_size < 0:
        raise UncertifiedDrawPile("Invalid public draw pile count")
    pool = Counter(_variant_token(card) for card in deck)
    for card in hand:
        token = _variant_token(card)
        if pool[token] <= 0:
            raise UncertifiedDrawPile("Visible hand disagrees with persistent deck")
        pool[token] -= 1
    counts = {card: n for card, n in pool.items() if n > 0}
    if sum(counts.values()) != pile_size:
        raise UncertifiedDrawPile(
            "Public draw count disagrees with known persistent deck leftovers"
        )
    return DrawBelief.from_counts(counts)
