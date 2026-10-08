"""Synthetic exact checks for a conditional, seed-blind combat draw belief."""
from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from sts2_ai.emulator import Observation
from sts2_ai.emulator.draw_belief import (
    DRAW_BELIEF_VERSION,
    DrawBelief,
    UncertifiedDrawPile,
    certified_opening_draw_belief,
)


def test_ordered_known_multiset_probabilities_and_reveal_posteriors() -> None:
    belief = DrawBelief.from_counts({"Strike": 2, "Defend": 1})
    assert DRAW_BELIEF_VERSION == "certified-exchangeable-combat-draw-v1"
    distribution = belief.ordered_distribution(2)
    probs = {outcome.cards: outcome.probability for outcome in distribution}
    assert probs == pytest.approx({
        ("Strike", "Strike"): 1 / 3,
        ("Strike", "Defend"): 1 / 3,
        ("Defend", "Strike"): 1 / 3,
    })
    after, chance = belief.observe_next("Defend")
    assert chance == pytest.approx(1 / 3)
    assert after.ordered_distribution(2)[0].cards == ("Strike", "Strike")
    assert after.ordered_distribution(2)[0].probability == pytest.approx(1)
    after, chance = belief.observe_next("Strike")
    assert chance == pytest.approx(2 / 3)
    assert after.hidden_counts == (("Defend", 1), ("Strike", 1))


def test_known_top_cards_use_no_hidden_seed_and_are_deterministic_prefix() -> None:
    belief = DrawBelief.from_counts({"A": 1, "B": 2, "C": 1})
    top = belief.certify_known_top(("B", "A"))
    assert top.known_top == ("B", "A")
    assert top.unknown_size == 2
    assert top.ordered_distribution(1)[0].cards == ("B",)
    assert top.ordered_distribution(1)[0].probability == 1.0
    for seed in range(20):
        cards = top.sample_ordered(4, search_rng=random.Random(seed))
        assert cards[:2] == ("B", "A")
        assert Counter(cards) == Counter({"A": 1, "B": 2, "C": 1})
    after1, likelihood1 = top.observe_next("B")
    after2, likelihood2 = after1.observe_next("A")
    assert likelihood1 == likelihood2 == 1
    assert after2.known_top == ()
    assert sum(item.probability for item in top.ordered_distribution(4)) == pytest.approx(1)


def test_ordered_sampling_matches_analytic_chance() -> None:
    belief = DrawBelief.from_counts({"Strike": 3, "Defend": 2})
    outcomes = Counter(
        belief.sample_ordered(1, search_rng=random.Random(i))[0]
        for i in range(10_000)
    )
    assert outcomes["Strike"] / 10_000 == pytest.approx(3 / 5, abs=0.025)
    assert outcomes["Defend"] / 10_000 == pytest.approx(2 / 5, abs=0.025)


def test_must_exhaust_drawpile_before_reshuffle_and_reveal_contradictions() -> None:
    belief = DrawBelief.from_counts({"A": 2})
    with pytest.raises(UncertifiedDrawPile, match="before draw pile is empty"):
        belief.after_reshuffle({"B": 1})
    empty, probability = belief.observe_next("A")
    assert probability == 1
    empty, probability = empty.observe_next("A")
    assert probability == 1
    assert empty.remaining == 0
    shuffled = empty.after_reshuffle({"B": 1, "C": 1})
    assert shuffled.unknown_size == 2
    with pytest.raises(UncertifiedDrawPile):
        belief.observe_next("B")
    with pytest.raises(UncertifiedDrawPile):
        belief.certify_known_top(("C",))
    with pytest.raises(UncertifiedDrawPile):
        belief.certify_known_top(("A",)).observe_next("B")


def test_exact_support_cap_and_zero_draws() -> None:
    belief = DrawBelief.from_counts({f"card{i}": 1 for i in range(24)})
    outcomes = belief.ordered_distribution(1, max_outcomes=24)
    assert len(outcomes) == 24
    assert sum(o.probability for o in outcomes) == pytest.approx(1)
    with pytest.raises(ValueError, match="support exceeds"):
        belief.ordered_distribution(2, max_outcomes=20)
    assert belief.ordered_distribution(0)[0].cards == ()
    assert belief.sample_ordered(0, search_rng=random.Random(0)) == ()


def observation(
    *, hand: list[dict[str, object]] | None = None,
    count: int = 2,
    deck: list[dict[str, object]] | None = None,
    turn: int = 1,
) -> Observation:
    cards: list[dict[str, object]] = deck if deck is not None else [
        {"card_id": "Strike", "upgrade_level": 0, "state": {}},
        {"card_id": "Strike", "upgrade_level": 0, "state": {}},
        {"card_id": "Defend", "upgrade_level": 1, "state": {}},
    ]
    combat = {
        "turn": turn,
        "hand": hand if hand is not None else [
            {"card_id": "Strike", "upgrade_level": 0, "state": {},
             "is_temporary": False}
        ],
        "draw_pile_count": count,
        "discard_pile": [],
        "exhaust_pile": [],
        "pending_choice_id": None,
    }
    return Observation("prototype-fair-v0", json.dumps({
        "combat": combat, "deck": cards
    }), "visible")


def test_public_snapshot_requires_external_opening_certification() -> None:
    snapshot = observation()
    with pytest.raises(UncertifiedDrawPile, match="certificate"):
        certified_opening_draw_belief(snapshot)
    result = certified_opening_draw_belief(snapshot, opening_frame_certified=True)
    assert result.remaining == 2
    assert json.dumps(("Defend", 1), separators=(",", ":")) in dict(result.hidden_counts)
    assert json.dumps(("Strike", 0), separators=(",", ":")) in dict(result.hidden_counts)


@pytest.mark.parametrize("changed", ["extra_card", "bad_count", "turn_two", "temporary", "mutated"])
def test_public_snapshot_rejects_uncertified_combat_mutations(changed: str) -> None:
    baseline = json.loads(observation().payload_json)
    if changed == "extra_card":
        baseline["combat"]["hand"][0]["card_id"] = "Generated"
    elif changed == "bad_count":
        baseline["combat"]["draw_pile_count"] = 3
    elif changed == "turn_two":
        baseline["combat"]["turn"] = 2
    elif changed == "temporary":
        baseline["combat"]["hand"][0]["is_temporary"] = True
    elif changed == "mutated":
        baseline["deck"][0]["state"] = {"effect": "unspecified"}
    bad = Observation("prototype-fair-v0", json.dumps(baseline), "changed")
    with pytest.raises(UncertifiedDrawPile):
        certified_opening_draw_belief(bad, opening_frame_certified=True)
