from __future__ import annotations

import random
from collections import Counter

import pytest

from sts2_ai.emulator.chance import (
    CHANCE_LAW_VERSION,
    FairContinuationUnavailable,
    KnownDeckDrawLaw,
    require_fair_sampler,
)


def test_known_deck_hypergeometric_probabilities() -> None:
    law = KnownDeckDrawLaw.from_counts({"strike": 2, "defend": 1})
    assert CHANCE_LAW_VERSION == "known-deck-draw-v1"
    assert law.remaining == 3
    distribution = law.distribution(2)
    by_counts = {outcome.counts: outcome.probability for outcome in distribution}
    assert by_counts[(("defend", 1), ("strike", 1))] == pytest.approx(2 / 3)
    assert by_counts[(("strike", 2),)] == pytest.approx(1 / 3)
    assert sum(by_counts.values()) == pytest.approx(1.0)
    assert law.distribution(0)[0].probability == 1.0


def test_ordered_draw_frequencies_are_seed_blind() -> None:
    law = KnownDeckDrawLaw.from_counts({"strike": 2, "defend": 1})
    counts = Counter(
        law.sample_ordered(1, rng=random.Random(seed))[0]
        for seed in range(10_000)
    )
    assert abs(counts["strike"] / 10_000 - 2 / 3) < 0.025
    assert abs(counts["defend"] / 10_000 - 1 / 3) < 0.025
    # The known multiset, rather than a private run seed, determines the law.
    assert law == KnownDeckDrawLaw.from_counts({"defend": 1, "strike": 2})


def test_draws_are_without_replacement() -> None:
    law = KnownDeckDrawLaw.from_counts({"strike": 2, "defend": 1})
    for seed in range(60):
        drawn = Counter(law.sample_ordered(3, rng=random.Random(seed)))
        assert drawn == Counter({"strike": 2, "defend": 1})


def test_invalid_or_large_support_fails_closed() -> None:
    with pytest.raises(ValueError):
        KnownDeckDrawLaw.from_counts({"card": -1})
    law = KnownDeckDrawLaw.from_counts({"A": 1, "B": 1, "C": 1})
    with pytest.raises(ValueError):
        law.distribution(2, max_outcomes=2)
    with pytest.raises(ValueError):
        law.sample_ordered(4, rng=random.Random(1))


def test_real_fair_root_sampler_is_explicitly_unavailable() -> None:
    class OracleOnly:
        def fork(self, state: str) -> str:
            return state

    with pytest.raises(FairContinuationUnavailable, match="oracle-only"):
        require_fair_sampler(OracleOnly())

    class UnverifiedSampler:
        fair_continuation_capability_id = "randomly-reseeded-oracle-v0"

        def sample_fair_continuations(
            self, history: object, *, search_rng: object, count: int
        ) -> tuple[str, ...]:
            return ("unsafe",) * count

    with pytest.raises(FairContinuationUnavailable):
        require_fair_sampler(UnverifiedSampler())
