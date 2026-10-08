"""Exact full-public-history posterior for a DECLARED FINITE seed universe.

This is a separate *game model*, not a Monte Carlo approximation claim about
native STS2. The prior is uniform over the exact, user-declared distinct set
of seed strings `belief-pool:<32-hex-digits>`. Every seed in the support is
replayed and conditioned against the entire observed action/observation/menu
history. Sampling with replacement is therefore exact under this finite
discrete prior, without accessing the true game's hidden seed.

Finite support can make a distinctive map identify the entire hidden seed.
That is an inherent property of this small experimental prior and does NOT
mean that a real STS2 player could infer the native seed.
"""
from __future__ import annotations

import random
from collections.abc import Sequence

from .chance import PublicHistoryStep
from .particle_posterior import FiniteSeedPosteriorSampler
from .protocol import EmulatorBackend
from .rejection import FairHistoryRejectionSampler

EXACT_FINITE_SEED_PRIOR_ID = "exact-uniform-enumerated-128bit-belief-seeds-v1"


class _EnumeratedSearchRng(random.Random):
    """Deterministic private iterator over every atom in an explicit prior."""

    def __init__(self, seed_values: tuple[int, ...]) -> None:
        super().__init__(0)
        self._seed_values = iter(seed_values)

    def getrandbits(self, k: int) -> int:
        if k != 128:
            raise ValueError("The enumerated prior uses 128-bit search seeds")
        try:
            return next(self._seed_values)
        except StopIteration as exc:
            raise RuntimeError("Enumerated prior consumed more than its support") from exc


class ExactFiniteSeedPosteriorSampler(FiniteSeedPosteriorSampler):
    """Exact public posterior over a completely enumerated seed support.

    The posterior is uniform on all originally declared seeds which yield the
    entire public transcript. No resampling, hidden-information fallback or
    posterior probability estimation is involved in determining its support.

    The backend must be deterministic for each seed. This is true of the
    pinned prototype emulator; it must be separately tested for other models.
    """

    seed_prior_id = EXACT_FINITE_SEED_PRIOR_ID

    def __init__(
        self,
        backend: EmulatorBackend,
        *,
        seed_values: Sequence[int],
        max_support: int = 4096,
    ) -> None:
        super().__init__(backend)
        if type(max_support) is not int or max_support <= 0:
            raise ValueError("max_support must be positive")
        values = tuple(seed_values)
        if not values or len(values) > max_support:
            raise ValueError("Seed support must be nonempty and within max_support")
        if any(type(value) is not int or value < 0 or value >= 1 << 128
               for value in values):
            raise ValueError("Every seed atom must be a 128-bit unsigned integer")
        if len(set(values)) != len(values):
            raise ValueError("Duplicate seeds would silently change prior weights")
        self._seed_values = values

    @property
    def prior_support_size(self) -> int:
        return len(self._seed_values)

    @property
    def posterior_support_size(self) -> int:
        return self.stats.surviving

    @property
    def evidence_probability(self) -> float:
        """Exact finite-prior probability of the complete observed history.

        Before initialization this value is undefined, not zero.
        """
        if self.public_history is None:
            raise RuntimeError("Posterior is not initialized")
        return self.stats.surviving / self.prior_support_size

    def initialize_history(
        self,
        history: Sequence[PublicHistoryStep],
    ) -> None:
        """Replay and condition **every** seed against the entire public history.

        On exhaustion, all sampled hypothetical handles are released and
        the sampler closes. Historic frames must include full legal menus.
        """
        transcript = tuple(history)
        FairHistoryRejectionSampler._validate_history(transcript)
        first_menu = transcript[0].legal_actions
        assert first_menu is not None
        self.initialize(
            transcript[0].observation,
            first_menu,
            search_rng=_EnumeratedSearchRng(self._seed_values),
            cohort_size=self.prior_support_size,
        )
        for previous, following in zip(transcript, transcript[1:], strict=False):
            assert previous.chosen_action is not None
            assert following.legal_actions is not None
            self.advance(
                previous.chosen_action,
                following.observation,
                following.legal_actions,
            )
        if self.public_history != transcript:
            raise ValueError("The replayed history does not equal its public transcript")
