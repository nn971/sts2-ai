"""Joint full-history rejection from an *exact factorized RunStart proposal*.

At RunStart the prototype's declared *alternative* independent-stream prior
factorizes map/boss evidence. After a map action, mechanics may couple
streams. This sampler never factors those later observations: it samples a
complete hypothetical state from the correct factorized two-frame posterior,
executes every later observed player action with that same full hidden state,
and accepts the WHOLE resulting public trace or rejects the candidate.

Accepted samples are independent conditional draws under this specified
synthetic prior, assuming the backend is deterministic and search-side draws
follow the prior. The finite candidate cap is an explicit refusal, not a
posterior approximation. This does NOT establish native STS2 RNG fidelity.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .chance import (
    FAIR_CONTINUATION_CAPABILITY_ID,
    PublicHistoryStep,
)
from .factorized_runstart import (
    FACTORIZED_RUNSTART_PRIOR_ID,
    FactorizedRunStartBackend,
    FactorizedRunStartPosteriorSampler,
)
from .protocol import InformationPolicy, StateHandle
from .rejection import FairHistoryRejectionSampler, HistoryConditioningExhausted

COUPLED_FACTORIZED_PRIOR_ID = FACTORIZED_RUNSTART_PRIOR_ID
COUPLED_REJECTION_VERSION = "coupled-factorized-public-history-rejection-v1"


@dataclass(frozen=True, slots=True)
class CoupledRejectionStats:
    candidates: int
    accepted: int
    replay_steps: int
    public_checks: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.candidates if self.candidates else 0.0


class CoupledFactorizedHistoryRejectionSampler:
    """Full latent-stream replay from the factorized RunStart posterior.

    The source sampler *must* be an exact FactorizedRunStartPosteriorSampler.
    This class owns no persistent state handles; returned accepted handles
    belong to the caller, and every rejected candidate is released. Both
    samplers use only public frames/actions and fresh search-side randomness.

    The source sampler must not be closed while this sampler is in use.
    """

    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID
    seed_prior_id = COUPLED_FACTORIZED_PRIOR_ID
    conditioning_version = COUPLED_REJECTION_VERSION

    def __init__(
        self,
        backend: FactorizedRunStartBackend,
        *,
        runstart_sampler: FactorizedRunStartPosteriorSampler,
        max_candidates: int = 1024,
    ) -> None:
        if not isinstance(runstart_sampler, FactorizedRunStartPosteriorSampler):
            raise TypeError("An exact factorized RunStart posterior is required")
        if type(max_candidates) is not int or max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        self._backend = backend
        self._source = runstart_sampler
        self.max_candidates = max_candidates
        self.last_stats = CoupledRejectionStats(0, 0, 0, 0)

    @staticmethod
    def _prefix(
        history: tuple[PublicHistoryStep, ...],
    ) -> tuple[PublicHistoryStep, PublicHistoryStep]:
        if len(history) < 2:
            raise ValueError("Full history must include RunStart and MapChoice frames")
        first, second = history[:2]
        if first.chosen_action is None:
            raise ValueError("RunStart chosen action is missing")
        return (
            first,
            PublicHistoryStep(second.observation, None, second.legal_actions),
        )

    def sample_fair_continuations(
        self,
        history: Sequence[PublicHistoryStep],
        *,
        search_rng: random.Random,
        count: int,
    ) -> tuple[StateHandle, ...]:
        transcript = tuple(history)
        policy = FairHistoryRejectionSampler._validate_history(transcript)
        prefix = self._prefix(transcript)
        if type(count) is not int or count <= 0:
            raise ValueError("Requested sample count must be positive")

        # Full posterior samples must have been conditioned on the exact
        # public start-run transcript; no reconditioning from a foreign root.
        source_history = self._source.public_history
        if source_history is None:
            self._source.initialize(prefix)
        elif source_history != prefix:
            raise ValueError("Source factorized posterior differs from the public prefix")

        accepted: list[StateHandle] = []
        trials = 0
        replay_steps = 0
        public_checks = 0
        try:
            while len(accepted) < count and trials < self.max_candidates:
                trials += 1
                # Source produces iid complete latent states conditioned on
                # exactly the first two public frames, with advanced cursors.
                roots = self._source.sample_fair_continuations(
                    prefix, search_rng=search_rng, count=1
                )
                state: StateHandle | None = roots[0]
                consistent = True
                try:
                    # The first two observations were already checked exactly
                    # by the source sampler. For later frames, preserve the
                    # *same entire hidden state* through all game actions.
                    for index in range(1, len(transcript)):
                        record = transcript[index]
                        assert state is not None
                        public_checks += 1
                        if (
                            self._backend.observe(state, policy) != record.observation
                            or tuple(self._backend.legal_actions(state))
                            != record.legal_actions
                        ):
                            consistent = False
                            break
                        if record.chosen_action is not None:
                            actual_menu = tuple(self._backend.legal_actions(state))
                            if record.chosen_action not in actual_menu:
                                consistent = False
                                break
                            child = self._backend.step(state, record.chosen_action).child
                            replay_steps += 1
                            self._backend.release_many((state,))
                            state = child
                    if consistent:
                        assert state is not None
                        accepted.append(state)
                        state = None
                finally:
                    if state is not None:
                        self._backend.release_many((state,))

            self.last_stats = CoupledRejectionStats(
                trials, len(accepted), replay_steps, public_checks
            )
            if len(accepted) != count:
                raise HistoryConditioningExhausted(
                    "Joint factorized full-history rejection exhausted "
                    f"{trials} candidate complete states; accepted "
                    f"{len(accepted)}/{count}. No oracle or approximate fallback."
                )
            return tuple(accepted)
        except BaseException:
            self.last_stats = CoupledRejectionStats(
                trials, len(accepted), replay_steps, public_checks
            )
            if accepted:
                self._backend.release_many(tuple(accepted))
            raise
