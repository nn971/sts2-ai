"""Incrementally conditioned finite-seed posterior: a bounded research baseline.

Every particle is reset from a fresh *search-side* seed, never copied from the
actual run. Observed actions advance each particle exactly once; mismatching
public observations or legal menus remove it. Sampling forks one of the surviving
hypothetical states, not the live hidden state.

Conditional on the generated finite cohort, this is the **exact empirical
posterior**. It is NOT exact for the original 128-bit seed distribution:
rare-but-valid alternatives missing from the cohort never reappear, and a pool
may collapse. Do not use these labels as independently sampled ground truth
for full-distribution variance training.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .chance import (
    FAIR_CONTINUATION_CAPABILITY_ID,
    FairContinuationUnavailable,
    PublicHistoryStep,
)
from .protocol import EmulatorBackend, InformationPolicy, LegalAction, Observation, StateHandle
from .rejection import FairHistoryRejectionSampler

FINITE_COHORT_PRIOR_ID = "search-seeded-finite-empirical-prior-v1"


class ParticlePosteriorExhausted(FairContinuationUnavailable):
    """No generated hypothetical seed matches the observed public transcript."""


@dataclass(frozen=True, slots=True)
class ParticlePosteriorStats:
    initial_candidates: int
    surviving: int
    observed_steps: int
    simulator_steps: int
    forks: int

    @property
    def survival_rate(self) -> float:
        if not self.initial_candidates:
            return 0.0
        return self.surviving / self.initial_candidates


class FiniteSeedPosteriorSampler:
    """Reusable finite-cohort approximation with exact incremental replay.

    Lifecycle: initialize(first public frame) -> advance(observed action, next
    public frame) -> sample_fair_continuations(current history) -> close().
    Owns only hypothetical seed-derived handles; sampled forks belong to caller.
    """

    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID
    seed_prior_id = FINITE_COHORT_PRIOR_ID

    def __init__(self, backend: EmulatorBackend) -> None:
        self._backend = backend
        self._states: list[StateHandle] = []
        self._history: tuple[PublicHistoryStep, ...] | None = None
        self._initial_candidates = 0
        self._observed_steps = 0
        self._simulator_steps = 0
        self._forks = 0
        self._closed = False

    @property
    def stats(self) -> ParticlePosteriorStats:
        return ParticlePosteriorStats(
            initial_candidates=self._initial_candidates,
            surviving=len(self._states),
            observed_steps=self._observed_steps,
            simulator_steps=self._simulator_steps,
            forks=self._forks,
        )

    @property
    def public_history(self) -> tuple[PublicHistoryStep, ...] | None:
        return self._history

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Posterior sampler has been closed")

    @staticmethod
    def _matches(
        backend: EmulatorBackend,
        state: StateHandle,
        step: PublicHistoryStep,
    ) -> bool:
        actual = backend.observe(state, InformationPolicy(step.observation.policy_id))
        return (
            actual.policy_id == step.observation.policy_id
            and actual.payload_json == step.observation.payload_json
            and tuple(backend.legal_actions(state)) == step.legal_actions
        )

    def initialize(
        self,
        observation: Observation,
        legal_actions: Sequence[LegalAction],
        *,
        search_rng: random.Random,
        cohort_size: int = 256,
    ) -> None:
        self._check_open()
        if self._history is not None:
            raise RuntimeError("Posterior already initialized")
        if type(cohort_size) is not int or cohort_size <= 0:
            raise ValueError("cohort_size must be a positive integer")
        first = PublicHistoryStep(observation, None, tuple(legal_actions))
        FairHistoryRejectionSampler._validate_history((first,))
        self._initial_candidates = cohort_size
        try:
            for _ in range(cohort_size):
                seed = f"belief-pool:{search_rng.getrandbits(128):032x}"
                state = self._backend.reset(seed)
                try:
                    matching = self._matches(self._backend, state, first)
                except BaseException:
                    self._backend.release_many((state,))
                    raise
                if matching:
                    self._states.append(state)
                else:
                    self._backend.release_many((state,))
            if not self._states:
                raise ParticlePosteriorExhausted(
                    "All independently seeded cohort candidates disagree with "
                    "the initial public observation; increase cohort size or "
                    "use a mechanically conditioned sampler."
                )
            self._history = (first,)
        except BaseException:
            self.close()
            raise

    def advance(
        self,
        action: LegalAction,
        next_observation: Observation,
        next_legal_actions: Sequence[LegalAction],
    ) -> None:
        """Condition all existing hypothetical states on one observed transition."""
        self._check_open()
        history = self._history
        if history is None:
            raise RuntimeError("Initialize the finite cohort before advancing")
        if action not in history[-1].legal_actions:
            raise ValueError("Chosen action is not in the last public legal menu")
        next_step = PublicHistoryStep(next_observation, None, tuple(next_legal_actions))
        if next_observation.policy_id != history[-1].observation.policy_id:
            raise ValueError("Cannot change the information policy while conditioning")
        current = self._states
        self._states = []
        remaining: set[StateHandle] = set(current)
        produced: set[StateHandle] = set()
        try:
            for state in current:
                transition = self._backend.step(state, action)
                self._simulator_steps += 1
                child = transition.child
                produced.add(child)
                self._backend.release_many((state,))
                remaining.remove(state)
                if self._matches(self._backend, child, next_step):
                    self._states.append(child)
                else:
                    self._backend.release_many((child,))
                    produced.remove(child)
            if not self._states:
                raise ParticlePosteriorExhausted(
                    "All finite-prior candidates were eliminated by the observed "
                    "transition. The approximate posterior cannot be repaired "
                    "by duplicating survivors or by copying the live hidden seed."
                )
            self._history = history[:-1] + (
                PublicHistoryStep(history[-1].observation, action, history[-1].legal_actions),
                next_step,
            )
            self._observed_steps += 1
        except BaseException:
            # All surviving/new and unprocessed old handles belong to this
            # sampler, including states from partial transitions.
            owned = remaining | produced
            if owned:
                self._backend.release_many(tuple(owned))
            self._states = []
            self._closed = True
            raise

    def sample_fair_continuations(
        self,
        history: Sequence[PublicHistoryStep],
        *,
        search_rng: random.Random,
        count: int,
    ) -> tuple[StateHandle, ...]:
        """With-replacement iid draws from the *empirical* conditional cohort."""
        self._check_open()
        if self._history is None or tuple(history) != self._history:
            raise ValueError("Requested history differs from the conditioned cohort")
        if type(count) is not int or count <= 0:
            raise ValueError("Sample count must be positive")
        if not self._states:
            raise ParticlePosteriorExhausted("No consistent particles")
        sampled: list[StateHandle] = []
        try:
            for _ in range(count):
                index = search_rng.randrange(len(self._states))
                sampled.append(self._backend.fork(self._states[index]))
            self._forks += count
            return tuple(sampled)
        except BaseException:
            if sampled:
                self._backend.release_many(tuple(sampled))
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        states, self._states = self._states, []
        if states:
            self._backend.release_many(tuple(states))

    def __enter__(self) -> FiniteSeedPosteriorSampler:
        self._check_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
