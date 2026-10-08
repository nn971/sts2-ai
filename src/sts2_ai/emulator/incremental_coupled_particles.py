"""Incrementally filtered joint latent states under the factorized research prior.

First sample one iid finite cohort from the *exact* independent-stream
RunStart posterior, then propagate each whole-state particle through the
actual observed actions. Eliminate particles whose full public observation
or legal-action menu differs. No RNG streams are independently rekeyed or
decorrelated after gameplay starts; each survivor carries its exact cursor.

CRITICAL: uniform survivor sampling is exact for the *realized empirical
cohort* only. It is NOT the exact posterior of the underlying independent-
stream prior, even when the RunStart source itself was exact. We never
resample or duplicate survivors on collapse; we fail closed.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .chance import (
    FAIR_CONTINUATION_CAPABILITY_ID,
    FairContinuationUnavailable,
    PublicHistoryStep,
)
from .factorized_runstart import (
    FACTORIZED_RUNSTART_PRIOR_ID,
    FactorizedRunStartBackend,
    FactorizedRunStartPosteriorSampler,
)
from .protocol import InformationPolicy, LegalAction, Observation, StateHandle
from .rejection import FairHistoryRejectionSampler

INCREMENTAL_COUPLED_COHORT_PRIOR_ID = (
    "empirical-cohort-of-independent-initial-streams-v1"
)


class IncrementalCoupledPosteriorExhausted(FairContinuationUnavailable):
    """A finite empirical cohort has no matching joint latent particles."""


class IncrementalCoupledBackend(FactorizedRunStartBackend, Protocol):
    def fork(self, state: StateHandle) -> StateHandle: ...

    @property
    def pristine_reward_proposal_schema(self) -> str | None: ...

    def propose_pristine_reward(
        self, state: StateHandle, action: LegalAction, *,
        reward_initial_state: int
    ) -> StateHandle: ...


@dataclass(frozen=True, slots=True)
class IncrementalCoupledStats:
    initial_particles: int
    surviving_particles: int
    observed_transitions: int
    simulator_transitions: int
    sampled_forks: int

    @property
    def survival_fraction(self) -> float:
        return (
            self.surviving_particles / self.initial_particles
            if self.initial_particles else 0.0
        )

    @property
    def empirical_effective_sample_size(self) -> int:
        """Uniform unweighted survivor count, NOT underlying-prior ESS."""
        return self.surviving_particles


class IncrementalCoupledParticlePosterior:
    """Persistent, fixed-size independent finite cohort of full latent states.

    Initialize at the public RunStart -> MapChoice boundary. The source
    factorized prior is conditioned exactly there; each cohort particle
    samples *all six* streams. The cohort's empirical law is then filtered
    jointly, one observed decision at a time. Samples fork surviving
    hypothetical states, never actual/live state handles.

    The source factorized sampler must remain open for initialization.
    After initialization the cohort owns its states independently.
    """

    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID
    seed_prior_id = INCREMENTAL_COUPLED_COHORT_PRIOR_ID
    source_prior_id = FACTORIZED_RUNSTART_PRIOR_ID

    def __init__(
        self,
        backend: IncrementalCoupledBackend,
        *,
        runstart_sampler: FactorizedRunStartPosteriorSampler,
    ) -> None:
        if not isinstance(runstart_sampler, FactorizedRunStartPosteriorSampler):
            raise TypeError("An exact factorized RunStart posterior is required")
        self._backend = backend
        self._source = runstart_sampler
        self._states: list[StateHandle] = []
        self._history: tuple[PublicHistoryStep, ...] | None = None
        self._initial_particles = 0
        self._observed_transitions = 0
        self._simulator_transitions = 0
        self._sampled_forks = 0
        self._closed = False

    @property
    def stats(self) -> IncrementalCoupledStats:
        return IncrementalCoupledStats(
            initial_particles=self._initial_particles,
            surviving_particles=len(self._states),
            observed_transitions=self._observed_transitions,
            simulator_transitions=self._simulator_transitions,
            sampled_forks=self._sampled_forks,
        )

    @property
    def public_history(self) -> tuple[PublicHistoryStep, ...] | None:
        return self._history

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Incremental coupled posterior is closed")

    def initialize(
        self,
        history: Sequence[PublicHistoryStep],
        *,
        search_rng: random.Random,
        cohort_size: int = 256,
    ) -> None:
        """Draw one finite cohort at MapChoice, without later-state peeking."""
        self._check_open()
        if self._history is not None:
            raise RuntimeError("Incremental posterior already initialized")
        transcript = tuple(history)
        if len(transcript) != 2:
            raise ValueError("Initialize at RunStart -> MapChoice only")
        FairHistoryRejectionSampler._validate_history(transcript)
        if type(cohort_size) is not int or cohort_size <= 0:
            raise ValueError("cohort_size must be positive")

        existing = self._source.public_history
        if existing is None:
            self._source.initialize(transcript)
        elif existing != transcript:
            raise ValueError("Factorized source has a different public history")

        # Source itself handles partial-allocation cleanup on failure.
        self._states = list(self._source.sample_fair_continuations(
            transcript, search_rng=search_rng, count=cohort_size
        ))
        self._initial_particles = cohort_size
        self._history = transcript

    def advance(
        self,
        action: LegalAction,
        next_observation: Observation,
        next_legal_actions: Sequence[LegalAction],
    ) -> None:
        """Replay one actual chosen action on *every* surviving hidden state.

        On a backend exception, empty cohort or internal mismatch we close
        and release all handles rather than keep a half-updated posterior.
        Invalid caller arguments are rejected without changing the cohort.
        """
        self._check_open()
        history = self._history
        if history is None:
            raise RuntimeError("Initialize at MapChoice first")
        if action not in (history[-1].legal_actions or ()):
            raise ValueError("Chosen action is not in the public legal menu")
        if next_observation.policy_id != history[-1].observation.policy_id:
            raise ValueError("Cannot change the public information policy")
        next_frame = PublicHistoryStep(
            next_observation, None, tuple(next_legal_actions)
        )
        prior_states, self._states = self._states, []
        remaining = set(prior_states)
        children: set[StateHandle] = set()
        accepted: list[StateHandle] = []
        policy = InformationPolicy(next_observation.policy_id)
        try:
            for parent in prior_states:
                child = self._backend.step(parent, action).child
                self._simulator_transitions += 1
                children.add(child)
                self._backend.release_many((parent,))
                remaining.remove(parent)
                if (
                    self._backend.observe(child, policy) == next_observation
                    and tuple(self._backend.legal_actions(child))
                    == next_frame.legal_actions
                ):
                    accepted.append(child)
                    children.remove(child)
                else:
                    self._backend.release_many((child,))
                    children.remove(child)

            if not accepted:
                raise IncrementalCoupledPosteriorExhausted(
                    "All independently sampled joint-state particles were "
                    "eliminated by the public observation; finite cohort "
                    "collapse cannot be repaired using the live seed."
                )
            self._states = accepted
            previous = history[-1]
            self._history = history[:-1] + (
                PublicHistoryStep(
                    previous.observation, action, previous.legal_actions
                ),
                next_frame,
            )
            self._observed_transitions += 1
        except BaseException:
            owned = remaining | children | set(accepted)
            self._states = []
            self._closed = True
            if owned:
                self._backend.release_many(tuple(owned))
            raise

    def advance_pristine_reward(
        self,
        action: LegalAction,
        next_observation: Observation,
        next_legal_actions: Sequence[LegalAction],
        *,
        search_rng: random.Random,
        proposals_per_parent: int = 32,
        max_total_proposals: int = 16384,
    ) -> None:
        """Condition on first reward by expanding each empirical parent equally.

        Each pre-reward particle has equal empirical mass. For each parent,
        independently draw K full-uniform reward initial states while keeping
        *every other stream and cursor unchanged*. All accepted children
        have equal weight. This estimates the parent's public reward evidence
        probability and therefore automatically weights ancestor particles
        correctly (rather than accepting one candidate per ancestor).

        Exact for the generated N*K empirical joint proposal cohort;
        NOT exact for the full independent-stream prior at finite K.
        The pinned emulator enforces pristine reward cursor==0 and
        experimental hypothetical provenance; unsupported states fail closed.
        """
        self._check_open()
        history = self._history
        if history is None:
            raise RuntimeError("Initialize at MapChoice first")
        if self._backend.pristine_reward_proposal_schema != (
            "prototype-pristine-reward-branch-v1"
        ):
            raise RuntimeError("Pristine reward proposal capability unavailable")
        if type(proposals_per_parent) is not int or proposals_per_parent <= 0:
            raise ValueError("proposals_per_parent must be a positive integer")
        if type(max_total_proposals) is not int or max_total_proposals <= 0:
            raise ValueError("max_total_proposals must be positive")
        if len(self._states) * proposals_per_parent > max_total_proposals:
            raise ValueError("Requested reward cohort expansion exceeds max_total_proposals")
        if action not in (history[-1].legal_actions or ()):
            raise ValueError("Chosen action is not in the public legal menu")
        if next_observation.policy_id != history[-1].observation.policy_id:
            raise ValueError("Cannot change the public information policy")

        import json

        current_payload = json.loads(history[-1].observation.payload_json)
        next_payload = json.loads(next_observation.payload_json)
        if (
            not isinstance(current_payload, dict)
            or not isinstance(next_payload, dict)
            or current_payload.get("phase") != 3
            or next_payload.get("phase") != 5
        ):
            raise ValueError("Pristine reward proposals require Combat -> Reward")

        next_frame = PublicHistoryStep(
            next_observation, None, tuple(next_legal_actions)
        )
        prior_states, self._states = self._states, []
        remaining = set(prior_states)
        unhandled_children: set[StateHandle] = set()
        accepted: list[StateHandle] = []
        policy = InformationPolicy(next_observation.policy_id)
        try:
            for parent in prior_states:
                for _ in range(proposals_per_parent):
                    child = self._backend.propose_pristine_reward(
                        parent, action, reward_initial_state=search_rng.getrandbits(64)
                    )
                    self._simulator_transitions += 1
                    unhandled_children.add(child)
                    if (
                        self._backend.observe(child, policy) == next_observation
                        and tuple(self._backend.legal_actions(child))
                        == next_frame.legal_actions
                    ):
                        accepted.append(child)
                        unhandled_children.remove(child)
                    else:
                        self._backend.release_many((child,))
                        unhandled_children.remove(child)
                self._backend.release_many((parent,))
                remaining.remove(parent)

            if not accepted:
                raise IncrementalCoupledPosteriorExhausted(
                    "Every independent pristine reward proposal was inconsistent "
                    "with the observed reward; budget exhaustion does not imply "
                    "zero true reward probability."
                )
            self._states = accepted
            previous = history[-1]
            self._history = history[:-1] + (
                PublicHistoryStep(
                    previous.observation, action, previous.legal_actions
                ),
                next_frame,
            )
            self._observed_transitions += 1
        except BaseException:
            owned = remaining | unhandled_children | set(accepted)
            self._states = []
            self._closed = True
            if owned:
                self._backend.release_many(tuple(owned))
            raise

    def sample_fair_continuations(
        self,
        history: Sequence[PublicHistoryStep],
        *,
        search_rng: random.Random,
        count: int,
    ) -> tuple[StateHandle, ...]:
        """IID with replacement under the fixed empirical survivor measure."""
        self._check_open()
        if self._history is None or tuple(history) != self._history:
            raise ValueError("Requested history differs from the conditioned cohort")
        if type(count) is not int or count <= 0:
            raise ValueError("Sample count must be positive")
        if not self._states:
            raise IncrementalCoupledPosteriorExhausted("No compatible particles")
        results: list[StateHandle] = []
        try:
            for _ in range(count):
                particle = search_rng.choice(self._states)
                results.append(self._backend.fork(particle))
            self._sampled_forks += count
            return tuple(results)
        except BaseException:
            if results:
                self._backend.release_many(tuple(results))
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        states, self._states = self._states, []
        if states:
            self._backend.release_many(tuple(states))

    def __enter__(self) -> IncrementalCoupledParticlePosterior:
        self._check_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
