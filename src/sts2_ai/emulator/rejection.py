"""Exact rejection conditioning under a declared independent *prototype* seed prior.

This correctness/reference sampler only uses public observations, legal actions
and newly reset hypothetical runs. It never forks the actual run or reads its
run seed. Conditioning on a long distinctive public history can be expensive;
exhaustion raises explicitly instead of substituting an oracle continuation.

The seed prior is uniform over 128-bit search-generated hex strings. This
defines a reproducible experimental distribution, *not* a verified native
Slay the Spire 2 seed distribution.
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
from .protocol import EmulatorBackend, InformationPolicy, StateHandle

SEED_PRIOR_ID = "uniform-search-128bit-hex-v1"


@dataclass(frozen=True, slots=True)
class RejectionStats:
    candidates: int
    accepted: int
    rejected: int
    emulator_steps: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.candidates if self.candidates else 0.0


class HistoryConditioningExhausted(FairContinuationUnavailable):
    """Finite rejection cap reached; the requested conditional law remains unknown."""


class FairHistoryRejectionSampler:
    """Draw posterior emulator states by replaying an observed action history.

    With iid candidate seeds from the declared prior and exact public
    observation matching, each accepted state has the correct *conditional*
    distribution under that prior. Every accepted handle belongs to the caller;
    all rejected/cancelled candidate handles are released here.
    """

    fair_continuation_capability_id = FAIR_CONTINUATION_CAPABILITY_ID
    seed_prior_id = SEED_PRIOR_ID

    def __init__(self, backend: EmulatorBackend, *, max_candidates: int = 256) -> None:
        if type(max_candidates) is not int or max_candidates <= 0:
            raise ValueError("max_candidates must be a positive integer")
        self._backend = backend
        self.max_candidates = max_candidates
        self.last_stats = RejectionStats(0, 0, 0, 0)

    @staticmethod
    def _validate_history(history: Sequence[PublicHistoryStep]) -> InformationPolicy:
        if not history:
            raise ValueError("An observed history must contain a root observation")
        if any(not step.observation.policy_id for step in history):
            raise ValueError("Observed information policy IDs must be nonempty")
        policy_id = history[0].observation.policy_id
        if any(step.observation.policy_id != policy_id for step in history):
            raise ValueError("Cannot condition on mixed information policies")
        if any(step.legal_actions is None for step in history):
            raise ValueError("Fair conditioning requires each visible legal-action menu")
        if any(step.chosen_action is None for step in history[:-1]):
            raise ValueError("Every historical observation requires its chosen action")
        if history[-1].chosen_action is not None:
            raise ValueError("The most recent observation must have no chosen action")
        return InformationPolicy(policy_id)

    def sample_fair_continuations(
        self,
        history: Sequence[PublicHistoryStep],
        *,
        search_rng: random.Random,
        count: int,
    ) -> tuple[StateHandle, ...]:
        policy = self._validate_history(history)
        if type(count) is not int or count <= 0:
            raise ValueError("Requested sample count must be positive")
        accepted: list[StateHandle] = []
        trials = 0
        steps = 0
        try:
            while len(accepted) < count and trials < self.max_candidates:
                trials += 1
                # Search-side RNG is independent from the live game seed.
                candidate = f"fair-sampling:{search_rng.getrandbits(128):032x}"
                state: StateHandle | None = self._backend.reset(candidate)
                try:
                    consistent = True
                    for record in history:
                        assert state is not None
                        observed = self._backend.observe(state, policy)
                        if (
                            observed.policy_id != record.observation.policy_id
                            or observed.payload_json != record.observation.payload_json
                        ):
                            consistent = False
                            break
                        actions = tuple(self._backend.legal_actions(state))
                        if actions != record.legal_actions:
                            consistent = False
                            break
                        if record.chosen_action is not None:
                            selected = next(
                                (
                                    action for action in actions
                                    if action.action_id == record.chosen_action.action_id
                                    and action.kind == record.chosen_action.kind
                                    and action.payload_json == record.chosen_action.payload_json
                                ),
                                None,
                            )
                            if selected is None:
                                consistent = False
                                break
                            transition = self._backend.step(state, selected)
                            steps += 1
                            self._backend.release_many([state])
                            state = transition.child
                    if consistent:
                        assert state is not None
                        accepted.append(state)
                        state = None
                finally:
                    if state is not None:
                        self._backend.release_many([state])
            self.last_stats = RejectionStats(trials, len(accepted), trials - len(accepted), steps)
            if len(accepted) != count:
                raise HistoryConditioningExhausted(
                    "Insufficient public-history-compatible independent seeds: "
                    f"accepted {len(accepted)}/{trials} candidate runs for "
                    f"{len(history)} observed decisions/frames. "
                    "The posterior is too selective for bounded rejection sampling."
                )
            return tuple(accepted)
        except BaseException:
            if accepted:
                self._backend.release_many(accepted)
            raise
