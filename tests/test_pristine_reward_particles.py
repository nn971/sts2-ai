"""Regression: likelihood-weighted pristine reward proposals preserve joint priors."""
from __future__ import annotations

import json
import random

import pytest

from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.factorized_runstart import (
    FactorizedRunStartPosteriorSampler,
)
from sts2_ai.emulator.incremental_coupled_particles import (
    IncrementalCoupledParticlePosterior,
    IncrementalCoupledPosteriorExhausted,
)
from sts2_ai.emulator.protocol import InformationPolicy, LegalAction, Observation, Transition
from test_coupled_factorized_rejection import CoupledToyBackend, START, CHOOSE, FINISH

POLICY = InformationPolicy("prototype-fair-v0")


class RewardLikelihoodToyBackend(CoupledToyBackend):
    pristine_reward_proposal_schema = "prototype-pristine-reward-branch-v1"

    def __init__(self) -> None:
        super().__init__()
        self.reward_proposal_calls = 0

    def _reward_matches(self, data: dict[str, int]) -> bool:
        # Same visible boss for combat=0 and 2, but different probability
        # of a rare public reward: P(reward|combat0)=1/2,
        # P(reward|combat2)=1/4 under a fresh uniform reward stream.
        modulus = 4 if ((data["combat"] >> 1) & 1) else 2
        return data["reward"] % modulus == 0

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        data, phase = self._states[state]
        obj: dict[str, object] = {"phase": (1, 2, 3, 5)[phase]}
        if phase >= 1:
            obj["map_value"] = data["map"] & 1
            obj["act_one_boss_encounter_id"] = f"boss-{data['combat'] & 1}"
        if phase == 3:
            obj["public_reward"] = "special" if self._reward_matches(data) else "ordinary"
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return Observation(policy.policy_id, payload, payload)

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        return ((START,), (CHOOSE,), (FINISH,), ())[self._states[state][1]]

    def propose_pristine_reward(
        self,
        state: str,
        action: LegalAction,
        *,
        reward_initial_state: int,
    ) -> str:
        data, phase = self._states[state]
        if phase != 2 or action != FINISH:
            raise ValueError("Only a combat-to-first-reward action is eligible")
        self.reward_proposal_calls += 1
        if not 0 <= reward_initial_state < 1 << 64:
            raise ValueError("Invalid reward stream")
        replacement = data.copy()
        replacement["reward"] = reward_initial_state
        return self._store(replacement, 3)


def _reward_history(
    backend: RewardLikelihoodToyBackend,
) -> tuple[PublicHistoryStep, ...]:
    data = {
        "map": 11, "combat": 2, "reward": 0,
        "combat_targets": 0, "event": 0, "shop": 0
    }
    handles: list[str] = []
    try:
        state = backend.reset_factorized_hypothetical(data)
        handles.append(state)
        steps: list[PublicHistoryStep] = []
        for action in (START, CHOOSE, FINISH):
            steps.append(PublicHistoryStep(
                backend.observe(state, POLICY), action, backend.legal_actions(state)
            ))
            state = backend.step(state, action).child
            handles.append(state)
        steps.append(PublicHistoryStep(
            backend.observe(state, POLICY), None, backend.legal_actions(state)
        ))
        return tuple(steps)
    finally:
        backend.release_many(handles)


def _initialize_belief(
    backend: RewardLikelihoodToyBackend,
    history: tuple[PublicHistoryStep, ...],
    *,
    cohort_size: int,
) -> tuple[FactorizedRunStartPosteriorSampler, IncrementalCoupledParticlePosterior]:
    source = FactorizedRunStartPosteriorSampler(
        backend, map_states=(11,), combat_states=(0, 2)
    )
    belief = IncrementalCoupledParticlePosterior(
        backend, runstart_sampler=source
    )
    belief.initialize((
        history[0],
        PublicHistoryStep(history[1].observation, None, history[1].legal_actions),
    ), search_rng=random.Random(19), cohort_size=cohort_size)
    belief.advance(CHOOSE, history[2].observation, history[2].legal_actions or ())
    return source, belief


def test_reward_branching_weights_parents_by_evidence_probability() -> None:
    backend = RewardLikelihoodToyBackend()
    history = _reward_history(backend)
    source, belief = _initialize_belief(backend, history, cohort_size=256)
    try:
        assert belief.stats.surviving_particles == 256
        reset_calls = backend.reset_calls
        belief.advance_pristine_reward(
            FINISH,
            history[3].observation,
            history[3].legal_actions or (),
            search_rng=random.Random(201),
            proposals_per_parent=16,
        )
        assert backend.reset_calls == reset_calls
        assert backend.reward_proposal_calls == 256 * 16
        assert belief.stats.observed_transitions == 2
        assert belief.stats.surviving_particles > 800
        assert belief.stats.surviving_particles < 2100
        assert belief.public_history == history
        handles = belief.sample_fair_continuations(
            history, search_rng=random.Random(202), count=400
        )
        try:
            assert all(backend.observe(s, POLICY) == history[3].observation for s in handles)
            # Bayes: combat=2 has probability (1/4)/(1/4+1/2) = 1/3.
            ratio = sum(backend.latent(s)["combat"] == 2 for s in handles) / len(handles)
            assert ratio == pytest.approx(1/3, abs=0.085)
        finally:
            backend.release_many(handles)
    finally:
        belief.close()
        source.close()
    assert backend.active == 0


def test_reward_proposal_validation_preserves_cohort_and_cleanup_on_collapse() -> None:
    backend = RewardLikelihoodToyBackend()
    history = _reward_history(backend)
    source, belief = _initialize_belief(backend, history, cohort_size=16)
    try:
        retained = backend.active
        with pytest.raises(ValueError, match="positive"):
            belief.advance_pristine_reward(
                FINISH, history[3].observation, (),
                search_rng=random.Random(1), proposals_per_parent=0,
            )
        with pytest.raises(ValueError, match="exceeds"):
            belief.advance_pristine_reward(
                FINISH, history[3].observation, (),
                search_rng=random.Random(1), proposals_per_parent=5,
                max_total_proposals=30,
            )
        assert backend.active == retained
        unreachable = Observation(POLICY.policy_id, '{"phase":5,"public_reward":"never"}', "x")
        with pytest.raises(IncrementalCoupledPosteriorExhausted):
            belief.advance_pristine_reward(
                FINISH, unreachable, (),
                search_rng=random.Random(5), proposals_per_parent=4,
            )
        assert backend.active == 0
    finally:
        belief.close()
        source.close()
