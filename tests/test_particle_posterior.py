"""Incremental public-history particle filtering: artificial latent bits.

This fixture makes the empirical seed cohort precisely inspectable. A finite
cohort conditions exactly on its own discrete prior, NOT the full game law.
"""
from __future__ import annotations

import hashlib
import json
import random

import pytest

from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.emulator.chance import PublicHistoryStep, require_fair_sampler
from sts2_ai.emulator.particle_posterior import (
    FINITE_COHORT_PRIOR_ID,
    FiniteSeedPosteriorSampler,
    ParticlePosteriorExhausted,
)
from sts2_ai.search.fair_replay import FairReplayPuctAdapter
from sts2_ai.search.puct import StochasticPuct

POLICY = InformationPolicy("toy-public-v1")
REVEAL = LegalAction("reveal", "reveal")
FINISH = LegalAction("finish", "finish")


class TwoStepBackend:
    def __init__(self) -> None:
        self.states: dict[str, tuple[str, int]] = {}
        self.next_id = 0
        self.seeds: list[str] = []
        self.fork_calls = 0
        self.step_calls = 0

    @staticmethod
    def bit(seed: str) -> int:
        return hashlib.sha256(seed.encode()).digest()[0] % 2

    def _store(self, seed: str, phase: int) -> str:
        self.next_id += 1
        handle = f"hypothetical-{self.next_id}"
        self.states[handle] = (seed, phase)
        return handle

    def reset(self, seed: str, ascension: int = 0) -> str:
        assert ascension == 0
        self.seeds.append(seed)
        return self._store(seed, 0)

    def fork(self, state: str) -> str:
        self.fork_calls += 1
        return self._store(*self.states[state])

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        seed, phase = self.states[state]
        data: dict[str, object] = {"phase": phase}
        if phase >= 1:
            data["hint"] = self.bit(seed)
        if phase >= 2:
            data["terminal_outcome"] = (
                "victory" if self.bit(seed) == 1 else "defeat"
            )
        payload = json.dumps(data, sort_keys=True)
        return Observation(policy.policy_id, payload, payload)

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        phase = self.states[state][1]
        return (REVEAL,) if phase == 0 else (FINISH,) if phase == 1 else ()

    def step(self, state: str, action: LegalAction) -> Transition:
        if action not in self.legal_actions(state):
            raise ValueError("Illegal action")
        self.step_calls += 1
        seed, phase = self.states[state]
        child = self._store(seed, phase + 1)
        return Transition(state, action, child, phase == 1)

    def release_many(self, states: tuple[str, ...] | list[str]) -> int:
        for handle in states:
            del self.states[handle]
        return len(states)


def initial(backend: TwoStepBackend) -> tuple[Observation, tuple[LegalAction, ...]]:
    handle = backend.reset("unrelated-live-seed")
    obs = backend.observe(handle, POLICY)
    actions = backend.legal_actions(handle)
    backend.release_many([handle])
    return obs, actions


def test_cohort_is_empirical_prior_not_hidden_live_seed() -> None:
    backend = TwoStepBackend()
    observed, legal = initial(backend)
    with FiniteSeedPosteriorSampler(backend) as cohort:
        cohort.initialize(observed, legal, search_rng=random.Random(7), cohort_size=400)
        assert require_fair_sampler(cohort) is cohort
        assert cohort.seed_prior_id == FINITE_COHORT_PRIOR_ID
        assert cohort.stats.initial_candidates == 400
        assert cohort.stats.surviving == 400
        assert all(seed.startswith("belief-pool:") for seed in backend.seeds[1:])
        prior_rate = sum(backend.bit(seed) for seed in backend.seeds[1:]) / 400
        assert prior_rate == pytest.approx(0.5, abs=0.07)
        reset_count = len(backend.seeds)
        history = cohort.public_history
        assert history is not None
        samples = cohort.sample_fair_continuations(
            history, search_rng=random.Random(42), count=1000
        )
        try:
            empirical = sum(backend.bit(backend.states[h][0]) for h in samples) / 1000
            assert empirical == pytest.approx(prior_rate, abs=0.06)
            assert all(backend.states[h][1] == 0 for h in samples)
        finally:
            backend.release_many(samples)
        assert backend.fork_calls == 1000
        assert len(backend.seeds) == reset_count  # no replays per simulation
    assert not backend.states


def test_observed_reveal_filters_cohort_and_keeps_hidden_future_correlation() -> None:
    backend = TwoStepBackend()
    observed, legal = initial(backend)
    with FiniteSeedPosteriorSampler(backend) as cohort:
        cohort.initialize(observed, legal, search_rng=random.Random(21), cohort_size=220)
        # A hypothetical observation is drawn independently from the cohort.
        real = backend.reset("observed-real-hidden-seed")
        live_next = backend.step(real, REVEAL).child
        revealed = backend.observe(live_next, POLICY)
        backend.release_many([real, live_next])
        cohort.advance(REVEAL, revealed, (FINISH,))
        stats = cohort.stats
        assert stats.observed_steps == 1
        assert stats.simulator_steps == 220
        assert 65 < stats.surviving < 155
        assert stats.survival_rate == pytest.approx(0.5, abs=0.2)
        history = cohort.public_history
        assert history is not None
        assert history[0] == PublicHistoryStep(observed, REVEAL, (REVEAL,))
        assert history[1] == PublicHistoryStep(revealed, None, (FINISH,))
        handles = cohort.sample_fair_continuations(
            history, search_rng=random.Random(93), count=500
        )
        try:
            expected = json.loads(revealed.payload_json)["hint"]
            for handle in handles:
                assert backend.observe(handle, POLICY) == revealed
                assert backend.bit(backend.states[handle][0]) == expected
            # After revealing the hidden bit, the future outcome is certain.
            child = backend.step(handles[0], FINISH).child
            try:
                outcome = json.loads(backend.observe(child, POLICY).payload_json)
                assert (outcome["terminal_outcome"] == "victory") == (expected == 1)
            finally:
                backend.release_many([child])
        finally:
            backend.release_many(handles)
    assert not backend.states


def test_collapse_refuses_oracle_fallback_and_cleans_up() -> None:
    backend = TwoStepBackend()
    observed, legal = initial(backend)
    cohort = FiniteSeedPosteriorSampler(backend)
    cohort.initialize(observed, legal, search_rng=random.Random(7), cohort_size=11)
    forged = Observation(POLICY.policy_id, '{"phase":999}', "forged")
    with pytest.raises(ParticlePosteriorExhausted):
        cohort.advance(REVEAL, forged, (FINISH,))
    assert not backend.states
    with pytest.raises(RuntimeError, match="closed"):
        cohort.sample_fair_continuations(
            (PublicHistoryStep(forged, None, (FINISH,)),),
            search_rng=random.Random(12),
            count=1,
        )


def test_foreign_history_and_illegal_action_are_rejected_before_sampling() -> None:
    backend = TwoStepBackend()
    observed, legal = initial(backend)
    with FiniteSeedPosteriorSampler(backend) as cohort:
        cohort.initialize(observed, legal, search_rng=random.Random(7), cohort_size=10)
        bad = (PublicHistoryStep(observed, None, (FINISH,)),)
        with pytest.raises(ValueError, match="history differs"):
            cohort.sample_fair_continuations(
                bad, search_rng=random.Random(2), count=1
            )
        with pytest.raises(ValueError, match="not in the last public"):
            cohort.advance(FINISH, observed, legal)
        with pytest.raises(ValueError, match="positive"):
            cohort.sample_fair_continuations(
                cohort.public_history or (), search_rng=random.Random(3), count=0
            )
        assert cohort.stats.surviving == 10
    assert not backend.states


def test_fair_puct_accepts_cohort_posterior_and_never_forks_live_state() -> None:
    backend = TwoStepBackend()
    observed, legal = initial(backend)
    with FiniteSeedPosteriorSampler(backend) as cohort:
        cohort.initialize(observed, legal, search_rng=random.Random(17), cohort_size=160)
        adapter = FairReplayPuctAdapter(
            backend, sampler=cohort, goal="prototype-full-victory-v1"
        )
        assert adapter.sampler is cohort
        root = adapter.root(cohort.public_history or (), legal)
        report = StochasticPuct(adapter, seed=33, max_depth=3).search(
            root, simulations=300
        )
        assert report.actions[0].visits == 300
        assert report.actions[0].success_mean == pytest.approx(0.5, abs=0.12)
        assert cohort.stats.forks == 300
        assert len(backend.states) == cohort.stats.surviving
    assert not backend.states
