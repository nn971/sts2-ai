"""Exact finite-prior posterior on complete public histories.

Unlike iid particle Monte Carlo, the prior here is DEFINED as every unique
seed in the enumerated universe. Analytic posterior weights can be checked
by counting all atoms. No actual-run hidden seed is inspected.
"""
from __future__ import annotations

import hashlib
import json
import random

import pytest

from sts2_ai.emulator.chance import PublicHistoryStep, require_fair_sampler
from sts2_ai.emulator.exact_finite_posterior import (
    EXACT_FINITE_SEED_PRIOR_ID,
    ExactFiniteSeedPosteriorSampler,
)
from sts2_ai.emulator.particle_posterior import ParticlePosteriorExhausted
from sts2_ai.emulator.protocol import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.search.fair_replay import FairReplayPuctAdapter
from sts2_ai.search.puct import StochasticPuct

POLICY = InformationPolicy("public-v1")
REVEAL = LegalAction("reveal", "reveal")
FINISH = LegalAction("finish", "finish")


class FiniteUniverseBackend:
    def __init__(self) -> None:
        self.states: dict[str, tuple[str, int]] = {}
        self.next_handle = 0
        self.seeds: list[str] = []
        self.step_calls = 0
        self.raise_on_second_reveal = False
        self.revealed_count = 0

    @staticmethod
    def hint(seed: str) -> int:
        return hashlib.sha256(seed.encode()).digest()[0] % 2

    @staticmethod
    def outcome(seed: str) -> int:
        return hashlib.sha256(seed.encode()).digest()[1] % 2

    def _store(self, seed: str, phase: int) -> str:
        self.next_handle += 1
        handle = f"state-{self.next_handle}"
        self.states[handle] = seed, phase
        return handle

    def reset(self, seed: str, ascension: int = 0) -> str:
        assert ascension == 0
        self.seeds.append(seed)
        return self._store(seed, 0)

    def fork(self, state: str) -> str:
        return self._store(*self.states[state])

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        seed, phase = self.states[state]
        if phase == 1:
            self.revealed_count += 1
            if self.raise_on_second_reveal and self.revealed_count == 2:
                raise RuntimeError("injected observation failure")
        data: dict[str, object] = {"phase": phase}
        if phase >= 1:
            data["hint"] = self.hint(seed)
        if phase >= 2:
            data["terminal_outcome"] = (
                "victory" if self.outcome(seed) else "defeat"
            )
            data["hp"] = 10
            data["max_hp"] = 10
        payload = json.dumps(data, sort_keys=True)
        return Observation(policy.policy_id, payload, payload)

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        phase = self.states[state][1]
        return (REVEAL,) if phase == 0 else (FINISH,) if phase == 1 else ()

    def step(self, state: str, action: LegalAction) -> Transition:
        assert action in self.legal_actions(state)
        self.step_calls += 1
        seed, phase = self.states[state]
        return Transition(state, action, self._store(seed, phase + 1), phase == 1)

    def release_many(self, states: tuple[str, ...] | list[str]) -> int:
        for handle in states:
            del self.states[handle]
        return len(states)


def seed_string(number: int) -> str:
    return f"belief-pool:{number:032x}"


def history_for_seed(
    backend: FiniteUniverseBackend, seed: str
) -> tuple[PublicHistoryStep, ...]:
    current = backend.reset(seed)
    try:
        first = PublicHistoryStep(backend.observe(current, POLICY), REVEAL, (REVEAL,))
        nxt = backend.step(current, REVEAL).child
        try:
            second = PublicHistoryStep(
                backend.observe(nxt, POLICY), None, (FINISH,)
            )
        finally:
            backend.release_many([nxt])
        return first, second
    finally:
        backend.release_many([current])


def test_exact_full_history_posterior_matches_finite_prior_enumeration() -> None:
    backend = FiniteUniverseBackend()
    universe = tuple(range(96))
    public = history_for_seed(backend, seed_string(11))
    wanted_hint = json.loads(public[-1].observation.payload_json)["hint"]
    surviving = [
        number for number in universe
        if backend.hint(seed_string(number)) == wanted_hint
    ]
    assert 25 < len(surviving) < 70
    predicted_success = sum(
        backend.outcome(seed_string(number)) for number in surviving
    ) / len(surviving)

    with ExactFiniteSeedPosteriorSampler(
        backend, seed_values=universe
    ) as posterior:
        posterior.initialize_history(public)
        assert require_fair_sampler(posterior) is posterior
        assert posterior.seed_prior_id == EXACT_FINITE_SEED_PRIOR_ID
        assert posterior.prior_support_size == len(universe)
        assert posterior.posterior_support_size == len(surviving)
        assert posterior.evidence_probability == len(surviving) / len(universe)
        assert posterior.stats.simulator_steps == len(universe)
        assert posterior.public_history == public

        independent_seeds = set(backend.seeds[1:])
        assert independent_seeds == {seed_string(value) for value in universe}
        handles = posterior.sample_fair_continuations(
            public, search_rng=random.Random(831), count=2500
        )
        try:
            count_wins = sum(
                backend.outcome(backend.states[handle][0])
                for handle in handles
            )
            assert count_wins / len(handles) == pytest.approx(
                predicted_success, abs=0.045
            )
            assert all(backend.states[handle][1] == 1 for handle in handles)
            assert all(
                backend.hint(backend.states[handle][0]) == wanted_hint
                for handle in handles
            )
        finally:
            backend.release_many(handles)

        adapter = FairReplayPuctAdapter(
            backend, sampler=posterior, goal="prototype-full-victory-v1"
        )
        root = adapter.root(public, (FINISH,))
        report = StochasticPuct(adapter, seed=117, max_depth=2).search(
            root, simulations=1000
        )
        assert report.actions[0].visits == 1000
        assert report.actions[0].success_mean == pytest.approx(
            predicted_success, abs=0.06
        )
        assert not any("private-real-seed" in seed for seed in backend.seeds)
    assert not backend.states


def test_exact_prior_support_validation_and_diagnostics() -> None:
    backend = FiniteUniverseBackend()
    for values in ((), (1, 1), (-1,), (1 << 128,), (True,)):
        with pytest.raises(ValueError):
            ExactFiniteSeedPosteriorSampler(backend, seed_values=values)
    with pytest.raises(ValueError, match="max_support"):
        ExactFiniteSeedPosteriorSampler(
            backend, seed_values=tuple(range(5)), max_support=4
        )
    with ExactFiniteSeedPosteriorSampler(backend, seed_values=[3]) as sampler:
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = sampler.evidence_probability
        with pytest.raises(ValueError, match="legal-action menu"):
            sampler.initialize_history(
                (PublicHistoryStep(Observation(POLICY.policy_id, "{}", "{}"), None),)
            )
    assert not backend.states


def test_unsupported_reveal_exhausts_all_known_atoms_without_seed_fallback() -> None:
    backend = FiniteUniverseBackend()
    universe = tuple(range(20))
    source = history_for_seed(backend, seed_string(2))
    impossible = Observation(POLICY.policy_id, '{"phase":1,"hint":99}', "fake")
    transcript = (
        source[0], PublicHistoryStep(impossible, None, (FINISH,))
    )
    sampler = ExactFiniteSeedPosteriorSampler(backend, seed_values=universe)
    with pytest.raises(ParticlePosteriorExhausted):
        sampler.initialize_history(transcript)
    assert sampler.stats.surviving == 0
    assert not backend.states
    sampler.close()


def test_partial_observation_failure_releases_already_accepted_particles() -> None:
    backend = FiniteUniverseBackend()
    universe = tuple(range(8))
    # The first atom always matches its own revealed public frame.
    source = history_for_seed(backend, seed_string(universe[0]))
    backend.revealed_count = 0
    backend.raise_on_second_reveal = True
    sampler = ExactFiniteSeedPosteriorSampler(backend, seed_values=universe)
    with pytest.raises(RuntimeError, match="injected observation failure"):
        sampler.initialize_history(source)
    # Before the fix, the already accepted first candidate was leaked.
    assert not backend.states
    sampler.close()


def test_public_reveal_not_observed_from_real_hidden_seed() -> None:
    backend = FiniteUniverseBackend()
    public = history_for_seed(backend, "private-run-outside-support")
    chosen_hint = json.loads(public[1].observation.payload_json)["hint"]
    with ExactFiniteSeedPosteriorSampler(
        backend, seed_values=tuple(range(35))
    ) as posterior:
        posterior.initialize_history(public)
        samples = posterior.sample_fair_continuations(
            public, search_rng=random.Random(4), count=40
        )
        try:
            assert all(
                backend.hint(backend.states[handle][0]) == chosen_hint
                for handle in samples
            )
        finally:
            backend.release_many(samples)
    assert not backend.states
