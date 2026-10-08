"""Proof-of-correctness tests for fair replay-conditioned emulator trajectories."""
from __future__ import annotations

import hashlib
import json
import random

import pytest

from sts2_ai.emulator import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.emulator.chance import PublicHistoryStep, require_fair_sampler
from sts2_ai.emulator.rejection import (
    FairHistoryRejectionSampler,
    HistoryConditioningExhausted,
)
from sts2_ai.search.fair_replay import FairReplayPuctAdapter
from sts2_ai.search.puct import StochasticPuct

POLICY = InformationPolicy("test-fair-v1")
REVEAL = LegalAction("reveal", "reveal")
FINISH = LegalAction("finish", "finish")


class ScriptedHiddenBackend:
    """Seed-dependent hidden coin; all public observations omit the seed.

    The hidden bit is correlated across actions in the same trajectory.
    """

    def __init__(self) -> None:
        self.states: dict[str, tuple[str, int]] = {}
        self.next_id = 0
        self.seeds: list[str] = []

    def _store(self, seed: str, phase: int) -> str:
        self.next_id += 1
        key = f"handle-{self.next_id}"
        self.states[key] = seed, phase
        return key

    @staticmethod
    def bit(seed: str) -> int:
        return hashlib.sha256(seed.encode()).digest()[0] % 2

    def reset(self, seed: str, ascension: int = 0) -> str:
        assert ascension == 0
        self.seeds.append(seed)
        return self._store(seed, 0)

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        seed, phase = self.states[state]
        data: dict[str, object] = {"phase": phase, "hp": 10, "max_hp": 10}
        if phase >= 1:
            data["hint"] = "even" if self.bit(seed) == 0 else "odd"
        if phase == 2:
            data["terminal_outcome"] = "victory" if self.bit(seed) == 0 else "defeat"
        raw = json.dumps(data, sort_keys=True)
        return Observation(policy.policy_id, raw, hashlib.sha256(raw.encode()).hexdigest())

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        phase = self.states[state][1]
        return (REVEAL,) if phase == 0 else (FINISH,) if phase == 1 else ()

    def step(self, state: str, action: LegalAction) -> Transition:
        seed, phase = self.states[state]
        assert action in self.legal_actions(state)
        child = self._store(seed, phase + 1)
        return Transition(state, action, child, phase == 1)

    def release_many(self, states: list[str] | tuple[str, ...]) -> int:
        for state in states:
            del self.states[state]
        return len(states)


def test_sampling_is_seed_blind_and_independent_of_real_hidden_seed() -> None:
    backend = ScriptedHiddenBackend()
    state1, state2 = backend.reset("real-run-a"), backend.reset("real-run-b")
    obs1, obs2 = backend.observe(state1, POLICY), backend.observe(state2, POLICY)
    assert obs1 == obs2
    backend.release_many([state1, state2])
    sampler = FairHistoryRejectionSampler(backend, max_candidates=400)
    assert require_fair_sampler(sampler) is sampler

    history = (PublicHistoryStep(obs1, None),)
    a = sampler.sample_fair_continuations(history, search_rng=random.Random(99), count=64)
    outcomes_a = [backend.bit(backend.states[handle][0]) for handle in a]
    backend.release_many(a)
    b = sampler.sample_fair_continuations(history, search_rng=random.Random(99), count=64)
    outcomes_b = [backend.bit(backend.states[handle][0]) for handle in b]
    backend.release_many(b)
    assert outcomes_a == outcomes_b
    assert abs(sum(outcomes_a) / len(outcomes_a) - 0.5) < 0.2
    assert all(seed.startswith("fair-sampling:") for seed in backend.seeds[2:])
    assert len(backend.states) == 0


def test_rejection_conditions_on_public_history_and_preserves_hidden_correlations() -> None:
    backend = ScriptedHiddenBackend()
    seed = next(f"run-{n}" for n in range(100) if backend.bit(f"run-{n}") == 0)
    initial = backend.reset(seed)
    root_obs = backend.observe(initial, POLICY)
    after = backend.step(initial, REVEAL).child
    observed_hint = backend.observe(after, POLICY)
    backend.release_many([initial, after])
    history = (
        PublicHistoryStep(root_obs, REVEAL),
        PublicHistoryStep(observed_hint, None),
    )
    sampler = FairHistoryRejectionSampler(backend, max_candidates=250)
    handles = sampler.sample_fair_continuations(history, search_rng=random.Random(13), count=30)
    assert sampler.last_stats.rejected > 0
    assert sampler.last_stats.acceptance_rate < 1.0
    for handle in handles:
        assert backend.observe(handle, POLICY) == observed_hint
        seed_at_handle, phase = backend.states[handle]
        assert phase == 1
        assert backend.bit(seed_at_handle) == 0
        terminal = backend.step(handle, FINISH).child
        assert "victory" in backend.observe(terminal, POLICY).payload_json
        backend.release_many([terminal])
    backend.release_many(handles)
    assert not backend.states


def test_exhausted_conditioning_releases_all_candidates() -> None:
    backend = ScriptedHiddenBackend()
    initial = backend.reset("original")
    impossible = Observation(POLICY.policy_id, '{"phase":99}', "impossible")
    sampler = FairHistoryRejectionSampler(backend, max_candidates=5)
    with pytest.raises(HistoryConditioningExhausted, match="0/5"):
        sampler.sample_fair_continuations(
            (PublicHistoryStep(impossible, None),),
            search_rng=random.Random(3), count=1,
        )
    backend.release_many([initial])
    assert not backend.states
    assert sampler.last_stats.candidates == 5


def test_fair_puct_runs_with_one_latent_trajectory_per_simulation() -> None:
    backend = ScriptedHiddenBackend()
    live = backend.reset("private-real-seed")
    root_obs = backend.observe(live, POLICY)
    root_actions = backend.legal_actions(live)
    backend.release_many([live])
    adapter = FairReplayPuctAdapter(
        backend, max_candidates=100, goal="prototype-full-victory-v1"
    )
    root = adapter.root([PublicHistoryStep(root_obs, None)], root_actions)
    result = StochasticPuct(adapter, seed=45, max_depth=4).search(
        root, simulations=200
    )
    assert result.simulations == 200
    assert result.sampled_transitions == 400
    assert result.actions[0].visits == 200
    assert result.actions[0].success_mean == pytest.approx(0.5, abs=0.11)
    assert result.actions[0].hp_mean == 1.0
    assert not backend.states


def test_deep_distinctive_reveal_exhausts_without_oracle_fallback() -> None:
    backend = ScriptedHiddenBackend()
    live = backend.reset("real")
    obs = backend.observe(live, POLICY)
    progressed = backend.step(live, REVEAL).child
    reveal = backend.observe(progressed, POLICY)
    backend.release_many([live, progressed])
    adapter = FairReplayPuctAdapter(backend, max_candidates=1)
    node = adapter.root(
        [PublicHistoryStep(obs, REVEAL), PublicHistoryStep(reveal, None)],
        (FINISH,),
    )
    # Rejection either succeeds with a legitimate matching draw or fails
    # explicitly; retry at most one independent candidate.
    try:
        report = StochasticPuct(adapter, seed=7, max_depth=2).search(
            node, simulations=1
        )
        assert report.simulations == 1
    except HistoryConditioningExhausted:
        pass
    assert not backend.states
