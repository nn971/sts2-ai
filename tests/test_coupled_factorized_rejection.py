"""A deliberately coupled game tests rejection past factorized RunStart.

Map and boss are separate observable stream functions at RunStart, but a
later public reveal is an XOR of combat and event stream bits. Independent
post-reveal filtering would invent an incorrect posterior. The tested sampler
instead replays entire hypothetical states with the original joint cursors.
"""
from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence

import pytest

from sts2_ai.emulator.chance import PublicHistoryStep, require_fair_sampler
from sts2_ai.emulator.coupled_factorized_rejection import (
    COUPLED_FACTORIZED_PRIOR_ID,
    CoupledFactorizedHistoryRejectionSampler,
)
from sts2_ai.emulator.factorized_runstart import (
    FACTORIZED_RUNSTART_SCHEMA_ID,
    FactorizedRunStartPosteriorSampler,
)
from sts2_ai.emulator.incremental_coupled_particles import (
    INCREMENTAL_COUPLED_COHORT_PRIOR_ID,
    IncrementalCoupledParticlePosterior,
    IncrementalCoupledPosteriorExhausted,
)
from sts2_ai.emulator.protocol import InformationPolicy, LegalAction, Observation, Transition
from sts2_ai.emulator.rejection import HistoryConditioningExhausted
from sts2_ai.search.fair_replay import FairReplayPuctAdapter
from sts2_ai.search.puct import StochasticPuct

POLICY = InformationPolicy("prototype-fair-v0")
START = LegalAction("start-run", "start_run")
CHOOSE = LegalAction("first-combat", "choose_map_node")
FINISH = LegalAction("finish-combat", "end_turn")


class CoupledToyBackend:
    factorized_initial_stream_schema = FACTORIZED_RUNSTART_SCHEMA_ID

    def __init__(self) -> None:
        self._states: dict[str, tuple[dict[str, int], int]] = {}
        self._next = 0
        self.reset_calls = 0
        self.fork_calls = 0
        self.step_calls = 0
        self.fail_at_step: int | None = None

    def _store(self, data: dict[str, int], phase: int) -> str:
        self._next += 1
        handle = f"s{self._next}"
        self._states[handle] = (data.copy(), phase)
        return handle

    def reset_factorized_hypothetical(
        self, initial_streams: Mapping[str, int], ascension: int = 0
    ) -> str:
        assert ascension == 0
        assert len(initial_streams) == 6
        self.reset_calls += 1
        return self._store(dict(initial_streams), 0)

    def fork(self, state: str) -> str:
        self.fork_calls += 1
        data, phase = self._states[state]
        return self._store(data, phase)

    def observe(self, state: str, policy: InformationPolicy) -> Observation:
        data, phase = self._states[state]
        observation: dict[str, object] = {"phase": (1, 2, 3, 10)[phase]}
        if phase >= 1:
            observation["map_value"] = data["map"] & 1
            observation["act_one_boss_encounter_id"] = f"boss-{data['combat'] & 1}"
        if phase >= 2:
            observation["coupled_reveal"] = (
                (data["combat"] >> 1) & 1
            ) ^ (data["event"] & 1)
        if phase >= 3:
            observation["terminal_outcome"] = (
                "victory" if data["event"] & 1 else "defeat"
            )
            observation["hp"] = 8
            observation["max_hp"] = 10
        payload = json.dumps(observation, sort_keys=True, separators=(",", ":"))
        return Observation(policy.policy_id, payload, payload)

    def legal_actions(self, state: str) -> tuple[LegalAction, ...]:
        phase = self._states[state][1]
        return ((START,), (CHOOSE,), (FINISH,), ())[phase]

    def step(self, state: str, action: LegalAction) -> Transition:
        assert action in self.legal_actions(state)
        self.step_calls += 1
        if self.fail_at_step == self.step_calls:
            raise RuntimeError("injected transition failure")
        data, phase = self._states[state]
        return Transition(state, action, self._store(data, phase + 1), phase == 2)

    def release_many(self, states: Sequence[str]) -> int:
        for state in states:
            del self._states[state]
        return len(states)

    @property
    def active(self) -> int:
        return len(self._states)

    def latent(self, state: str) -> dict[str, int]:
        return self._states[state][0]


def initial_history(
    backend: CoupledToyBackend,
) -> tuple[PublicHistoryStep, ...]:
    sample = {
        "map": 1, "combat": 2, "event": 1,
        "reward": 0, "shop": 0, "combat_targets": 0,
    }
    state = backend.reset_factorized_hypothetical(sample)
    handles = [state]
    try:
        result: list[PublicHistoryStep] = []
        for action in (START, CHOOSE):
            result.append(PublicHistoryStep(
                backend.observe(state, POLICY), action, backend.legal_actions(state)
            ))
            state = backend.step(state, action).child
            handles.append(state)
        result.append(PublicHistoryStep(
            backend.observe(state, POLICY), None, backend.legal_actions(state)
        ))
        return tuple(result)
    finally:
        backend.release_many(handles)


def setup(
    backend: CoupledToyBackend,
) -> tuple[
    tuple[PublicHistoryStep, ...],
    FactorizedRunStartPosteriorSampler,
    CoupledFactorizedHistoryRejectionSampler,
]:
    history = initial_history(backend)
    source = FactorizedRunStartPosteriorSampler(
        backend, map_states=(1, 3, 5), combat_states=(0, 2, 4, 6)
    )
    sampler = CoupledFactorizedHistoryRejectionSampler(
        backend, runstart_sampler=source, max_candidates=4096
    )
    return history, source, sampler


def test_coupled_rejection_preserves_event_combat_correlations() -> None:
    backend = CoupledToyBackend()
    history, source, sampler = setup(backend)
    assert require_fair_sampler(sampler) is sampler
    assert sampler.seed_prior_id == COUPLED_FACTORIZED_PRIOR_ID
    assert source.public_history is None

    handles = sampler.sample_fair_continuations(
        history, search_rng=random.Random(713), count=450
    )
    try:
        assert source.stats.posterior_pairs == 12
        # Source filters map and boss using 3+4, never the 12 pairs;
        # one extra complete simulation for each candidate is needed.
        assert source.stats.inspected_run_starts == 7
        assert 0 < sampler.last_stats.acceptance_rate < 1
        assert sampler.last_stats.accepted == 450
        assert sampler.last_stats.replay_steps == sampler.last_stats.candidates
        assert all(backend.observe(h, POLICY) == history[-1].observation for h in handles)
        assert all(backend.legal_actions(h) == (FINISH,) for h in handles)
        for handle in handles:
            latent = backend.latent(handle)
            # The observed XOR was zero, so combat and event bits must AGREE.
            assert (latent["event"] & 1) == ((latent["combat"] >> 1) & 1)
        success_rate = sum(backend.latent(h)["event"] & 1 for h in handles) / len(handles)
        assert success_rate == pytest.approx(0.5, abs=0.07)
    finally:
        backend.release_many(handles)
    assert backend.active == 0


def test_deeper_fair_puct_can_use_jointly_conditioned_full_states() -> None:
    backend = CoupledToyBackend()
    history, source, sampler = setup(backend)
    with source:
        adapter = FairReplayPuctAdapter(
            backend, sampler=sampler, goal="prototype-full-victory-v1"
        )
        root = adapter.root(history, (FINISH,))
        result = StochasticPuct(adapter, seed=31, max_depth=2).search(
            root, simulations=280
        )
        assert result.actions[0].visits == 280
        assert result.actions[0].success_mean == pytest.approx(0.5, abs=0.09)
        assert result.simulations == 280
        assert result.sampled_transitions == 280
    assert backend.active == 0


def test_exhaustion_releases_all_accepted_and_rejected_handles() -> None:
    backend = CoupledToyBackend()
    history, source, _ = setup(backend)
    impossible = Observation(
        POLICY.policy_id, '{"phase":3,"coupled_reveal":99}', "impossible"
    )
    transcript = history[:2] + (
        PublicHistoryStep(impossible, None, (FINISH,)),
    )
    sampler = CoupledFactorizedHistoryRejectionSampler(
        backend, runstart_sampler=source, max_candidates=8
    )
    with pytest.raises(HistoryConditioningExhausted, match="exhausted"):
        sampler.sample_fair_continuations(
            transcript, search_rng=random.Random(29), count=1
        )
    assert sampler.last_stats.candidates == 8
    assert sampler.last_stats.accepted == 0
    assert backend.active == 0


def test_finite_budget_failure_after_accepted_samples_cleans_up() -> None:
    backend = CoupledToyBackend()
    history, source, _ = setup(backend)
    # Exactly one of two required samples can be accepted before cap.
    sampler = CoupledFactorizedHistoryRejectionSampler(
        backend, runstart_sampler=source, max_candidates=1
    )
    with pytest.raises(HistoryConditioningExhausted):
        sampler.sample_fair_continuations(
            history, search_rng=random.Random(7), count=2
        )
    assert sampler.last_stats.accepted in (0, 1)
    assert backend.active == 0


def test_wrong_prefix_bad_requests_and_transition_errors_fail_closed() -> None:
    backend = CoupledToyBackend()
    history, source, sampler = setup(backend)
    with pytest.raises(ValueError, match="positive"):
        sampler.sample_fair_continuations(
            history, search_rng=random.Random(1), count=0
        )
    with pytest.raises(ValueError, match="RunStart and MapChoice"):
        sampler.sample_fair_continuations(
            (PublicHistoryStep(
                history[0].observation, None, history[0].legal_actions
            ),), search_rng=random.Random(1), count=1
        )
    source.initialize((
        history[0], PublicHistoryStep(
            history[1].observation, None, history[1].legal_actions
        ),
    ))
    wrong = list(history)
    wrong[0] = PublicHistoryStep(
        Observation(POLICY.policy_id, '{"forged":true}', "forged"),
        START, (START,),
    )
    with pytest.raises(ValueError, match="differs from the public prefix"):
        sampler.sample_fair_continuations(
            wrong, search_rng=random.Random(1), count=1
        )
    backend.fail_at_step = backend.step_calls + 2
    # Source sampling performs one start_run step, then deeper replay will
    # throw. Rejection must release whichever handle was most recently owned.
    with pytest.raises(RuntimeError, match="injected transition failure"):
        sampler.sample_fair_continuations(
            history, search_rng=random.Random(3), count=1
        )
    assert backend.active == 0

def _incremental_prefix(
    history: tuple[PublicHistoryStep, ...],
) -> tuple[PublicHistoryStep, ...]:
    return (
        history[0],
        PublicHistoryStep(history[1].observation, None, history[1].legal_actions),
    )


def test_incremental_joint_particle_filter_and_empirical_sampling() -> None:
    backend = CoupledToyBackend()
    history, source, _ = setup(backend)
    rng = random.Random(501)
    with source, IncrementalCoupledParticlePosterior(
        backend, runstart_sampler=source
    ) as belief:
        belief.initialize(
            _incremental_prefix(history), search_rng=rng, cohort_size=600
        )
        assert require_fair_sampler(belief) is belief
        assert belief.seed_prior_id == INCREMENTAL_COUPLED_COHORT_PRIOR_ID
        assert belief.stats.initial_particles == 600
        assert belief.stats.surviving_particles == 600
        assert source.stats.inspected_run_starts == 7
        resets = backend.reset_calls

        belief.advance(CHOOSE, history[-1].observation, (FINISH,))
        assert backend.reset_calls == resets  # Only existing particles advanced
        assert belief.public_history == history
        stats = belief.stats
        assert stats.simulator_transitions == 600
        assert stats.observed_transitions == 1
        assert 200 < stats.surviving_particles < 400
        assert stats.empirical_effective_sample_size == stats.surviving_particles
        assert backend.active == stats.surviving_particles

        sampled = belief.sample_fair_continuations(
            history, search_rng=random.Random(502), count=1500
        )
        try:
            assert backend.fork_calls == 1500
            assert backend.reset_calls == resets
            assert all(backend.observe(s, POLICY) == history[-1].observation for s in sampled)
            assert all(
                backend.latent(s)["event"] & 1
                == (backend.latent(s)["combat"] >> 1) & 1
                for s in sampled
            )
            frac = sum(backend.latent(s)["event"] & 1 for s in sampled) / len(sampled)
            assert frac == pytest.approx(0.5, abs=0.1)
        finally:
            backend.release_many(sampled)
        adapter = FairReplayPuctAdapter(
            backend, sampler=belief, goal="prototype-full-victory-v1"
        )
        root = adapter.root(history, (FINISH,))
        report = StochasticPuct(adapter, seed=52, max_depth=2).search(
            root, simulations=350
        )
        assert report.actions[0].visits == 350
        assert report.actions[0].success_mean == pytest.approx(frac, abs=0.12)
        assert backend.reset_calls == resets
        assert backend.active == stats.surviving_particles
    assert backend.active == 0


def test_incremental_cohort_collapse_cleans_every_owned_handle() -> None:
    backend = CoupledToyBackend()
    history, source, _ = setup(backend)
    with source:
        belief = IncrementalCoupledParticlePosterior(
            backend, runstart_sampler=source
        )
        belief.initialize(
            _incremental_prefix(history),
            search_rng=random.Random(107), cohort_size=32
        )
        forged = Observation(
            POLICY.policy_id, '{"phase":3,"coupled_reveal":900}', "forged"
        )
        with pytest.raises(IncrementalCoupledPosteriorExhausted):
            belief.advance(CHOOSE, forged, (FINISH,))
        assert backend.active == 0
        assert belief.stats.surviving_particles == 0
        with pytest.raises(RuntimeError, match="closed"):
            belief.sample_fair_continuations(
                history, search_rng=random.Random(4), count=1
            )


def test_incremental_mid_transition_failure_releases_accepted_particles() -> None:
    backend = CoupledToyBackend()
    history, source, _ = setup(backend)
    with source:
        belief = IncrementalCoupledParticlePosterior(
            backend, runstart_sampler=source
        )
        belief.initialize(
            _incremental_prefix(history),
            search_rng=random.Random(23), cohort_size=40
        )
        backend.fail_at_step = backend.step_calls + 6
        with pytest.raises(RuntimeError, match="injected transition failure"):
            belief.advance(CHOOSE, history[-1].observation, (FINISH,))
        assert backend.active == 0
        belief.close()


def test_incremental_history_provenance_and_validation_leave_cohort_intact() -> None:
    backend = CoupledToyBackend()
    history, source, _ = setup(backend)
    with source, IncrementalCoupledParticlePosterior(
        backend, runstart_sampler=source
    ) as belief:
        with pytest.raises(ValueError, match="MapChoice only"):
            belief.initialize(history, search_rng=random.Random(1))
        belief.initialize(
            _incremental_prefix(history), search_rng=random.Random(1), cohort_size=24
        )
        before = backend.active
        with pytest.raises(ValueError, match="public legal menu"):
            belief.advance(FINISH, history[-1].observation, (FINISH,))
        with pytest.raises(ValueError, match="history differs"):
            belief.sample_fair_continuations(
                history, search_rng=random.Random(3), count=1
            )
        with pytest.raises(ValueError, match="positive"):
            belief.sample_fair_continuations(
                _incremental_prefix(history), search_rng=random.Random(3), count=0
            )
        assert backend.active == before
        assert belief.stats.observed_transitions == 0
    assert backend.active == 0
