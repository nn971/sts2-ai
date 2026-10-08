"""Pinned native-shaped prototype test: condition jointly through first combat.

The source run is in the *declared experimental factorized stream prior*.
Only its publicly exposed observations, selected actions and legal menus
are passed into the sampler; no real-game hidden RNG or exact state is used.
"""
from __future__ import annotations

import random
import shutil

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.coupled_factorized_rejection import (
    CoupledFactorizedHistoryRejectionSampler,
)
from sts2_ai.emulator.factorized_runstart import FactorizedRunStartPosteriorSampler
from sts2_ai.emulator.incremental_coupled_particles import (
    IncrementalCoupledParticlePosterior,
)
from sts2_ai.emulator.rejection import HistoryConditioningExhausted


def test_joint_conditioning_from_visible_map_to_opening_combat() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned JSONL bridge needs dotnet")
    initial_streams = {
        "map": 11, "combat": 29, "combat_targets": 5,
        "reward": 7, "shop": 9, "event": 13,
    }
    with JsonlEmulatorBackend(build=True) as backend:
        policy = InformationPolicy(FAIR_POLICY_ID)
        source = backend.reset_factorized_hypothetical(initial_streams)
        owned = [source]
        try:
            first_action = backend.legal_actions(source)[0]
            first = PublicHistoryStep(
                backend.observe(source, policy),
                first_action,
                tuple(backend.legal_actions(source)),
            )
            map_state = backend.step(source, first_action).child
            owned.append(map_state)
            options = tuple(backend.legal_actions(map_state))
            selected = next(
                option for option in options if option.kind == "choose_map_node"
            )
            second = PublicHistoryStep(
                backend.observe(map_state, policy), selected, options
            )
            combat_state = backend.step(map_state, selected).child
            owned.append(combat_state)
            third = PublicHistoryStep(
                backend.observe(combat_state, policy),
                None, tuple(backend.legal_actions(combat_state)),
            )
            history = (first, second, third)
            with FactorizedRunStartPosteriorSampler(
                backend,
                map_states=(11, 12, 13),
                combat_states=(29,),
            ) as factorized:
                sampler = CoupledFactorizedHistoryRejectionSampler(
                    backend, runstart_sampler=factorized, max_candidates=500
                )
                sampled = sampler.sample_fair_continuations(
                    history, search_rng=random.Random(440), count=12
                )
                try:
                    assert sampler.last_stats.accepted == 12
                    assert sampler.last_stats.candidates <= 500
                    assert factorized.stats.inspected_run_starts == 4
                    for handle in sampled:
                        assert backend.observe(handle, policy) == third.observation
                        assert tuple(backend.legal_actions(handle)) == third.legal_actions
                    assert len({backend.exact_hash(h) for h in sampled}) >= 1
                finally:
                    backend.release_many(sampled)
                altered = PublicHistoryStep(
                    third.observation.__class__(
                        third.observation.policy_id,
                        '{"unreachable":"combat"}',
                        "fake-observation",
                    ), None, third.legal_actions
                )
                with pytest.raises(HistoryConditioningExhausted):
                    CoupledFactorizedHistoryRejectionSampler(
                        backend, runstart_sampler=factorized,
                        max_candidates=4
                    ).sample_fair_continuations(
                        (first, second, altered),
                        search_rng=random.Random(9),
                        count=1,
                    )
        finally:
            backend.release_many(owned)


def test_incremental_joint_particles_in_real_prototype_combat() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned JSONL bridge needs dotnet")

    prior = {
        "map": 11, "combat": 29,
        "combat_targets": 5, "reward": 7, "shop": 9, "event": 13,
    }
    with JsonlEmulatorBackend(build=True) as backend:
        policy = InformationPolicy(FAIR_POLICY_ID)
        actual_root = backend.reset_factorized_hypothetical(prior)
        owned = [actual_root]
        try:
            start = backend.legal_actions(actual_root)[0]
            public_root = PublicHistoryStep(
                backend.observe(actual_root, policy), start,
                tuple(backend.legal_actions(actual_root))
            )
            actual_map = backend.step(actual_root, start).child
            owned.append(actual_map)
            map_actions = tuple(backend.legal_actions(actual_map))
            selected = next(a for a in map_actions if a.kind == "choose_map_node")
            public_map = PublicHistoryStep(
                backend.observe(actual_map, policy), None, map_actions
            )
            with FactorizedRunStartPosteriorSampler(
                backend, map_states=(11, 12, 13), combat_states=(29,)
            ) as exact_start:
                with IncrementalCoupledParticlePosterior(
                    backend, runstart_sampler=exact_start
                ) as belief:
                    belief.initialize(
                        (public_root, public_map),
                        search_rng=random.Random(7), cohort_size=48
                    )
                    before = belief.stats
                    assert before.initial_particles == 48
                    assert before.surviving_particles == 48
                    assert exact_start.stats.inspected_run_starts == 4

                    actual_combat = backend.step(actual_map, selected).child
                    owned.append(actual_combat)
                    public_combat = backend.observe(actual_combat, policy)
                    combat_actions = tuple(backend.legal_actions(actual_combat))
                    belief.advance(selected, public_combat, combat_actions)
                    assert belief.stats.observed_transitions == 1
                    assert belief.stats.simulator_transitions == 48
                    assert belief.stats.surviving_particles > 0

                    end_turn = next(a for a in combat_actions if a.kind == "end_turn")
                    actual_next = backend.step(actual_combat, end_turn).child
                    owned.append(actual_next)
                    next_observation = backend.observe(actual_next, policy)
                    next_menu = tuple(backend.legal_actions(actual_next))
                    earlier_survivors = belief.stats.surviving_particles
                    belief.advance(end_turn, next_observation, next_menu)
                    assert belief.stats.observed_transitions == 2
                    assert belief.stats.simulator_transitions == 48 + earlier_survivors
                    assert belief.stats.surviving_particles > 0

                    history = belief.public_history
                    assert history is not None and len(history) == 4
                    handles = belief.sample_fair_continuations(
                        history, search_rng=random.Random(22), count=16
                    )
                    try:
                        assert all(
                            backend.observe(handle, policy) == next_observation
                            for handle in handles
                        )
                        assert all(
                            tuple(backend.legal_actions(handle)) == next_menu
                            for handle in handles
                        )
                    finally:
                        backend.release_many(handles)
        finally:
            backend.release_many(owned)
