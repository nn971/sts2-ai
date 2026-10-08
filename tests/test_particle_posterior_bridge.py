"""Pinned emulator integration for a finite search-seeded belief cohort."""
from __future__ import annotations

import random
import shutil

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.particle_posterior import FiniteSeedPosteriorSampler


def test_pinned_bridge_incremental_public_history_filtering_and_forks() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned bridge integration requires the dotnet SDK")

    with JsonlEmulatorBackend(build=True) as backend:
        policy = InformationPolicy(FAIR_POLICY_ID)
        live = backend.reset("unrelated-private-game-seed")
        try:
            observation = backend.observe(live, policy)
            actions = tuple(backend.legal_actions(live))
            assert len(actions) == 1 and actions[0].kind == "start_run"

            with FiniteSeedPosteriorSampler(backend) as cohort:
                cohort.initialize(
                    observation, actions,
                    search_rng=random.Random(41), cohort_size=16,
                )
                assert cohort.stats.surviving == 16
                assert cohort.stats.observed_steps == 0

                root_history = cohort.public_history
                assert root_history == (PublicHistoryStep(observation, None, actions),)
                hypothetical = cohort.sample_fair_continuations(
                    root_history, search_rng=random.Random(42), count=1,
                )[0]
                try:
                    transition = backend.step(hypothetical, actions[0])
                    try:
                        revealed = backend.observe(transition.child, policy)
                        revealed_actions = tuple(backend.legal_actions(transition.child))
                    finally:
                        backend.release_many([transition.child])
                finally:
                    backend.release_many([hypothetical])

                # The revealed hypothetical map has at least one original
                # cohort member. The conditioning operation must retain it.
                cohort.advance(actions[0], revealed, revealed_actions)
                assert 1 <= cohort.stats.surviving <= 16
                assert cohort.stats.simulator_steps == 16
                assert cohort.stats.observed_steps == 1
                history = cohort.public_history
                assert history is not None
                assert history[0].chosen_action == actions[0]
                assert history[1].observation == revealed

                forks = cohort.sample_fair_continuations(
                    history, search_rng=random.Random(43), count=12
                )
                try:
                    assert all(backend.observe(h, policy) == revealed for h in forks)
                    assert all(tuple(backend.legal_actions(h)) == revealed_actions
                               for h in forks)
                    assert cohort.stats.forks == 13
                finally:
                    backend.release_many(forks)
        finally:
            backend.release_many([live])
