"""Pinned prototype: exact finite-prior posterior after native-shaped map reveal.

The actual fixture run comes from one declared seed atom. The sampler is
given ONLY its public observations, action choices and full legal menus.
Even a highly distinctive map must retain at least its generating seed,
unlike bounded random-seed rejection, which can easily miss the same map.
"""
from __future__ import annotations

import random
import shutil

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.exact_finite_posterior import ExactFiniteSeedPosteriorSampler


def test_exact_finite_seed_posterior_after_public_map_reveal() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned emulator JSONL integration requires dotnet")
    universe = tuple(range(40))
    generating_atom = 17
    with JsonlEmulatorBackend(build=True) as backend:
        policy = InformationPolicy(FAIR_POLICY_ID)
        real = backend.reset(f"belief-pool:{generating_atom:032x}")
        handles = [real]
        try:
            history: list[PublicHistoryStep] = []
            action0 = backend.legal_actions(real)[0]
            history.append(PublicHistoryStep(
                backend.observe(real, policy), action0,
                tuple(backend.legal_actions(real)),
            ))
            mapped = backend.step(real, action0).child
            handles.append(mapped)

            actions = tuple(backend.legal_actions(mapped))
            map_choice = next(
                option for option in actions if option.kind == "choose_map_node"
            )
            history.append(PublicHistoryStep(
                backend.observe(mapped, policy), map_choice, actions,
            ))
            entered = backend.step(mapped, map_choice).child
            handles.append(entered)
            final_actions = tuple(backend.legal_actions(entered))
            public_combat = backend.observe(entered, policy)
            history.append(PublicHistoryStep(public_combat, None, final_actions))

            with ExactFiniteSeedPosteriorSampler(
                backend, seed_values=universe
            ) as sampler:
                sampler.initialize_history(history)
                assert 1 <= sampler.posterior_support_size <= len(universe)
                assert sampler.stats.observed_steps == 2
                assert sampler.stats.simulator_steps >= len(universe)
                assert sampler.public_history == tuple(history)
                assert sampler.evidence_probability == (
                    sampler.posterior_support_size / len(universe)
                )
                samples = sampler.sample_fair_continuations(
                    tuple(history), search_rng=random.Random(404), count=24
                )
                try:
                    assert all(
                        backend.observe(handle, policy) == public_combat
                        for handle in samples
                    )
                    assert all(
                        tuple(backend.legal_actions(handle)) == final_actions
                        for handle in samples
                    )
                finally:
                    backend.release_many(samples)
        finally:
            backend.release_many(handles)
