"""Pinned real-emulator smoke for the seed-independent public-history sampler."""
from __future__ import annotations

import shutil

import pytest

from sts2_ai.emulator import (
    FAIR_POLICY_ID,
    InformationPolicy,
    JsonlEmulatorBackend,
)
from sts2_ai.emulator.chance import PublicHistoryStep
from sts2_ai.emulator.rejection import FairHistoryRejectionSampler


def test_pinned_bridge_accepts_public_run_start_without_live_rng() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("The optional pinned-emulator integration needs dotnet")

    with JsonlEmulatorBackend(build=True) as backend:
        policy = InformationPolicy(FAIR_POLICY_ID)
        live_a = backend.reset("private-actual-seed-one")
        live_b = backend.reset("private-actual-seed-two")
        try:
            initial = backend.observe(live_a, policy)
            assert initial == backend.observe(live_b, policy)
            sampler = FairHistoryRejectionSampler(backend, max_candidates=16)
            import random

            draws = sampler.sample_fair_continuations(
                (PublicHistoryStep(initial, None, tuple(backend.legal_actions(live_a))),),
                search_rng=random.Random(314159),
                count=8,
            )
            assert sampler.last_stats.accepted == 8
            assert sampler.last_stats.rejected == 0
            next_observations: list[str] = []
            for handle in draws:
                try:
                    assert backend.observe(handle, policy) == initial
                    actions = backend.legal_actions(handle)
                    assert len(actions) == 1
                    child = backend.step(handle, actions[0]).child
                    try:
                        next_observations.append(backend.observe(child, policy).payload_json)
                    finally:
                        backend.release_many([child])
                finally:
                    backend.release_many([handle])
            # Independent synthetic seeds produce distinct visible post-start maps.
            assert len(set(next_observations)) > 1
        finally:
            backend.release_many([live_a, live_b])
