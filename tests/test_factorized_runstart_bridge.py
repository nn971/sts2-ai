"""Pinned reference enumeration for factorized map/combat initial stream priors.

This checks the mathematical factorization against FULL Cartesian enumeration
of a small alternate research prior, using only player-visible observations.
The sampler itself must examine |map|+|combat| candidates, never |map|*|combat|.
"""
from __future__ import annotations

import random
import shutil

import pytest

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, JsonlEmulatorBackend
from sts2_ai.emulator.chance import PublicHistoryStep, require_fair_sampler
from sts2_ai.emulator.factorized_runstart import (
    FACTORIZED_RUNSTART_PRIOR_ID,
    FactorizedRunStartPosteriorSampler,
    FactorizedRunStartUnavailable,
)
from sts2_ai.search.fair_replay import FairReplayPuctAdapter

MAP_ATOMS = (3, 11, 19, 27, 35, 43, 51)
COMBAT_ATOMS = tuple(range(12, 40))


def streams(map_atom: int, combat_atom: int) -> dict[str, int]:
    return {
        "map": map_atom, "combat": combat_atom,
        "combat_targets": 0, "reward": 0, "shop": 0, "event": 0,
    }


def start(
    backend: JsonlEmulatorBackend, initial: dict[str, int]
) -> tuple[tuple[PublicHistoryStep, PublicHistoryStep], tuple[str, str]]:
    policy = InformationPolicy(FAIR_POLICY_ID)
    root = backend.reset_factorized_hypothetical(initial)
    child: str | None = None
    try:
        start_action = backend.legal_actions(root)[0]
        assert start_action.kind == "start_run"
        before = PublicHistoryStep(
            backend.observe(root, policy), start_action,
            tuple(backend.legal_actions(root)),
        )
        child = backend.step(root, start_action).child
        after = PublicHistoryStep(
            backend.observe(child, policy), None,
            tuple(backend.legal_actions(child)),
        )
        return (before, after), (root, child)
    finally:
        if child is not None:
            backend.release_many((child,))
        backend.release_many((root,))


def test_factorized_runstart_matches_exhaustive_joint_enumeration() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned emulator integration requires dotnet")

    with JsonlEmulatorBackend(build=True) as backend:
        assert backend.factorized_initial_stream_schema == (
            "prototype-independent-initial-streams-v1"
        )
        observed, _ = start(backend, streams(MAP_ATOMS[3], COMBAT_ATOMS[9]))
        # Exhaust the full finite map x combat joint prior as a test oracle.
        expected_pairs: set[tuple[int, int]] = set()
        for map_atom in MAP_ATOMS:
            for combat_atom in COMBAT_ATOMS:
                candidate, _ = start(backend, streams(map_atom, combat_atom))
                if candidate == observed:
                    expected_pairs.add((map_atom, combat_atom))
        assert expected_pairs

        with FactorizedRunStartPosteriorSampler(
            backend, map_states=MAP_ATOMS, combat_states=COMBAT_ATOMS
        ) as posterior:
            posterior.initialize(observed)
            assert require_fair_sampler(posterior) is posterior
            assert posterior.seed_prior_id == FACTORIZED_RUNSTART_PRIOR_ID
            stats = posterior.stats
            assert stats.inspected_run_starts == len(MAP_ATOMS) + len(COMBAT_ATOMS)
            assert stats.posterior_pairs == len(expected_pairs)
            assert stats.evidence_probability == (
                len(expected_pairs) / (len(MAP_ATOMS) * len(COMBAT_ATOMS))
            )
            assert posterior.public_history == observed
            assert stats.posterior_pairs >= 1
            adapter = FairReplayPuctAdapter(backend, sampler=posterior)
            root = adapter.root(observed, observed[-1].legal_actions or ())
            assert root.information_key.startswith("public-history-v1:")

            samples = posterior.sample_fair_continuations(
                observed, search_rng=random.Random(151), count=60
            )
            try:
                for state in samples:
                    assert backend.observe(
                        state, InformationPolicy(FAIR_POLICY_ID)
                    ) == observed[-1].observation
                    assert tuple(backend.legal_actions(state)) == observed[-1].legal_actions
                # Fresh independent non-map/reward/combat_target/event states
                # change exact hidden states but not the visible starting map.
                assert len({backend.exact_hash(s) for s in samples}) > 1
            finally:
                backend.release_many(samples)


def test_factorized_input_validation_and_history_length_gate() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned emulator integration requires dotnet")
    with JsonlEmulatorBackend(build=True) as backend:
        with pytest.raises(ValueError, match="six named"):
            backend.reset_factorized_hypothetical({"map": 1})
        with pytest.raises(ValueError, match="64-bit"):
            backend.reset_factorized_hypothetical(
                streams(0, 1) | {"reward": 1 << 64}
            )
        with pytest.raises(ValueError, match="repeated"):
            FactorizedRunStartPosteriorSampler(
                backend, map_states=(1, 1), combat_states=(2,)
            )
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            FactorizedRunStartPosteriorSampler(
                backend, map_states=(True,), combat_states=(2,)
            )

        public, _ = start(backend, streams(11, 21))
        with FactorizedRunStartPosteriorSampler(
            backend, map_states=(11,), combat_states=(21,)
        ) as posterior:
            with pytest.raises(ValueError, match="longer coupled histories"):
                posterior.initialize(public + (public[1],))
            with pytest.raises(ValueError, match="Requested history differs"):
                posterior.sample_fair_continuations(
                    public, search_rng=random.Random(0), count=1
                )
            posterior.initialize(public)
            assert posterior.stats.posterior_pairs == 1
            with pytest.raises(ValueError, match="Requested history differs"):
                posterior.sample_fair_continuations(
                    public[:1], search_rng=random.Random(0), count=1
                )
            with pytest.raises(ValueError, match="positive"):
                posterior.sample_fair_continuations(
                    public, search_rng=random.Random(0), count=0
                )
        with pytest.raises(FactorizedRunStartUnavailable):
            with FactorizedRunStartPosteriorSampler(
                backend, map_states=(2,), combat_states=(21,)
            ) as unavailable:
                unavailable.initialize(public)
