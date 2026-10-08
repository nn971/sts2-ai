"""Pinned bridge regression for local, synthetic combat stream conditioning.

Successful full-state replays and cursor agreement are proved in the
emulator's C# tests; this suite checks the Python capability boundary and
safe rejection without attempting a long-history rare-state search.
"""
from __future__ import annotations

import random
import shutil

import pytest

from sts2_ai.emulator import (
    FAIR_POLICY_ID,
    InformationPolicy,
    JsonlBridgeError,
    JsonlEmulatorBackend,
    LegalAction,
    Observation,
)
from sts2_ai.emulator.chance import FairContinuationUnavailable, require_fair_sampler


def test_local_combat_stream_is_experimental_and_fails_closed() -> None:
    if shutil.which("dotnet") is None:
        pytest.skip("Pinned integration requires the dotnet SDK")
    with JsonlEmulatorBackend(build=True) as backend:
        assert backend.local_combat_entry_schema == (
            "prototype-local-combat-stream-condition-v1"
        )
        with pytest.raises(FairContinuationUnavailable):
            require_fair_sampler(backend)

        public_policy = InformationPolicy(FAIR_POLICY_ID)
        live = backend.reset("hidden-live-game-seed")
        synthetic = backend.reset_hypothetical(random.Random(41).getrandbits(128))
        states = [live, synthetic]
        try:
            for original, is_hypothetical in ((live, False), (synthetic, True)):
                root_action = backend.legal_actions(original)[0]
                mapped = backend.step(original, root_action).child
                states.append(mapped)
                observation = backend.observe(mapped, public_policy)
                legal_map = tuple(backend.legal_actions(mapped))
                chosen = next(
                    action for action in legal_map
                    if action.kind == "choose_map_node"
                )

                impossible = Observation(
                    FAIR_POLICY_ID,
                    '{"phase":"unreachable"}',
                    "intentionally-incorrect-public-observation-hash",
                )
                if is_hypothetical:
                    error_match = "exhausted"
                else:
                    error_match = "hypothetical"
                with pytest.raises(JsonlBridgeError, match=error_match):
                    backend.condition_local_combat_entry(
                        mapped, chosen, impossible,
                        (LegalAction("impossible-card-action", "end_turn"),),
                        search_seed=101,
                        max_candidates=2,
                    )
                assert backend.observe(mapped, public_policy) == observation
                assert tuple(backend.legal_actions(mapped)) == legal_map

                with pytest.raises(ValueError, match="128-bit"):
                    backend.condition_local_combat_entry(
                        mapped, chosen, impossible, legal_map,
                        search_seed=1 << 128,
                    )
                with pytest.raises(ValueError, match="map-node"):
                    backend.condition_local_combat_entry(
                        mapped, LegalAction("wrong", "play_card"),
                        impossible, legal_map, search_seed=1,
                    )
        finally:
            backend.release_many(states)
