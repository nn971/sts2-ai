"""Synthetic H15 real-entry conditioning contract; no game-fidelity claims."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from sts2_ai.emulator.chance import (
    FairContinuationUnavailable,
    PublicHistoryStep,
    require_fair_sampler,
)
from sts2_ai.emulator.h15_experimental_posterior import (
    ExperimentalConditioningUnavailable,
    H15ExperimentalReferencePosterior,
    capture_source_public_history,
)
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation

LAW = {
    "lawId": "mock-law", "eventId": "draw-1",
    "probabilityStatus": "UNVERIFIED",
    "totalWeight": 2,
    "outcomes": [
        {"outcomeId": "card_instance:1", "weight": 1},
        {"outcomeId": "card_instance:2", "weight": 1},
    ],
}
MENU = (
    LegalAction("card", "play_card", "{}"),
    LegalAction("end", "end_turn", "{}"),
)
NEXT_MENU = (LegalAction("end2", "end_turn", "{}"),)
MODEL = "h15-native-structure-prototype-rng-candidate-unverified-v1"


def observation(name: str) -> Observation:
    payload = json.dumps({"combat": {"public": name}}, sort_keys=True)
    return Observation(FAIR_POLICY_ID, payload, "hash:" + name)


def frame(handle: str, name: str, version: int, *,
          kind: str = "player", selected=None) -> dict:
    obs = observation(name)
    menu = MENU if name == "root" else NEXT_MENU
    return {
        "h15Handle": handle, "boundaryKind": kind, "version": version,
        "candidateModelId": MODEL,
        "fairContinuationCapability": "unavailable",
        "referenceNatureSampleCount": 1 if selected is not None else 0,
        "observationPolicyId": FAIR_POLICY_ID if kind == "player" else None,
        "observationPayloadJson": obs.payload_json if kind == "player" else None,
        "observationHash": obs.observation_hash if kind == "player" else None,
        "legalActions": [
            {"actionId": a.action_id, "kind": a.kind, "payloadJson": a.payload_json}
            for a in menu
        ] if kind == "player" else None,
        "chanceLaw": LAW if kind == "draw_card" else None,
        "referenceNatureSelection": selected,
    }


class SyntheticNativeEntryBackend:
    """Source's original hidden draw is fixed; reference candidates are rekeyed."""

    def __init__(self):
        self.sessions = {}
        self.source_forks = {}
        self.released_candidates = []
        self.released_sources = []
        self.seed_history = []
        self.next_handle = 0
        self.fail_unsupported = False
        self.corrupt_receipt = False

    def h15_experimental_begin(self, source, *, independent_seed):
        assert source == "source"
        assert len(independent_seed) == 64
        assert int(independent_seed, 16) >= 0
        handle = "h15-" + str(self.next_handle)
        self.next_handle += 1
        self.sessions[handle] = (int(independent_seed, 16) % 2, "root", 0)
        self.seed_history.append(independent_seed)
        return frame(handle, "root", 0)

    def h15_experimental_player_action(self, handle, version, action_id):
        parity, name, v = self.sessions[handle]
        assert (name, v, version, action_id) == ("root", 0, 0, "card")
        self.sessions[handle] = (parity, "card-used", 1)
        return frame(handle, "card-used", 1)

    def h15_experimental_end_turn(self, handle, version, action_id):
        parity, name, v = self.sessions[handle]
        assert version == v
        assert action_id == ("end" if name == "root" else "end2")
        if self.fail_unsupported:
            raise RuntimeError("UNSUPPORTED_SEMANTIC_DECK source subset")
        self.sessions[handle] = (parity, "draw", v + 1)
        return frame(handle, "draw", v + 1, kind="draw_card")

    def h15_experimental_reference_choose(self, handle, version):
        parity, name, v = self.sessions[handle]
        assert (name, v) == ("draw", version)
        result = "even" if parity == 0 else "odd"
        selection = {
            "lawId": "wrong-law" if self.corrupt_receipt else LAW["lawId"],
            "eventId": LAW["eventId"],
            "outcomeId": LAW["outcomes"][parity]["outcomeId"],
            "outcomeIndex": parity,
            "providerId": "reference-nature-splitmix64-v1",
        }
        self.sessions[handle] = (parity, result, v + 1)
        return frame(handle, result, v + 1, selected=selection)

    def h15_experimental_release_many(self, handles):
        for handle in handles:
            assert handle in self.sessions
            self.sessions.pop(handle)
            self.released_candidates.append(handle)
        return len(handles)

    # Source-only API: exact replay of the original prototype native-structure
    # combat, with the fixed committed source draw. The source RNG is never
    # accessible to H15ExperimentalReferencePosterior.
    def fork(self, handle):
        assert handle == "source"
        self.source_forks["native-copy"] = "root"
        return "native-copy"

    def observe(self, handle, policy):
        assert policy.policy_id == FAIR_POLICY_ID
        return observation(self.source_forks[handle])

    def legal_actions(self, handle):
        return MENU if self.source_forks[handle] == "root" else NEXT_MENU

    def step(self, handle, action):
        assert handle in self.source_forks
        assert action.action_id == "end"
        self.source_forks["native-after"] = "even"
        return SimpleNamespace(child="native-after")

    def release_many(self, handles):
        for handle in handles:
            self.source_forks.pop(handle)
            self.released_sources.append(handle)
        return len(handles)


def history_from_source(backend):
    return capture_source_public_history(backend, "source", ("end",))


def test_capture_real_source_public_prefix_without_exposing_hidden_cursor():
    backend = SyntheticNativeEntryBackend()
    history = history_from_source(backend)
    assert [x.observation.observation_hash for x in history] == [
        "hash:root", "hash:even",
    ]
    assert history[0].chosen_action == MENU[1]
    assert history[1].chosen_action is None
    assert history[0].legal_actions == MENU
    assert history[1].legal_actions == NEXT_MENU
    assert backend.source_forks == {}
    assert backend.sessions == {}


def test_conditional_candidates_replay_complete_public_history_and_release():
    backend = SyntheticNativeEntryBackend()
    history = history_from_source(backend)
    posterior = H15ExperimentalReferencePosterior(
        backend, "source", max_candidates=100,
    )
    states = posterior.sample_conditioned(
        history, search_rng=random.Random(11), count=3,
    )
    assert len(states) == 3
    assert posterior.last_stats.attempted >= 3
    assert posterior.last_stats.accepted == 3
    assert posterior.last_stats.mismatched > 0
    assert posterior.last_stats.sampled_nature_events >= 3
    assert all(backend.sessions[h][1] == "even" for h in states)
    assert len(backend.sessions) == 3
    assert len(backend.released_candidates) == posterior.last_stats.mismatched
    assert len(set(backend.seed_history)) == posterior.last_stats.attempted
    with pytest.raises(FairContinuationUnavailable):
        require_fair_sampler(posterior)
    backend.h15_experimental_release_many(states)
    assert backend.sessions == {}


def test_changed_public_menu_is_rejected_even_if_hash_matches():
    backend = SyntheticNativeEntryBackend()
    history = list(history_from_source(backend))
    history[1] = PublicHistoryStep(
        history[1].observation, None, MENU,
    )
    posterior = H15ExperimentalReferencePosterior(
        backend, "source", max_candidates=5,
    )
    with pytest.raises(ExperimentalConditioningUnavailable, match="exhausted"):
        posterior.sample_conditioned(
            history, search_rng=random.Random(2), count=1,
        )
    assert posterior.last_stats.mismatched == 5
    assert backend.sessions == {}


def test_sample_exhaustion_discards_all_previously_accepted_handles():
    backend = SyntheticNativeEntryBackend()
    posterior = H15ExperimentalReferencePosterior(
        backend, "source", max_candidates=1,
    )
    # At least two accepted continuations are impossible with one proposal.
    with pytest.raises(ExperimentalConditioningUnavailable, match="exhausted"):
        posterior.sample_conditioned(
            history_from_source(backend), search_rng=random.Random(3),
            count=2,
        )
    assert backend.sessions == {}
    assert posterior.last_stats.attempted == 1
    assert len(backend.released_candidates) == 1


def test_unsupported_mechanics_abort_instead_of_selecting_covered_candidates():
    backend = SyntheticNativeEntryBackend()
    history = history_from_source(backend)
    backend.fail_unsupported = True
    posterior = H15ExperimentalReferencePosterior(
        backend, "source", max_candidates=10,
    )
    with pytest.raises(ExperimentalConditioningUnavailable, match="UNSUPPORTED"):
        posterior.sample_conditioned(
            history, search_rng=random.Random(8), count=1,
        )
    assert posterior.last_stats.attempted == 1
    assert backend.sessions == {}


def test_invalid_emulator_receipt_is_rejected_without_leaking_handles():
    backend = SyntheticNativeEntryBackend()
    history = history_from_source(backend)
    backend.corrupt_receipt = True
    posterior = H15ExperimentalReferencePosterior(backend, "source")
    with pytest.raises(ValueError, match="receipt mismatches"):
        posterior.sample_conditioned(
            history, search_rng=random.Random(8), count=1,
        )
    assert backend.sessions == {}


def test_illegal_or_incomplete_history_rejected_before_sampling():
    backend = SyntheticNativeEntryBackend()
    posterior = H15ExperimentalReferencePosterior(backend, "source")
    history = history_from_source(backend)
    with pytest.raises(ValueError, match="uncommitted"):
        posterior.sample_conditioned(
            (history[0],), search_rng=random.Random(2), count=1,
        )
    with pytest.raises(ValueError, match="illegal"):
        posterior.sample_conditioned(
            (PublicHistoryStep(
                history[0].observation,
                LegalAction("fake", "play_card"),
                history[0].legal_actions,
            ), history[1]),
            search_rng=random.Random(2), count=1,
        )
    assert backend.sessions == {}
