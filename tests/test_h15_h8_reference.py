"""H8 Reference Nature posterior: independent candidate seeds, public conditioning.

The fake transport models the actual H8 version-owned, in-place JSONL
contract. It tests ownership and conditioning, not native emulator fidelity.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from sts2_ai.emulator.chance import FairContinuationUnavailable, PublicHistoryStep
from sts2_ai.emulator.h15_h8_reference import (
    H8_REFERENCE_MODEL,
    H8FixtureEntry,
    H8ReferenceCombatBackend,
    H8ReferenceNaturePosterior,
)
from sts2_ai.evaluation.combat_snapshots import POLICY


class ToyH8Transport:
    def __init__(self):
        self.next_handle = 0
        self.statuses = {}
        self.seed_trace = []
        self.released = []

    def _status(self, handle, seed, *, version=0, outcome=None):
        status = {
            "schemaId": "sts2-h8-versioned-combat-runtime-v1",
            "h8Handle": handle, "runtimeId": "test-runtime", "boundaryVersion": version,
            "kind": "terminal" if outcome else "player",
            "outcome": outcome,
            "exitHp": None if not outcome else 35 if outcome == "victory" else 0,
            "defeatTotalNonMinionEnemyHp": 100 if outcome == "defeat" else None,
            "defeatRemainingNonMinionEnemyHp": 40 if outcome == "defeat" else None,
            "playerActionIds": ["safe", "risk"] if not outcome else [],
        }
        if outcome is None:
            status["playerFrame"] = {
                "observation": {
                    "act": 1, "floor": 2, "hp": 40, "max_hp": 70,
                    "deck": [], "relics": [], "potions": [], "map": [],
                    "combat": {
                        "hand": [], "discard_pile": [], "exhaust_pile": [],
                        "enemies": [{
                            "instance_id": 1, "enemy_id": "crawler",
                            "hp": 40, "move_id": "attack" if seed % 2 else "inhale",
                        }],
                    },
                },
                "actions": [
                    {"actionId": "safe", "kind": "end_turn", "payloadJson": "{}"},
                    {"actionId": "risk", "kind": "end_turn", "payloadJson": "{}"},
                ],
            }
        return status

    def call(self, op, **args):
        if op == "h8_begin_seeded":
            seed = args["local_reference_seed"]
            self.seed_trace.append(seed)
            handle = "h" + str(self.next_handle)
            self.next_handle += 1
            status = self._status(handle, seed)
            self.statuses[handle] = (status, seed)
            return {"result": status}
        if op == "h8_step":
            handle = args["h8_handle"]
            before, seed = self.statuses[handle]
            assert args["boundary_version"] == before["boundaryVersion"]
            assert args["local_reference_seed"] == seed
            outcome = "victory" if args["action_id"] == "safe" else "defeat"
            new = self._status(
                handle, seed, version=before["boundaryVersion"] + 1,
                outcome=outcome,
            )
            self.statuses[handle] = (new, seed)
            return {"result": new}
        if op == "h8_release_many":
            for handle in args["h8_handles"]:
                assert handle in self.statuses
                self.statuses.pop(handle)
                self.released.append(handle)
            return {"released": len(args["h8_handles"])}
        raise AssertionError("Unexpected H8 JSONL operation: " + op)


def fixture():
    return H8FixtureEntry(
        seed="fixed-geometry", encounter_id="proto.encounter.mawler_normal",
        native_entry="MawlerNormal", total_floor=2,
        fixture_deck_preset_id="silent-starter-v1",
    )


def test_h8_sampler_rejection_uses_independent_seeds_and_releases_all_candidates():
    transport = ToyH8Transport()
    backend = H8ReferenceCombatBackend(
        transport, fixture(), emulator_revision="a" * 40,
    )
    observed = backend.begin_reference(3)  # public inhale/attack encoded by parity
    frame = backend.observe(observed, POLICY)
    menu = backend.legal_actions(observed)
    sampler = H8ReferenceNaturePosterior(backend, max_candidates=200)
    accepted = sampler.sample_fair_continuations(
        (PublicHistoryStep(frame, None, menu),),
        search_rng=random.Random(100), count=3,
    )
    assert sampler.prior_id == H8_REFERENCE_MODEL
    assert len(accepted) == 3
    assert all(backend.observe(h, POLICY) == frame for h in accepted)
    assert all(backend.legal_actions(h) == menu for h in accepted)
    assert len(transport.seed_trace) > 3
    assert any(seed != 3 for seed in transport.seed_trace[1:])
    backend.release_many((*accepted, observed))
    assert backend.live_handles == 0


def test_h8_sampler_exhaustion_is_not_replaced_with_live_fork():
    backend = H8ReferenceCombatBackend(
        ToyH8Transport(), fixture(), emulator_revision="b" * 40,
    )
    source = backend.begin_reference(3)
    obs = backend.observe(source, POLICY)
    # Require impossible public HP. No candidate can match.
    data = json.loads(obs.payload_json)
    data["hp"] = 51
    impossible = type(obs)(obs.policy_id, json.dumps(
        data, sort_keys=True, separators=(",", ":")
    ), obs.observation_hash)
    sampler = H8ReferenceNaturePosterior(backend, max_candidates=9)
    with pytest.raises(FairContinuationUnavailable, match="exhausted"):
        sampler.sample_fair_continuations(
            (PublicHistoryStep(impossible, None, backend.legal_actions(source)),),
            search_rng=random.Random(42), count=1,
        )
    assert backend.live_handles == 1
    backend.release_many((source,))
    assert backend.live_handles == 0


def test_h8_native_reference_collector_has_real_outcome_and_no_handle_leaks():
    pytest.importorskip("torch")
    from sts2_ai.training.h15_native_search_collector import collect_live_combat_search
    from sts2_ai.training.h15_search_supervision import decode_target

    transport = ToyH8Transport()
    backend = H8ReferenceCombatBackend(
        transport, fixture(), emulator_revision="c" * 40,
    )
    source = backend.begin_reference(3)
    start = backend.observe(source, POLICY)
    history = (PublicHistoryStep(start, None, backend.legal_actions(source)),)
    sampler = H8ReferenceNaturePosterior(backend, max_candidates=500)
    class Teacher:
        checkpoint_sha256 = "d" * 64

        def __call__(self, observation, actions):
            return (0.0, 0.0), 0.5

    report = collect_live_combat_search(
        backend, source, history, fair_sampler=sampler, teacher=Teacher(),
        source_seed="fixture-nature-3", entry_hash=start.observation_hash,
        encounter_tier="normal", simulations=24, max_searches=1,
        max_decisions=1, tree_depth=2, rollout_depth=4,
    )
    assert report.outcome == "victory"
    assert report.reward == pytest.approx(1.5)
    assert report.combat_decisions == 1
    assert len(report.targets) == 1
    assert decode_target(report.targets[0]).realized_reward == pytest.approx(1.5)
    assert report.targets[0]["search_neural_cutoffs"] == 0
    assert backend.live_handles == 0
