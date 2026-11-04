"""Opt-in H16 PUCT exact-prefix cache preserves visits/teachers on CPU toy Native."""
from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.protocol import LegalAction, Observation, StepFrame, Transition  # noqa: E402
from sts2_ai.search.combat_puct import CombatFrame, CombatPuct  # noqa: E402
from sts2_ai.training.h16_joint_act_pilot import (  # noqa: E402
    NativeRootConditionedSampler,
)

ROOT_ACTIONS = (
    LegalAction("strike", "play_card", "{}"),
    LegalAction("block", "play_card", "{}"),
)
NEXT_ACTIONS = (LegalAction("end", "end_turn", "{}"),)
AFTER_ACTIONS = (LegalAction("leave", "leave_reward", "{}"),)
PUBLIC_POLICY_ID = "prototype-fair-v0"


class FrozenNativeBackend:
    def __init__(self):
        self.states: dict[str, tuple[str, ...]] = {"source": ()}
        self.next_id = 0
        self.steps = 0
        self.forks = 0
        self.released = 0

    def _new(self, path):
        self.next_id += 1
        handle = f"state-{self.next_id}"
        self.states[handle] = path
        return handle

    def fork(self, state):
        self.forks += 1
        return self._new(self.states[state])

    def exact_continuation_receipt(self, state):
        path = self.states[state]
        digest = lambda suffix: hashlib.sha256(
            json.dumps([path, suffix]).encode()
        ).hexdigest()
        return {
            "exactHash": digest("run-state"),
            "runStochasticHash": digest("rng-state"),
            "worldStochasticHash": digest("world-stochastic"),
            "combatNativeRngHash": None,
            "referenceNaturePresent": False,
            "semanticOtherChancePresent": False,
            "semanticRewardAuditPresent": False,
        }

    def observe(self, state, policy):
        assert policy.policy_id == PUBLIC_POLICY_ID
        path = self.states[state]
        result = {
            "hp": 70, "max_hp": 70,
            "act": 1, "floor": 1,
            "combat": (
                {"turn": 1, "enemies": [{"hp": max(0, 10 - 2 * len(path)),
                                        "max_hp": 10}]}
                if len(path) < 2 else None
            ),
        }
        encoded = json.dumps(result, sort_keys=True)
        return Observation(
            PUBLIC_POLICY_ID, encoded,
            hashlib.sha256(encoded.encode()).hexdigest(),
        )

    def legal_actions(self, state):
        depth = len(self.states[state])
        return ROOT_ACTIONS if depth == 0 else NEXT_ACTIONS if depth == 1 else AFTER_ACTIONS

    def batch_rollout_step_frame(self, items, policy):
        frames = []
        for handle, action in items:
            assert action in self.legal_actions(handle)
            self.steps += 1
            child = self._new((*self.states[handle], action.action_id))
            frames.append(StepFrame(
                Transition(handle, action, child, False),
                self.observe(child, policy),
                tuple(self.legal_actions(child)),
            ))
        return tuple(frames)

    def release_many(self, handles):
        for handle in handles:
            assert handle in self.states
            assert handle != "source"
            del self.states[handle]
            self.released += 1
        return len(handles)


class FrozenPolicy:
    combat_credit_weight = 0.25
    loss_progress_cap = 0.2

    def value_at(self, *args, **kwargs):
        return 0.5


def run(simulations, depth, cache_handles):
    backend = FrozenNativeBackend()
    policy = FrozenPolicy()
    root_obs = backend.observe(
        "source", type("P", (), {"policy_id": PUBLIC_POLICY_ID})(),
    )
    root = CombatFrame("native-root", root_obs, ROOT_ACTIONS)
    adapter = NativeRootConditionedSampler(
        backend, "source", policy,
        p2_prefix_cache_handles=cache_handles,
    )
    try:
        result = CombatPuct(
            adapter, lambda obs, menu: (tuple(0.0 for _ in menu), 0.5),
            max_depth=depth,
            tree_expansion_mode="full_depth", leaf_mode="value",
            seed=20261104, free_card_selection_steps=True,
        ).search(root, simulations=simulations)
        metrics = adapter.prefix_cache_metrics()
    finally:
        adapter.close()
    assert backend.states == {"source": ()}
    return result, backend.steps, metrics


@pytest.mark.parametrize("simulations", [8, 16, 64])
@pytest.mark.parametrize("depth", [8, None])
def test_exact_cached_puct_preserves_teacher_and_reduces_real_steps(
    simulations, depth,
):
    ordinary, ordinary_steps, _ = run(simulations, depth, 0)
    cached, cached_steps, metrics = run(simulations, depth, 8)
    assert cached.visits == ordinary.visits
    assert cached.policy == ordinary.policy
    assert cached.mean_action_values == ordinary.mean_action_values
    assert cached.root_mean_value == ordinary.root_mean_value
    assert cached.transitions == ordinary.transitions
    assert cached.terminal_evaluations == ordinary.terminal_evaluations
    assert cached_steps < ordinary_steps
    assert metrics is not None and metrics["hits"] > 0
    assert metrics["retained_handles"] <= 8
