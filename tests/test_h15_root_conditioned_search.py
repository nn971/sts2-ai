"""M16 root-conditioned H8 combat PUCT provenance and public search contract."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from sts2_ai.emulator.chance import PublicHistoryStep  # noqa: E402
from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID  # noqa: E402
from sts2_ai.emulator.protocol import (  # noqa: E402
    InformationPolicy, LegalAction, Observation, Transition,
)
from sts2_ai.search.combat_puct import (  # noqa: E402
    CombatPuct, ROOT_CONDITIONED_COMBAT_CAPABILITY,
)
from sts2_ai.search.h15_public_combat_adapter import (  # noqa: E402
    H15RootConditionedCombatAdapter,
)
from sts2_ai.training.h15_search_supervision import (  # noqa: E402
    decode_target, target_record,
)

ACTIONS = (
    LegalAction("safe", "end_turn", "{}"),
    LegalAction("risk", "end_turn", "{}"),
)


class ForkableToyH8:
    emulator_revision = "replay-reference-v1"
    mutable_transition_capability_id = "h15-h8-inplace-player-step-v1"

    def __init__(self):
        self.handles = {"source": (False, "")}
        self.forks = 0
        self.released = []

    def observe(self, handle, policy):
        assert isinstance(policy, InformationPolicy)
        is_terminal, chosen = self.handles[handle]
        if is_terminal:
            obj = {
                "combat": None, "hp": 50 if chosen == "safe" else 15,
                "max_hp": 70, "terminal_outcome": "victory",
            }
        else:
            obj = {
                "act": 1, "floor": 6, "hp": 50, "max_hp": 70,
                "combat": {
                    "turn": 9, "energy": 2, "hand": [],
                    "enemies": [{"instance_id": 1, "hp": 8}],
                },
            }
        return Observation(FAIR_POLICY_ID, json.dumps(obj), "visible:" + json.dumps(obj))

    def legal_actions(self, handle):
        return () if self.handles[handle][0] else ACTIONS

    def fork_reference(self, handle):
        self.forks += 1
        cloned = f"fork-{self.forks}"
        self.handles[cloned] = self.handles[handle]
        return cloned

    def step(self, handle, action):
        assert handle != "source"  # search never mutates the real player
        assert action in self.legal_actions(handle)
        self.handles[handle] = (True, action.action_id)
        return Transition(handle, action, handle, True)

    def release_many(self, handles):
        for handle in handles:
            self.released.append(handle)
            del self.handles[handle]
        return len(handles)


def test_root_conditioned_puct_uses_h8_forks_and_tags_policy_targets():
    backend = ForkableToyH8()
    adapter = H15RootConditionedCombatAdapter(backend, "source")
    obs = backend.observe("source", InformationPolicy(FAIR_POLICY_ID))
    root = adapter.root((PublicHistoryStep(obs, None, ACTIONS),))
    result = CombatPuct(
        adapter, lambda *_: ((0.0, 0.0), 0.5),
        leaf_mode="terminal", seed=10,
    ).search(root, simulations=32)
    assert result.chance_capability == ROOT_CONDITIONED_COMBAT_CAPABILITY
    assert result.terminal_evaluations == 32
    assert result.neural_cutoffs == 0
    assert result.visits[0] > result.visits[1]
    assert backend.forks == 32
    assert backend.handles == {"source": (False, "")}
    assert len(backend.released) == 32

    row = target_record(
        root, result, terminal_reward=1.5,
        search_model_sha256="a" * 64, source_seed="source:boss",
        emulator_revision=backend.emulator_revision,
    )
    assert row["chance_capability"] == ROOT_CONDITIONED_COMBAT_CAPABILITY
    assert sum(decode_target(row).visits) == 32


def test_root_conditioned_backend_refuses_stale_public_root():
    backend = ForkableToyH8()
    adapter = H15RootConditionedCombatAdapter(backend, "source")
    obs = backend.observe("source", InformationPolicy(FAIR_POLICY_ID))
    root = adapter.root((PublicHistoryStep(obs, None, ACTIONS),))
    backend.handles["source"] = (True, "safe")
    with pytest.raises(ValueError, match="changed"):
        adapter.begin_simulation(root, __import__("random").Random(1))
    assert backend.forks == 0
