"""PUCT search horizon counts real actions, not virtual card picks."""
from __future__ import annotations

from sts2_ai.emulator import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.search.combat_puct import (
    CombatFrame, CombatPuct, H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY,
)


class VirtualSelectionChain:
    fair_combat_capability_id = H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY

    def __init__(self, selections: int = 5) -> None:
        self.selections = selections

    def frame(self, index: int) -> CombatFrame:
        terminal = index == self.selections + 1
        action = None if terminal else (
            LegalAction(f"pick-{index}", "select_cards", '{"CardInstanceIds":[1]}')
            if index < self.selections else LegalAction("end", "end_turn", "{}")
        )
        return CombatFrame(
            information_key=f"virtual-{index}",
            observation=Observation(FAIR_POLICY_ID, "{}", f"hash-{index}"),
            legal_actions=() if terminal else (action,),
            terminal_reward=2.0 if terminal else None,
        )

    def begin_simulation(self, root, search_rng) -> None:
        assert root.information_key == "virtual-0"

    def step(self, frame, action, search_rng) -> CombatFrame:
        index = int(frame.information_key.rsplit("-", 1)[-1])
        return self.frame(index + 1)

    def end_simulation(self) -> None:
        pass


def test_compact_select_five_uses_one_real_game_action_depth():
    sampler = VirtualSelectionChain(selections=5)
    root = sampler.frame(0)
    evaluator = lambda state, actions: ((0.0,) * len(actions), 0.25)
    compact = CombatPuct(
        sampler, evaluator, max_depth=1, tree_expansion_mode="full_depth",
        free_card_selection_steps=True,
    ).search(root, simulations=2)
    assert compact.transitions == 12
    assert compact.terminal_evaluations == 2
    assert compact.root_mean_value == 1.0

    baseline = CombatPuct(
        sampler, evaluator, max_depth=1, tree_expansion_mode="full_depth",
    ).search(root, simulations=2)
    assert baseline.transitions == 2
    assert baseline.neural_cutoffs == 2
    assert baseline.root_mean_value == 0.25
