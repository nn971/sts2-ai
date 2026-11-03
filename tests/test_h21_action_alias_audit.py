"""Action aliases are permitted only when their public card classes agree."""
from __future__ import annotations

import pytest

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import ActionToken, EncodedDecision
from sts2_ai.training.h21_action_alias_audit import audit_h21_action_aliases


def action(name: str, kind: str, payload: str) -> LegalAction:
    return LegalAction(name, kind, payload)


def duplicate_tokens(*, combat: bool = False) -> EncodedDecision:
    shared = ActionToken(kind=1, semantic=7, source=-1, target=4)
    return EncodedDecision(
        "sts2-h21-public-grounded-macro-v1",
        "combat" if combat else "macro",
        (), (shared, shared),
    )


def test_two_identical_persistent_copies_may_share_a_remove_action():
    state = {"deck": [
        {"instance_id": 10, "card_id": "strike", "state": {}, "upgrade_level": 0},
        {"instance_id": 11, "card_id": "strike", "state": {}, "upgrade_level": 0},
    ]}
    actions = [
        action("a", "remove_card", '{"CardInstanceId":10}'),
        action("b", "remove_card", '{"CardInstanceId":11}'),
    ]
    result = audit_h21_action_aliases(state, actions, duplicate_tokens())
    assert result.equivalent_aliases == 1
    assert result.canonical_choices == 1


def test_same_card_with_different_enchantment_is_not_an_alias():
    state = {"deck": [
        {"instance_id": 10, "card_id": "strike", "state": {}},
        {"instance_id": 11, "card_id": "strike", "state": {},
         "enchantment": {"kind": "Sharp", "amount": 2}},
    ]}
    actions = [
        action("a", "remove_card", '{"CardInstanceId":10}'),
        action("b", "remove_card", '{"CardInstanceId":11}'),
    ]
    with pytest.raises(ValueError, match="Distinct public legal choices"):
        audit_h21_action_aliases(state, actions, duplicate_tokens())


def test_macro_card_subset_aliases_preserve_multisets_and_size():
    state = {"deck": [
        {"instance_id": i, "card_id": "strike", "state": {}}
        for i in range(3)
    ]}
    actions = [
        action("a", "rest_cook", '{"CardInstanceIds":[0,1]}'),
        action("b", "rest_cook", '{"CardInstanceIds":[0,2]}'),
    ]
    assert audit_h21_action_aliases(
        state, actions, duplicate_tokens()).equivalent_aliases == 1
    different = [
        action("a", "rest_cook", '{"CardInstanceIds":[0,1]}'),
        action("b", "rest_cook", '{"CardInstanceIds":[2]}'),
    ]
    with pytest.raises(ValueError, match="Distinct public legal choices"):
        audit_h21_action_aliases(state, different, duplicate_tokens())


def test_combat_compact_subsets_may_alias_only_when_public_cards_match():
    state = {"deck": [], "combat": {
        "hand": [], "discard_pile": [], "exhaust_pile": [],
        "selection_candidates": [
            {"instance_id": 10, "card_id": "strike", "state": {}},
            {"instance_id": 11, "card_id": "strike", "state": {}},
        ],
    }}
    actions = [
        action("a", "select_cards", '{"CardInstanceIds":[10]}'),
        action("b", "select_cards", '{"CardInstanceIds":[11]}'),
    ]
    result = audit_h21_action_aliases(
        state, actions, duplicate_tokens(combat=True))
    assert result.equivalent_aliases == 1
    actions = [
        action("a", "play_card", '{"CardInstanceId":10}'),
        action("b", "play_card", '{"CardInstanceId":11}'),
    ]
    with pytest.raises(ValueError, match="non-selection combat"):
        audit_h21_action_aliases(
            state, actions, duplicate_tokens(combat=True))


def test_extra_payload_dimensions_are_rejected_even_if_action_tokens_match():
    state = {"deck": [{"instance_id": 10, "card_id": "strike", "state": {}}]}
    actions = [
        action("a", "remove_card", '{"CardInstanceId":10}'),
        action("b", "remove_card", '{"CardInstanceId":10,"Price":77}'),
    ]
    with pytest.raises(ValueError, match="Unaudited extra payload"):
        audit_h21_action_aliases(state, actions, duplicate_tokens())
