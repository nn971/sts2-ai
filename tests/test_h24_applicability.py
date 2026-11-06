"""H24 predicate structure, fail-closed verdicts and source-revision proof gates."""
from copy import deepcopy
from dataclasses import replace

import pytest

from sts2_ai.training.h24_applicability import (
    ResourceAuthority,
    ResourceConstraintSelection,
    REQUIRED,
    audit_public_pair,
)

BASE = {
    "hp": 30, "max_hp": 70, "gold": 90, "floor": 5, "act": 1,
    "deck": [{"instance_id": 1, "card_id": "silent.strike", "state": {}}],
    "potions": [], "potion_slot_capacity": 3,
    "potion_slots_empty": 3, "potion_slots_occupied": 0,
    "relics": [], "combat": None,
}


def make_auth(resource, item=None):
    ok = frozenset(REQUIRED[resource])
    fields = dict(emulator_revision="pinned-r", user_approved=ok,
                  mechanics_verified=ok)
    if resource == "potion" and item:
        fields.update(user_potions=frozenset({item}),
                      verified_potions=frozenset({item}))
    if resource == "relic" and item:
        fields.update(user_relics=frozenset({item}),
                      verified_relics=frozenset({item}))
    return ResourceAuthority(**fields)


@pytest.mark.parametrize("resource", ["hp", "gold", "potion", "relic"])
def test_structure_and_external_proof_gates(resource):
    a, b = deepcopy(BASE), deepcopy(BASE)
    item = None
    if resource == "hp":
        b["hp"] = 45
    elif resource == "gold":
        b["gold"] = 99
    elif resource == "potion":
        item = "proto.potion.block"
        b["potions"] = [{"slot": 1, "potion_id": item, "state": {}}]
        b["potion_slots_occupied"] = 1
        b["potion_slots_empty"] = 2
    else:
        item = "proto.relic.paper_krane"
        b["relics"] = [{"relic_id": item, "state": {}}]
    selected = ResourceConstraintSelection(relic=True)
    blocked = audit_public_pair(a, b, resource=resource,
                                emulator_revision="pinned-r", selection=selected)
    assert not blocked.eligible
    assert blocked.structural_failures
    approved = audit_public_pair(a, b, resource=resource,
                                 authority=make_auth(resource, item),
                                 emulator_revision="pinned-r", selection=selected)
    assert approved.eligible, approved
    b["floor"] = 7
    assert not audit_public_pair(a, b, resource=resource,
                authority=make_auth(resource, item),
                emulator_revision="pinned-r").eligible


def test_invalid_potion_capacity_fails_closed():
    b = deepcopy(BASE)
    b["potions"] = [{"slot": 4, "potion_id": "test", "state": {}}]
    b["potion_slots_occupied"] = 1
    b["potion_slots_empty"] = 2
    result = audit_public_pair(
        BASE, b, resource="potion",
        authority=make_auth("potion", "test"), emulator_revision="pinned-r",
    )
    assert not result.eligible
    assert any("P1" in s for s in result.structural_failures)


def test_review_and_verdict_are_independent():
    a, b = deepcopy(BASE), deepcopy(BASE)
    b["hp"] = 40
    auth = make_auth("hp")
    for modified in (replace(auth, mechanics_verified=frozenset()),
                     replace(auth, user_approved=frozenset())):
        result = audit_public_pair(a, b, resource="hp",
                                   authority=modified, emulator_revision="pinned-r")
        assert not result.eligible
        assert result.missing_mechanics_review or result.missing_user_verdict


def test_nonpositive_hp_cannot_cross_death_boundary():
    a, b = deepcopy(BASE), deepcopy(BASE)
    a["hp"], b["hp"] = 0, 2
    result = audit_public_pair(a, b, resource="hp",
        authority=make_auth("hp"), emulator_revision="pinned-r")
    assert not result.eligible
    assert any("H1" in s for s in result.structural_failures)


def test_revision_mismatch_and_invalid_relic_state_fail():
    a, b = deepcopy(BASE), deepcopy(BASE)
    item = "proto.relic.paper_krane"
    b["relics"] = [{"relic_id": item, "state": {"counter": 1}}]
    result = audit_public_pair(a, b, resource="relic",
        authority=make_auth("relic", item), emulator_revision="different")
    assert not result.eligible
    assert any("R2" in s for s in result.structural_failures)
    assert any("G6" in s for s in result.structural_failures)


def test_no_mutation():
    a, b = deepcopy(BASE), deepcopy(BASE)
    b["gold"] = 91
    original = deepcopy(a), deepcopy(b)
    audit_public_pair(a, b, resource="gold", emulator_revision="pinned-r")
    assert (a, b) == original

def test_h24_relic_default_off_despite_full_predicate_authority():
    lower, upper = deepcopy(BASE), deepcopy(BASE)
    identity = "proto.relic.paper_krane"
    upper["relics"] = [{"relic_id": identity, "state": {}}]
    authority = make_auth("relic", identity)
    default = ResourceConstraintSelection()
    assert default.hp and default.gold and default.potion
    assert not default.relic
    result = audit_public_pair(lower, upper, resource="relic",
        emulator_revision="pinned-r", authority=authority)
    assert result.disabled_by_config
    assert not result.eligible
    explicitly_enabled = audit_public_pair(lower, upper, resource="relic",
        emulator_revision="pinned-r", authority=authority,
        selection=ResourceConstraintSelection(relic=True))
    assert explicitly_enabled.eligible
    assert not explicitly_enabled.disabled_by_config


def test_h24_relic_opt_in_still_needs_mechanics_and_item_verdict():
    lower, upper = deepcopy(BASE), deepcopy(BASE)
    upper["relics"] = [{"relic_id": "proto.relic.centennial_puzzle", "state": {}}]
    result = audit_public_pair(lower, upper, resource="relic",
        emulator_revision="pinned-r", selection=ResourceConstraintSelection(relic=True))
    assert not result.eligible
    assert any("R2:item:" in value for value in result.missing_mechanics_review)
    assert any("R2:item:" in value for value in result.missing_user_verdict)


def test_h24_rejects_non_boolean_selection_flag():
    with pytest.raises(ValueError, match="booleans"):
        ResourceConstraintSelection(relic=1)
