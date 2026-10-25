"""Native A10 count-only public combat must be accepted by frozen H4 v8."""
import json
from copy import deepcopy

import pytest
from test_enemy_instances_v8 import combat_frame, targeted
from test_nature_h3_joint_player import specimen

from sts2_ai.emulator import Observation
from sts2_ai.emulator.semantic_p0_public_projection import project_public_p0_observation
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.nature.protocol import canonical_sha256


def test_native_count_only_combat_reuses_public_projection_without_private_cards():
    model = specimen()
    visible = combat_frame()
    native = deepcopy(visible)
    combat = native["combat"]
    # Actual PrototypeAiCombat emits drawPileCount, not drawPile identities,
    # and no relicCounters field. Preserve the public count only.
    source_draw = combat.pop("draw_pile", [])
    combat["draw_pile_count"] = len(source_draw)
    combat.pop("relic_counters", None)
    native.pop("p0_public_projection", None)
    original = deepcopy(native)
    policy = "prototype-fair-v0"
    observation = Observation(policy, json.dumps(native), canonical_sha256(native))
    legal = (targeted(1), targeted(2))
    got = model.evaluate(observation, legal)
    expected_public = project_public_p0_observation(native)
    expected = model.combat.evaluate(
        Observation(
            policy, json.dumps(expected_public),
            canonical_sha256(expected_public),
        ),
        legal,
    )
    assert got == expected
    assert native == original
    assert expected_public["combat"]["draw_pile"] == []
    assert expected_public["combat"]["draw_pile_count"] == len(source_draw)
    assert expected_public["combat"]["draw_pile_opaque"] is True
    assert "combat.draw_pile" in expected_public["p0_opaque_public_fields"]
    assert "combat.relic_counters" in expected_public["p0_opaque_public_fields"]


def test_complete_public_combat_remains_untouched_and_no_hidden_data_allowed():
    model = specimen()
    state = combat_frame()
    legal = (targeted(1), targeted(2))
    observation = Observation(
        "prototype-fair-v0", json.dumps(state), canonical_sha256(state)
    )
    assert model.evaluate(observation, legal) == model.combat.evaluate(
        observation, legal
    )

    source = deepcopy(state)
    source["combat"].pop("draw_pile", None)
    source["combat"]["draw_pile_count"] = 3
    source["semanticDeck"] = {"hidden": True}
    observation = Observation(
        "prototype-fair-v0", json.dumps(source), canonical_sha256(source)
    )
    with pytest.raises(ValueError, match="Privileged Nature state"):
        model.evaluate(observation, legal)
