"""Contract and learning tests for public-state conditioned draw Nature."""
from __future__ import annotations

from copy import deepcopy

import pytest

from sts2_ai.emulator.semantic_p0_combat import _pending_event
from sts2_ai.nature.contextual_draw import (
    CONTEXTUAL_SCHEMA,
    ContextualDrawNature,
    encode_public_observation,
    restore_nature,
)
from sts2_ai.nature.draw_only_p0 import DrawOnlyRouter, SeededLawSampler
from sts2_ai.nature.two_site_opponent import PilotEpisode, TwoSiteNature
from test_nature_draw_only_law import pending


def visible():
    return {
        "character_id": "proto.silent",
        "hp": 42, "max_hp": 70, "gold": 60, "act": 1,
        "deck": [{"card_id": "proto.silent.strike", "instance_id": 2}],
        "relics": [{"relic_id": "proto.silent.starter_relic"}],
        "potions": [],
        "combat": {
            "turn": 2, "energy": 2, "player_block": 1,
            "hand": [{"card_id": "proto.silent.strike", "instance_id": 21}],
            "discard_pile": [{"card_id": "proto.silent.defend", "instance_id": 11}],
            "exhaust_pile": [], "play_pile": [], "draw_pile_count": 8,
            "enemies": [
                {"enemy_id": "proto.enemy.mawler", "hp": 40, "block": 0,
                 "move_id": "attack", "intent_damage": 15, "statuses": {"weak": 1},
                 "powers": []},
                {"enemy_id": "proto.enemy.summon", "hp": 11, "block": 3,
                 "move_id": "buff", "intent_damage": 0, "statuses": {},
                 "powers": []},
            ],
            "player_powers": [{"power_id": "strength", "stacks": 2}],
        },
    }


def event():
    return _pending_event(pending(), "parent")[0]


def test_full_observation_encoder_responds_to_combat_and_ignores_entity_order():
    original = visible()
    encoded = encode_public_observation(original)
    reversed_entities = deepcopy(original)
    reversed_entities["combat"]["enemies"].reverse()
    assert encode_public_observation(reversed_entities) == encoded

    renumbered = deepcopy(original)
    renumbered["combat"]["hand"][0]["instance_id"] = 999
    assert encode_public_observation(renumbered) == encoded

    for part, name, value in [
        ("root", "hp", 12),
        ("combat", "energy", 0),
        ("combat", "turn", 9),
        ("combat", "draw_pile_count", 3),
    ]:
        altered = deepcopy(original)
        target = altered if part == "root" else altered["combat"]
        target[name] = value
        assert encode_public_observation(altered) != encoded

    for field in ("hand", "discard_pile", "exhaust_pile", "enemies", "player_powers"):
        altered = deepcopy(original)
        altered["combat"][field].append({"card_id": "a", "power_id": "b"})
        assert encode_public_observation(altered) != encoded


def test_encoder_fails_closed_on_private_pending_or_order():
    for field in ("pending", "chanceLaw", "draw_pile_order", "semanticDeck"):
        altered = visible()
        altered[field] = {"secret": 1}
        with pytest.raises(ValueError, match="Privileged"):
            encode_public_observation(altered)


def test_policy_varies_with_public_state_and_keeps_chance_law_support():
    nature = ContextualDrawNature(seed=101)
    ev = event()
    obs = visible()
    choices = ("proto.silent.strike", "proto.silent.defend")
    a, pilot = nature.propose(
        ev, context_key="draw", candidate_features=choices,
        episode_key="same-world", public_observation=obs,
    )
    obs_changed = deepcopy(obs)
    obs_changed["combat"]["enemies"][0]["move_id"] = "buff"
    obs_changed["combat"]["enemies"][0]["intent_damage"] = 0
    b, different = nature.propose(
        ev, context_key="draw", candidate_features=choices,
        episode_key="same-world", public_observation=obs_changed,
    )
    assert pilot.state_features != different.state_features
    assert pilot.probabilities != different.probabilities
    assert sum(pilot.probabilities) == pytest.approx(1)
    assert min(pilot.probabilities) >= 0.125
    assert a.selected_id in ev.candidate_ids
    assert b.selected_id in ev.candidate_ids
    assert nature.propose(
        ev, context_key="draw", candidate_features=choices,
        episode_key="same-world", public_observation=obs,
    )[1] == pilot


def test_router_requires_public_state_for_contextual_draw_but_not_intent():
    nature = ContextualDrawNature(seed=101)
    router = DrawOnlyRouter(nature, SeededLawSampler(9))
    law = pending()
    with pytest.raises(ValueError, match="fair state"):
        router.choose(law, fingerprint="parent", episode_key="world", ordinal=0)
    chosen = router.choose(
        law, fingerprint="parent", episode_key="world", ordinal=0,
        public_observation=visible(),
    )
    assert chosen.controller == "learned-draw"
    assert chosen.nature_choice is not None
    assert chosen.nature_choice.state_features

    intent = router.choose(
        pending("enemy_intent"), fingerprint="parent",
        episode_key="world", ordinal=1,
    )
    assert intent.controller == "engine-law-reference"
    assert intent.nature_choice is None


def test_contextual_policy_gradient_and_checkpoint_resume():
    nature = ContextualDrawNature(seed=11, learning_rate=0.1)
    ev = event()
    options = dict(event=ev, context_key="draw", episode_key="world",
                   candidate_features=("strike", "defend"),
                   public_observation=visible())
    _, low = nature.propose(**options)
    different = deepcopy(visible())
    different["hp"] = 10
    _, high = nature.propose(**{**options, "public_observation": different})
    before = nature.dump()
    result = nature.update((
        PilotEpisode(0.1, (low,)), PilotEpisode(0.9, (high,)),
    ))
    assert result["nature_revision"] == 1
    assert result["nature_gradient_norm"] > 0
    assert nature.dump()["state_weight"] != before["state_weight"]
    restored = restore_nature(nature.dump())
    assert isinstance(restored, ContextualDrawNature)
    assert restored.dump() == nature.dump()
    assert restored.propose(**options) == nature.propose(**options)


def test_tabular_nature_retains_original_behavior():
    nature = TwoSiteNature(seed=5)
    assert restore_nature(nature.dump()).dump() == nature.dump()
    assert CONTEXTUAL_SCHEMA != nature.policy_id
