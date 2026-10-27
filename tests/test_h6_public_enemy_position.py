"""H6 native combat supports public ordered lineups without private positions."""
from copy import deepcopy

import pytest
from test_enemy_instances_v8 import DIM, combat_frame

from sts2_ai.models.enemy_instances import enemy_instance_vectors


def test_explicit_and_player_visible_lineup_position_agree():
    explicit = combat_frame()
    ordered = deepcopy(explicit)
    for enemy in ordered["combat"]["enemies"]:
        del enemy["formation_position"]
    assert enemy_instance_vectors(ordered, DIM) == enemy_instance_vectors(
        explicit, DIM,
    )


def test_lineup_fallback_does_not_infer_enemy_identity_or_species():
    ordered = combat_frame()
    for enemy in ordered["combat"]["enemies"]:
        del enemy["formation_position"]
    del ordered["combat"]["enemies"][1]["enemy_id"]
    with pytest.raises(ValueError, match="Enemy lacks public instance"):
        enemy_instance_vectors(ordered, DIM)


def test_explicit_position_still_wins_over_lineup_index():
    frame = combat_frame()
    original = enemy_instance_vectors(frame, DIM)
    frame["combat"]["enemies"].reverse()
    assert enemy_instance_vectors(frame, DIM) == original
