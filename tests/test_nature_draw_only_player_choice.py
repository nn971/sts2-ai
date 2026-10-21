"""Complete replay of semantic player_choice continuations through the fair menu."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from sts2_ai.nature.draw_only_episode import (
    fair_player_choice_actions,
    play_draw_only_episode,
)
from sts2_ai.nature.draw_only_p0 import DrawOnlyContractError
from sts2_ai.nature.two_site_opponent import TwoSiteNature


def _action(ident, *, kind="select_card"):
    return {"actionId": ident, "kind": kind, "payloadJson": "{}"}


def _pending(ident="choose:a"):
    return {
        "kind": "player_card_selection",
        "eventId": "source-choice-event",
        "supportFingerprint": "source-card-selection-support",
        "actions": [{"actionId": ident, "cardInstanceIds": [17]}],
    }


def test_choice_support_is_intersected_with_actual_public_menu():
    actions = [_action("play:other", kind="play_card"), _action("choose:a")]
    raw = _pending()
    assert fair_player_choice_actions(raw, actions) == [actions[1]]
    assert len(actions) == 2


@pytest.mark.parametrize("invalid", [
    None,
    {},
    {"kind": "made_up", "actions": [{"actionId": "choose:a"}]},
    {"kind": "player_card_selection", "actions": []},
    {"kind": "player_card_selection", "actions": [{"actionId": "missing"}]},
    {"kind": "player_card_selection", "actions": [{"actionId": "choose:a"}] * 2},
])
def test_bad_choice_support_fails_closed(invalid):
    with pytest.raises(DrawOnlyContractError):
        fair_player_choice_actions(invalid, [_action("choose:a")])


def test_source_draw_hook_actions_follow_only_owned_candidate_menu():
    event = {
        "kind": "draw_hook_action_choice",
        "candidates": [
            {"actionId": "hook:b", "kind": "play_card", "payloadFingerprint": "sha"},
        ],
    }
    assert fair_player_choice_actions(
        event, [_action("end", kind="end_turn"), _action("hook:b", kind="play_card")]
    ) == [_action("hook:b", kind="play_card")]


@dataclass
class FakeState:
    boundary: str
    fingerprint: str
    pending: dict | None
    terminal: bool
    outcome: str | None = None
    exit_hp: int | None = None

    def status(self):
        return {"sourceOpeningFingerprint": "frozen-combat-opening"}

    def observe(self):
        return {
            "hp": 58,
            "max_hp": 66,
            "combat": {"hand": [], "enemies": [], "draw_pile": 0},
        }

    def actions(self):
        if self.boundary == "player_choice":
            # Source's generic action menu may include unrelated actions.
            return [_action("end", kind="end_turn"), _action("choose:a")]
        return [_action("play:survivor", kind="play_card")]

    def step(self, selected):
        if self.boundary == "player":
            assert selected == "play:survivor"
            return FakeState("player_choice", "after-survivor", _pending(), False)
        assert self.boundary == "player_choice"
        assert selected == "choose:a"
        return FakeState("terminal", "after-choice", None, True, "victory", 55)

    def save_snapshot(self):
        return "terminal:after-choice"


class FakeSession:
    capabilities = {
        "semanticChanceSchemaId": "sts2-v111-semantic-playable-p0-v1",
        "semanticPlayable": True,
        "semanticIntentSupportSchemaId": "sts2-v111-semantic-weighted-intent-v1",
        "semanticControlledSites": ["enemy_intent", "draw_card"],
        "referenceNatureSessionSchemaId": "sts2-v111-reference-nature-session-v1",
        "semanticSeededRouteEntryProviderId":
            "engine-owned-v111-native-weak-route-entry-unverified-v1",
        "stochasticCapabilities": {"chanceLawSchemaId": "sts2-v111-chance-law-v1"},
    }

    def begin_seeded_semantic_combat(self, seed, **kwargs):
        return FakeState("player", "opening", None, False)

    def load_semantic_snapshot(self, snap):
        assert snap == "terminal:after-choice"
        return FakeState("terminal", "after-choice", None, True, "victory", 55)

    def release_semantic_many(self, states):
        return len(states)


def test_player_choice_transcript_exact_replay_and_nonleaking_policy_menu():
    seen = []

    def choose(_obs, actions):
        seen.append(tuple(x["actionId"] for x in actions))
        return actions[0]["actionId"]

    kw = dict(
        source_seed="choice-fixture", reference_seed=19,
        nature=TwoSiteNature(seed=13), emulator_revision="pinned",
        entry_mode="seeded-fixed-shrinker", draw_mode="reference",
    )
    a = play_draw_only_episode(FakeSession(), player_choose=choose, **kw)
    assert a.outcome == "victory"
    assert a.terminal_replay_verified
    assert [s.phase for s in a.decisions] == ["player", "player_choice"]
    assert seen == [("play:survivor",), ("choose:a",)]
    b = play_draw_only_episode(
        FakeSession(), replay=a.decisions,
        player_choose=lambda *_: (_ for _ in ()).throw(AssertionError("replayed")),
        **kw,
    )
    assert a.digest == b.digest


def test_player_choice_replay_rejects_wrong_source_support():
    kw = dict(
        source_seed="choice-fixture", reference_seed=19,
        nature=TwoSiteNature(seed=13), emulator_revision="pinned",
        entry_mode="seeded-fixed-shrinker", draw_mode="reference",
    )
    a = play_draw_only_episode(
        FakeSession(), player_choose=lambda _, a: a[0]["actionId"], **kw
    )
    altered = list(a.decisions)
    from dataclasses import replace

    altered[1] = replace(altered[1], support_id="different-source-support")
    with pytest.raises(DrawOnlyContractError, match="replay transition mismatch"):
        play_draw_only_episode(FakeSession(), replay=tuple(altered), **kw)
