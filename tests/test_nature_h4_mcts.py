"""UCT root search must discover better actions without inspecting hidden state."""
from dataclasses import dataclass

from sts2_ai.nature.h4_mcts import search_player_action


class DummySession:
    def __init__(self):
        self.released = 0

    def release_semantic_many(self, states):
        self.released += len(states)


@dataclass
class FakeState:
    session: DummySession
    outcome: str | None = None

    @property
    def terminal(self):
        return self.outcome is not None

    @property
    def boundary(self):
        return "terminal" if self.terminal else "player"

    @property
    def exit_hp(self):
        return 35 if self.outcome == "victory" else 0

    def status(self):
        return {"terminalEnemyHp": 60}

    def observe(self):
        return {"hp": 70, "max_hp": 70,
                "combat": {"enemies": [{"hp": 100}]}}

    def actions(self):
        return [{"actionId": "bad"}, {"actionId": "good"}]

    def step(self, action):
        return FakeState(
            self.session, "victory" if action == "good" else "defeat")


def test_uct_prefers_high_return_and_releases_branches():
    session = DummySession()
    root = FakeState(session)
    selected, stats = search_player_action(
        root, observation=root.observe(), legal_actions=root.actions(),
        rollout_choose=lambda *_: "bad",
        simulations=32, seed="fixed", opening_enemy_hp=100,
        player_max_hp=70, max_boundaries=10,
    )
    assert selected == "good"
    assert stats["terminal_rollouts"] == 32
    assert stats["cutoff_rollouts"] == 0
    assert stats["root_visited_actions"] == 2
    assert session.released == 32



class NonterminalFakeState(FakeState):
    def step(self, action):
        return NonterminalFakeState(self.session)


def test_uct_search_stops_at_short_horizon():
    session = DummySession()
    root = NonterminalFakeState(session)
    _, stats = search_player_action(
        root, observation=root.observe(), legal_actions=root.actions(),
        rollout_choose=lambda *_: "bad",
        simulations=8, seed="bounded", opening_enemy_hp=100,
        player_max_hp=70, max_boundaries=6,
    )
    assert stats["simulated_boundaries"] == 48
    assert stats["cutoff_rollouts"] == 8
    assert stats["terminal_rollouts"] == 0
    assert session.released == 48
