"""H4 semantic-combat UCT with Chance-Law rollouts and fair player actions.

This is an online, public-observation-keyed search baseline. It samples
unresolved draws and enemy intents from the declared reference Chance Law.
The emulator supplies the exact current continuation for simulation; hidden
states are NOT explicitly marginalized, so this is a root-conditioned
simulator MCTS baseline, not yet a certified belief-state agent.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sts2_ai.nature.draw_only_p0 import DeclaredLaw, SeededLawSampler
from sts2_ai.nature.h4_reward import boss_progress_score
from sts2_ai.nature.protocol import canonical_sha256


@dataclass
class _Edge:
    visits: int = 0
    value_sum: float = 0.0


@dataclass
class _Node:
    visits: int = 0
    edges: dict[str, _Edge] = field(default_factory=dict)


def _legal(state: Any) -> tuple[dict, list[dict]]:
    obs, actions = state.observe(), state.actions()
    if state.boundary == "player_choice":
        from sts2_ai.nature.draw_only_episode import fair_player_choice_actions
        actions = fair_player_choice_actions(state.pending, actions)
    if not actions:
        raise RuntimeError("MCTS continuation has no legal player action")
    return obs, actions


def _select(node: _Node, actions: list[dict], exploration: float) -> str:
    ids = sorted(a["actionId"] for a in actions)
    for action in ids:
        if action not in node.edges or node.edges[action].visits == 0:
            node.edges.setdefault(action, _Edge())
            return action
    return max(
        ids,
        key=lambda key: (
            node.edges[key].value_sum / node.edges[key].visits
            + exploration * math.sqrt(
                math.log(max(2, node.visits)) / node.edges[key].visits
            ),
            key,
        ),
    )


def _cutoff_value(obs: dict | None, opening_enemy_hp: int,
                  player_max_hp: int) -> float:
    """Public-state estimate at the search horizon, below any victory."""
    if not isinstance(obs, dict):
        return 0.0
    hp = obs.get("hp", 0)
    enemies = obs.get("combat", {}).get("enemies", [])
    if not isinstance(hp, int) or not isinstance(enemies, list):
        return 0.0
    remaining = sum(
        max(0, e.get("hp", 0)) for e in enemies
        if isinstance(e, dict) and isinstance(e.get("hp"), int)
    )
    progress = max(0., min(1., 1. - remaining / opening_enemy_hp))
    health = max(0., min(1., hp / player_max_hp))
    return min(0.999, 0.5 * progress + 0.5 * health)


def search_player_action(
    root: Any, *,
    observation: dict,
    legal_actions: list[dict],
    rollout_choose: Callable[[dict, list[dict]], str],
    simulations: int,
    seed: str,
    opening_enemy_hp: int,
    player_max_hp: int,
    max_boundaries: int = 24,
    exploration: float = 1.4,
) -> tuple[str, dict[str, float]]:
    """Horizon-limited UCT with chance nodes sampled from the reference law.

    A 1200-boundary rollout at *every* real card decision was prohibitively
    expensive. This search enforces a short horizon and a public-state cutoff
    estimate, with per-combat compute accounted for by its caller. Rollout
    actions use the provided fair greedy player; Nature is never consulted.
    Every forked handle is released before the next simulation.
    """
    if (simulations < 1 or max_boundaries < 1 or opening_enemy_hp < 1
            or player_max_hp < 1 or not legal_actions
            or root.boundary not in ("player", "player_choice")):
        raise ValueError("Invalid H4 MCTS root/search budget")
    root_ids = {a["actionId"] for a in legal_actions}
    if len(root_ids) != len(legal_actions):
        raise ValueError("Duplicate root action")
    tree: dict[str, _Node] = {}
    root_key = canonical_sha256([observation, sorted(root_ids)])
    root_node = tree.setdefault(root_key, _Node())
    terminals = cutoffs = total_boundaries = 0
    for sim in range(simulations):
        digest = hashlib.sha256(f"{seed}:{sim}".encode()).digest()
        sampler = SeededLawSampler(int.from_bytes(digest[:8], "big"))
        state = root
        created: list[Any] = []
        path: list[tuple[_Node, str]] = []
        expanded = False
        last_obs = observation
        try:
            for boundary in range(max_boundaries):
                if state.terminal:
                    if state.outcome not in ("victory", "defeat"):
                        raise RuntimeError("MCTS reached invalid terminal outcome")
                    terminal_hp = (state.status().get("terminalEnemyHp")
                                   if state.outcome == "defeat" else None)
                    value = boss_progress_score(
                        state.outcome,
                        player_exit_hp=state.exit_hp,
                        player_max_hp=player_max_hp,
                        opening_enemy_hp=opening_enemy_hp,
                        terminal_enemy_hp=terminal_hp,
                    )
                    terminals += 1
                    total_boundaries += boundary
                    break
                if state.boundary == "nature":
                    law = DeclaredLaw.from_pending(state.pending)
                    index = sampler.sample(law)
                    if law.operation == "enemy_intent":
                        child = state.resolve_intent(law.outcome_ids[index])
                    elif law.operation == "draw_card":
                        card = int(law.outcome_ids[index].removeprefix("card_instance:"))
                        child = state.resolve_draw_card(card)
                    else:
                        raise RuntimeError("Unsupported search Chance Law")
                elif state.boundary in ("player", "player_choice"):
                    if state is root:
                        obs, actions = observation, legal_actions
                    else:
                        obs, actions = _legal(state)
                    last_obs = obs
                    key = canonical_sha256([
                        obs, sorted(a["actionId"] for a in actions),
                    ])
                    if key != root_key and key not in tree:
                        tree[key] = _Node()
                        expanded = True
                    node = tree[key]
                    if expanded:
                        action = rollout_choose(obs, actions)
                    else:
                        action = _select(node, actions, exploration)
                        path.append((node, action))
                    if action not in {a["actionId"] for a in actions}:
                        raise RuntimeError("MCTS rollout policy chose illegal action")
                    child = state.step(action)
                else:
                    raise RuntimeError("Unexpected semantic boundary during MCTS")
                created.append(child)
                state = child
            else:
                cutoffs += 1
                total_boundaries += max_boundaries
                value = _cutoff_value(
                    last_obs, opening_enemy_hp, player_max_hp,
                )
            for node, action in path:
                node.visits += 1
                edge = node.edges[action]
                edge.visits += 1
                edge.value_sum += value
        finally:
            if created:
                root.session.release_semantic_many(created)
    visited = [
        (key, root_node.edges[key]) for key in root_ids
        if key in root_node.edges and root_node.edges[key].visits > 0
    ]
    if not visited:
        raise RuntimeError("MCTS failed to visit any root action")
    best = max(
        visited, key=lambda item: (
            item[1].visits, item[1].value_sum / item[1].visits, item[0]
        ),
    )
    return best[0], {
        "simulations": float(simulations),
        "tree_nodes": float(len(tree)),
        "terminal_rollouts": float(terminals),
        "cutoff_rollouts": float(cutoffs),
        "simulated_boundaries": float(total_boundaries),
        "root_visited_actions": float(len(visited)),
        "root_unvisited_actions": float(len(root_ids) - len(visited)),
    }
