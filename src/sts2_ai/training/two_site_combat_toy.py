"""Finite *synthetic* Silent-like combat testbed for two-site Nature learning.

The mechanics, reward and probabilities here are explicitly toy definitions,
never a replacement for the Slay the Spire 2 emulator. The same controller
can later attach to emulator-owned semantic events and real player returns.
This miniature game tests whether alternate Nature/player learning behaves
sensibly and whether evaluation stays outside the Nature proposal.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

from sts2_ai.nature.semantic import SemanticEvent
from sts2_ai.nature.two_site_opponent import PilotChoice, PilotEpisode, TwoSiteNature

TOY_SCHEMA = "sts2-ai-two-site-synthetic-combat-v1"
CARDS = ("Strike#1", "Defend#2", "Dagger#3", "Strike#4")
INTENTS = ("light", "heavy")
INTENT_P = (0.7, 0.3)


def seeded_rng(*parts: object) -> random.Random:
    data = json.dumps(parts, sort_keys=True).encode()
    return random.Random(int.from_bytes(hashlib.sha256(data).digest()[:16], "big"))


def semantic_toy_event(
    operation: str, candidates: tuple[str, ...], *, episode: int, turn: int,
    intent_weights: tuple[float, ...] | None = None,
) -> SemanticEvent:
    if operation not in ("enemy_intent", "draw_card"):
        raise ValueError("Synthetic two-site combat supports only intents/draws")
    unique = f"{episode}:{turn}:{operation}"
    return SemanticEvent(
        event_id=f"toy:{unique}",
        boundary_fingerprint=f"toy:state:{unique}",
        site_id=f"toy.combat.{operation}", owner_id="toy.combat",
        ordinal=turn, operation=operation, candidate_ids=candidates,
        support_fingerprint="toy.support:" + hashlib.sha256(
            repr(candidates).encode()
        ).hexdigest(), randomization_id="toy:deck-epoch-1",
        constraint_fingerprint=f"toy:known:{unique}",
        raw_weights=intent_weights,
        weight_total=sum(intent_weights) if intent_weights is not None else None,
    )


@dataclass(frozen=True, slots=True)
class PlayerChoice:
    state_key: str
    selected_action: int
    probability_use: float
    revision: int


@dataclass(frozen=True, slots=True)
class ToyEpisode:
    reward: float
    nature: PilotEpisode
    player_choices: tuple[PlayerChoice, ...]
    won: bool
    player_hp: int
    enemy_hp: int


@dataclass(slots=True)
class ToyPlayer:
    seed: int
    learning_rate: float = 0.075
    exploration: float = 0.12
    logits: dict[str, float] = field(default_factory=dict)
    revision: int = 0

    def probability_use(self, key: str) -> float:
        value = self.logits.get(key, 0)
        sigmoid = 1 / (1 + math.exp(-max(-40.0, min(40.0, value))))
        return (1 - self.exploration) * sigmoid + self.exploration * 0.5

    def select(self, key: str, *, episode: int, turn: int) -> PlayerChoice:
        p = self.probability_use(key)
        use = int(seeded_rng(self.seed, self.revision, episode, turn, key).random() < p)
        return PlayerChoice(key, use, p, self.revision)

    def update(self, episodes: tuple[ToyEpisode, ...]) -> None:
        if len(episodes) < 2:
            raise ValueError("Player update requires multiple terminal episodes")
        if any(ch.revision != self.revision
               for episode in episodes for ch in episode.player_choices):
            raise ValueError("Stale player behavior policy")
        total = math.fsum(ep.reward for ep in episodes)
        gradient: dict[str, float] = {}
        for episode in episodes:
            advantage = episode.reward - (
                total - episode.reward
            ) / (len(episodes) - 1)
            for choice in episode.player_choices:
                value = self.logits.get(choice.state_key, 0.0)
                base = 1 / (1 + math.exp(-max(-40.0, min(40.0, value))))
                p = choice.probability_use
                dp = (1 - self.exploration) * base * (1 - base)
                factor = dp * ((choice.selected_action / p) -
                               ((1 - choice.selected_action) / (1 - p)))
                gradient[choice.state_key] = (
                    gradient.get(choice.state_key, 0.0) + advantage * factor
                )
        for key, value in gradient.items():
            self.logits[key] = self.logits.get(key, 0.0) + (
                self.learning_rate * value / len(episodes)
            )
        self.revision += 1


def _native_choose(
    event: SemanticEvent, rng: random.Random,
) -> int:
    if event.operation == "enemy_intent":
        weights = event.normalized_intent_weights
    else:
        weights = (1 / event.support_count,) * event.support_count
    r = rng.random()
    cumulative = 0.0
    for index, p in enumerate(weights):
        cumulative += p
        if r < cumulative:
            return index
    return len(weights) - 1


def run_toy_combat(
    player: ToyPlayer, *, episode: int,
    nature: TwoSiteNature | None = None,
    native_seed: int = 0,
) -> ToyEpisode:
    rng = seeded_rng("synthetic-reference", native_seed, episode)
    hp, enemy_hp = 14, 13
    remaining = list(CARDS)
    player_choices: list[PlayerChoice] = []
    chance_choices: list[PilotChoice] = []
    for turn in range(3):
        intent_event = semantic_toy_event(
            "enemy_intent", INTENTS, episode=episode, turn=turn,
            intent_weights=INTENT_P,
        )
        draw_event = semantic_toy_event(
            "draw_card", tuple(remaining), episode=episode, turn=turn,
        )
        if nature is None:
            intent_index = _native_choose(intent_event, rng)
            draw_index = _native_choose(draw_event, rng)
        else:
            intent_proposal, i_choice = nature.propose(
                intent_event, context_key=f"turn:{turn}",
                candidate_features=INTENTS, episode_key=f"{episode}:intent",
            )
            draw_proposal, d_choice = nature.propose(
                draw_event, context_key=f"turn:{turn}",
                candidate_features=tuple(card.split("#")[0] for card in remaining),
                episode_key=f"{episode}:draw",
            )
            intent_index = intent_proposal.selected_index
            draw_index = draw_proposal.selected_index
            chance_choices.extend((i_choice, d_choice))
        intent = INTENTS[intent_index]
        card = remaining.pop(draw_index)
        # Player observes current selected card and committed enemy intent,
        # not the hidden remaining deck order, opponent scores or RNG state.
        player_key = f"turn:{turn}|hp:{hp // 5}|intent:{intent}|card:{card.split('#')[0]}"
        player_action = player.select(player_key, episode=episode, turn=turn)
        player_choices.append(player_action)
        # Action 1: use card; action 0: basic guard (+3 block,+2 attack).
        if player_action.selected_action == 1:
            attack, block = (
                (6, 0) if card.startswith("Strike") else
                (0, 6) if card.startswith("Defend") else (4, 2)
            )
        else:
            attack, block = 2, 3
        enemy_hp -= attack
        if enemy_hp <= 0:
            break
        hp -= max(0, (3 if intent == "light" else 7) - block)
        if hp <= 0:
            break
    won = enemy_hp <= 0 and hp > 0
    # Smooth training reward: terminal survival + damage progress + HP,
    # still clearly separate from genuine StS2 combat rewards.
    reward = (
        (2.0 if won else 0.0) +
        (max(0, hp) / 14.0) -
        (max(0, enemy_hp) / 13.0) -
        (1.0 if hp <= 0 else 0.0)
    )
    nature_episode = PilotEpisode(reward, tuple(chance_choices))
    return ToyEpisode(
        reward, nature_episode, tuple(player_choices),
        won, hp, enemy_hp,
    )


def _evaluate(player: ToyPlayer, *, seeds: tuple[int, ...]) -> dict[str, float]:
    # Fold training revision into the held-out player sampling only;
    # all arms get the same environment seeds and reference law.
    rows = tuple(
        run_toy_combat(player, episode=10_000_000 + i, native_seed=s)
        for i, s in enumerate(seeds)
    )
    return {
        "return": math.fsum(e.reward for e in rows) / len(rows),
        "win_rate": sum(e.won for e in rows) / len(rows),
        "survival": sum(e.player_hp > 0 for e in rows) / len(rows),
        "mean_exit_hp": math.fsum(max(0, e.player_hp) for e in rows) / len(rows),
    }


def run_paired_toy_pilot(
    *, rounds: int, episodes_per_batch: int, seed: int,
    evaluation_episodes: int, nature_mix: float = 0.75,
) -> dict[str, object]:
    """Equal simulator-episode budget: native 2 player batches vs adversarial 1+1.

    The Nature arm runs 1 frozen-player Nature-optimization batch and then
    1 player-update batch per round. The native arm gets 2 player-update
    batches. This prevents Nature's extra sampling cost from vanishing
    from the nominal budget.
    """
    if rounds < 1 or episodes_per_batch < 2 or evaluation_episodes < 2:
        raise ValueError("Need rounds, 2+ batch episodes and 2+ held-out episodes")
    control = ToyPlayer(seed=seed + 100)
    robust = ToyPlayer(seed=seed + 100)
    frozen_probe = ToyPlayer(seed=seed + 100)
    opponent = TwoSiteNature(seed=seed + 200, mix=nature_mix)
    untrained_opponent = TwoSiteNature(seed=seed + 200, mix=nature_mix)
    holdouts = tuple(seed + 1_000_000 + i for i in range(evaluation_episodes))
    progress = []
    for r in range(rounds):
        base = r * 2 * episodes_per_batch
        # Native has exactly the same total rollout simulation budget.
        for batch in range(2):
            native_rows = tuple(
                run_toy_combat(
                    control, episode=base + batch * episodes_per_batch + i,
                    native_seed=seed,
                )
                for i in range(episodes_per_batch)
            )
            control.update(native_rows)
        # Freeze the player while updating Nature.
        nature_rows = tuple(
            run_toy_combat(
                robust, nature=opponent, episode=base + i, native_seed=seed,
            ).nature
            for i in range(episodes_per_batch)
        )
        nature_stats = opponent.update(nature_rows)
        player_rows = tuple(
            run_toy_combat(
                robust, nature=opponent,
                episode=base + episodes_per_batch + i, native_seed=seed,
            )
            for i in range(episodes_per_batch)
        )
        robust.update(player_rows)
        if r == 0 or (r + 1) % 5 == 0 or r == rounds - 1:
            left, right = _evaluate(control, seeds=holdouts), _evaluate(
                robust, seeds=holdouts,
            )
            frozen_native = _evaluate(frozen_probe, seeds=holdouts)
            frozen_nature_rows = tuple(
                run_toy_combat(
                    frozen_probe, episode=10_000_000 + i,
                    nature=opponent, native_seed=sv,
                )
                for i, sv in enumerate(holdouts)
            )
            frozen_nature_return = math.fsum(
                row.reward for row in frozen_nature_rows
            ) / len(frozen_nature_rows)
            untrained_rows = tuple(
                run_toy_combat(
                    frozen_probe, episode=10_000_000 + i,
                    nature=untrained_opponent, native_seed=sv,
                )
                for i, sv in enumerate(holdouts)
            )
            frozen_untrained_return = math.fsum(
                row.reward for row in untrained_rows
            ) / len(untrained_rows)
            progress.append({
                "round": r + 1,
                "native": left, "nature_train": right,
                "frozen_player_difficulty": {
                    "native_return": frozen_native["return"],
                    "nature_return": frozen_nature_return,
                    "untrained_nature_return": frozen_untrained_return,
                    "nature_minus_native": (
                        frozen_nature_return - frozen_native["return"]
                    ),
                    "learned_minus_untrained_nature": (
                        frozen_nature_return - frozen_untrained_return
                    ),
                },
                "delta_native_eval_return": right["return"] - left["return"],
                "nature_objective": nature_stats,
                "simulated_combats_per_arm": (r + 1) * 2 * episodes_per_batch,
            })
    return {
        "schema": TOY_SCHEMA, "synthetic_not_sts2": True,
        "source_law": "toy:conditional-intents-0.7-0.3-and-deferred-uniform-deck-v1",
        "experiment_type": "equal_simulator_budget_native_vs_unweighted_stress_nature",
        "seed": seed, "rounds": rounds, "episodes_per_batch": episodes_per_batch,
        "evaluation_episodes": evaluation_episodes,
        "policy_player_observation": "current card/visible intent/turn/HP bucket",
        "nature_eligibility": sorted(("enemy_intent", "draw_card")),
        "progress": progress,
        "final_nature_checkpoint": opponent.dump(),
        "native_reference_eval_only": True,
    }
