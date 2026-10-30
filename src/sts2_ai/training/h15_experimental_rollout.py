"""Multi-turn H15 diagnostic combat rollouts under an UNVERIFIED Nature prior.

The player receives exactly the public observation and legal actions. The
frozen combat policy ranks those actions. Unsupported actions are skipped with
explicit accounting; the source emulator, not this module, owns all mechanics.

Chance outcomes are sampled from the law attached to each suspended boundary.
Those laws are NOT conditioned on the complete public history, and these runs
must never be used as fair PUCT or policy/value training labels.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator import FAIR_POLICY_ID, LegalAction, Observation
from sts2_ai.models.h15_policy_value import H15CombatEvaluator
from sts2_ai.nature.h4_reward import encounter_total_hp_score

NATURE_KINDS = frozenset({
    "enemy_action", "enemy_intent", "draw_card", "random_target",
})
H15_ROLLOUT_SCHEMA = "sts2-h15-unverified-policy-multiturn-rollout-v1"


def _policy_input(frame: Mapping[str, Any]) -> tuple[Observation, tuple[LegalAction, ...]]:
    if frame.get("boundaryKind") != "player":
        raise ValueError("H15 tactical inference requires a player boundary")
    if frame.get("observationPolicyId") != FAIR_POLICY_ID:
        raise ValueError("H15 status omitted the fair public observation policy")
    payload = frame.get("observationPayloadJson")
    digest = frame.get("observationHash")
    menu = frame.get("legalActions")
    if not isinstance(payload, str) or not isinstance(digest, str):
        raise ValueError("H15 status omitted the public observation")
    if not isinstance(menu, list) or not menu:
        raise ValueError("H15 player boundary has no legal actions")
    observation = Observation(FAIR_POLICY_ID, payload, digest)
    actions: list[LegalAction] = []
    seen: set[str] = set()
    for raw in menu:
        if (not isinstance(raw, dict)
                or not isinstance(raw.get("actionId"), str)
                or raw["actionId"] in seen
                or not isinstance(raw.get("kind"), str)
                or not isinstance(raw.get("payloadJson"), str)):
            raise ValueError("Invalid H15 public action menu")
        seen.add(raw["actionId"])
        actions.append(LegalAction(
            raw["actionId"], raw["kind"], raw["payloadJson"],
        ))
    return observation, tuple(actions)


def _sample_law(law: Mapping[str, Any], rng: random.Random) -> str:
    if law.get("probabilityStatus") != "UNVERIFIED":
        raise ValueError("Only unverified experimental Nature is supported")
    outcomes = law.get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise ValueError("Pending Nature boundary has no declared outcomes")
    total = 0
    pairs: list[tuple[str, int]] = []
    seen: set[str] = set()
    for entry in outcomes:
        if not isinstance(entry, dict):
            raise ValueError("Invalid Nature outcome")
        outcome_id = entry.get("outcomeId")
        weight = entry.get("weight")
        if (not isinstance(outcome_id, str) or not outcome_id
                or outcome_id in seen or type(weight) is not int
                or weight <= 0):
            raise ValueError("Malformed or duplicated Nature support")
        seen.add(outcome_id)
        total += weight
        pairs.append((outcome_id, total))
    if type(law.get("totalWeight")) is not int or law["totalWeight"] != total:
        raise ValueError("Nature law weight total is inconsistent")
    ticket = rng.randrange(total)
    for outcome_id, boundary in pairs:
        if ticket < boundary:
            return outcome_id
    raise AssertionError("Nature ticket outside declared support")


@dataclass(frozen=True)
class RolloutLimits:
    max_player_actions: int = 256
    max_turns: int = 32
    max_nature_decisions: int = 1024

    def validate(self) -> None:
        if min(self.max_player_actions, self.max_turns,
               self.max_nature_decisions) < 1:
            raise ValueError("H15 rollout limits must be positive")


def run_policy_combat(
    backend: Any, source: str, *, independent_seed: str,
    teacher: H15CombatEvaluator, rng: random.Random,
    limits: RolloutLimits = RolloutLimits(),
    forced_root_action_id: str | None = None,
    expected_root_observation_hash: str | None = None,
    expected_root_action_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Own a temporary H15 session until combat exit or an explicit limit.

    When a candidate is unsupported, the same public decision is reused and
    the next-highest policy action is tried. This is a coverage diagnostic,
    not a valid on-policy RL trajectory. No private state reaches the actor.
    """
    limits.validate()
    if forced_root_action_id is not None and not forced_root_action_id:
        raise ValueError("Forced root action ID must be nonempty")
    root_selected = False
    handle: str | None = None
    chosen: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    chance_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    final: dict[str, Any] | None = None
    try:
        frame = backend.h15_experimental_begin(
            source, independent_seed=independent_seed,
        )
        handle = frame["h15Handle"]
        if expected_root_observation_hash is not None:
            if frame.get("observationHash") != expected_root_observation_hash:
                raise ValueError(
                    "Independent candidate changed its public root observation"
                )
        if expected_root_action_ids is not None:
            _, observed_actions = _policy_input(frame)
            if tuple(a.action_id for a in observed_actions) != expected_root_action_ids:
                raise ValueError(
                    "Independent candidate changed the public root legal menu"
                )
        while True:
            kind = frame["boundaryKind"]
            if kind == "combat_exit":
                # The H9 tactical value target is the source terminal
                # total-nonminion-HP reward divided by two. No invented
                # reward, survival-only proxy, or censored value is saved.
                public_exit = frame.get("combatExitH9Reward")
                if not isinstance(public_exit, dict):
                    raise ValueError(
                        "Combat exit omitted authoritative H9 reward fields"
                    )
                hp = public_exit.get("hp")
                max_hp = public_exit.get("maxHp")
                if type(hp) is not int or type(max_hp) is not int:
                    raise ValueError("Combat exit omitted public HP/max_hp")
                outcome = frame["combatOutcome"]
                if outcome not in ("victory", "defeat"):
                    raise ValueError("Combat exit has no genuine outcome")
                measured_reward = encounter_total_hp_score(
                    outcome,
                    player_exit_hp=hp, player_max_hp=max_hp,
                    total_enemy_hp=public_exit.get(
                        "defeatTotalNonMinionEnemyHp"
                    ),
                    remaining_enemy_hp=public_exit.get(
                        "defeatRemainingNonMinionEnemyHp"
                    ),
                )
                final = {
                    "status": "combat_exit",
                    "outcome": outcome,
                    "exit_hp": hp,
                    "terminal_reward": measured_reward,
                    "terminal_value": measured_reward / 2,
                }
                break
            if kind in NATURE_KINDS:
                if counts["nature"] >= limits.max_nature_decisions:
                    final = {"status": "censored", "reason": "nature_budget"}
                    break
                law = frame["chanceLaw"]
                outcome_id = _sample_law(law, rng)
                at_version = frame["version"]
                frame = backend.h15_experimental_choose(
                    handle, at_version, law, outcome_id,
                )
                history.append({
                    "type": "nature",
                    "version": at_version,
                    "boundary_kind": kind,
                    "law_id": law["lawId"],
                    "outcome_id": outcome_id,
                })
                counts["nature"] += 1
                chance_counts[kind] += 1
                continue
            if kind != "player":
                final = {
                    "status": "unsupported_boundary",
                    "reason": kind,
                }
                break
            if counts["player"] >= limits.max_player_actions:
                final = {"status": "censored", "reason": "player_action_budget"}
                break
            if counts["end_turn"] >= limits.max_turns:
                final = {"status": "censored", "reason": "turn_budget"}
                break

            observation, menu = _policy_input(frame)
            prediction = teacher.predict(observation, menu)
            if len(prediction.logits) != len(menu):
                raise ValueError("Tactical policy logits/action count mismatch")
            # Prefer highest policy logit; preserve source menu order for ties.
            ranks = sorted(
                range(len(menu)),
                key=lambda i: (-prediction.logits[i], i),
            )
            # A candidate's first move is fixed for counterfactual
            # evaluation. Unsupported roots are censored; silently
            # falling back would evaluate a DIFFERENT action instead.
            if forced_root_action_id is not None and not root_selected:
                candidates = [
                    i for i in ranks
                    if menu[i].action_id == forced_root_action_id
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Forced root action absent from public legal menu"
                    )
                ranked_indices = candidates
            else:
                ranked_indices = ranks
            accepted = False
            for index in ranked_indices:
                rank = ranks.index(index)
                action = menu[index]
                if action.kind not in ("play_card", "use_potion", "end_turn"):
                    rejection_counts["unsupported_action_kind:" + action.kind] += 1
                    if forced_root_action_id is not None and not root_selected:
                        final = {
                            "status": "unsupported_root_action",
                            "reason": "unsupported_action_kind:" + action.kind,
                        }
                        break
                    continue
                try:
                    if action.kind == "end_turn":
                        advanced = backend.h15_experimental_end_turn(
                            handle, frame["version"], action.action_id,
                        )
                    else:
                        advanced = backend.h15_experimental_player_action(
                            handle, frame["version"], action.action_id,
                        )
                except Exception as exc:
                    # Skip only an explicitly unsupported effect/site. Other
                    # bridge failures can indicate broken mechanics or stale
                    # provenance; report them rather than masking the bug by
                    # selecting a different action.
                    if "UNSUPPORTED_" not in str(exc):
                        raise
                    # The bridge promises transactional rejection. Check its
                    # public fingerprint and version before attempting an
                    # alternative in the same decision state.
                    restored = backend.h15_experimental_status(handle)
                    if (restored["version"] != frame["version"] or
                            restored.get("observationHash") !=
                            frame.get("observationHash")):
                        raise RuntimeError(
                            "A rejected H15 player action mutated its session"
                        ) from exc
                    msg = str(exc)
                    prefix = msg.split(" : ", 1)[0][:220]
                    rejection_counts[prefix] += 1
                    if forced_root_action_id is not None and not root_selected:
                        final = {
                            "status": "unsupported_root_action",
                            "reason": prefix,
                        }
                        break
                    continue
                chosen.append({
                    "kind": action.kind,
                    "action_id": action.action_id,
                    "policy_rank": rank,
                    "turn": frame.get("combatTurn"),
                    "public_observation_hash": frame["observationHash"],
                    "value_estimate": prediction.value,
                })
                history.append({
                    "type": "player",
                    "version": frame["version"],
                    "public_observation_hash": frame["observationHash"],
                    "action_id": action.action_id,
                    "action_kind": action.kind,
                    "policy_rank": rank,
                })
                counts["player"] += 1
                root_selected = True
                if action.kind == "end_turn":
                    counts["end_turn"] += 1
                else:
                    counts["card" if action.kind == "play_card"
                           else "potion"] += 1
                frame = advanced
                accepted = True
                break
            if not accepted:
                if final is None:
                    final = {
                        "status": "unsupported_player_menu",
                        "reason": "no executable action in policy-ranked menu",
                    }
                break
    except Exception as exc:
        final = {
            "status": "rejected",
            "reason": type(exc).__name__ + ": " + str(exc),
        }
    finally:
        if handle is not None:
            backend.h15_experimental_release_many((handle,))

    return {
        "schema": H15_ROLLOUT_SCHEMA,
        "fair_search_roots": 0,
        "probability_status": "UNVERIFIED",
        "candidate_model": "h15-native-structure-prototype-rng-candidate-unverified-v1",
        **(final or {"status": "rejected", "reason": "missing_termination"}),
        "player_actions": counts["player"],
        "card_plays": counts["card"],
        "potion_uses": counts["potion"],
        "turns_ended": counts["end_turn"],
        "nature_decisions": counts["nature"],
        "nature_kinds": dict(chance_counts),
        "rejections": dict(rejection_counts),
        "policy_choices": chosen,
        "decision_history": history,
        "forced_root_action_id": forced_root_action_id,
    }
