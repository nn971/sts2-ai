"""Search-guided native A10 combat data, with full public-history provenance.

The source run is replayed *only* to recover an authentic combat entry and its
public action/menu/observation transcript. Search uses a separately supplied
certified fair continuation sampler, never the source run's exact combat fork.
All search roots are aligned to the SAME realized fight, and all value labels
come from its actual terminal H9 total-nonminion-HP reward.

The sampler is an explicit dependency: H8's current native snapshot fork
alone cannot satisfy it. Missing conditioning fails before any target is saved.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.chance import (
    FairContinuationSampler,
    PublicHistoryStep,
    require_fair_sampler,
)
from sts2_ai.emulator.protocol import EmulatorBackend, LegalAction, StateHandle
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run
from sts2_ai.evaluation.combat_snapshots import (
    POLICY,
    CombatSnapshotRecipe,
    _verify,
)
from sts2_ai.models.h15_policy_value import H15CombatEvaluator
from sts2_ai.search.combat_puct import CombatFrame, CombatPuct
from sts2_ai.search.h15_public_combat_adapter import H15PublicCombatAdapter
from sts2_ai.training.h15_search_supervision import target_record, value_target_record


@dataclass(frozen=True)
class NativeSearchCollection:
    source_seed: str
    entry_hash: str
    encounter_tier: str
    outcome: str
    reward: float
    exit_hp: int
    combat_decisions: int
    searched_decisions: int
    targets: tuple[dict[str, Any], ...]
    searched_root_diagnostics: tuple[dict[str, Any], ...] = ()
    final_player_turn: dict[str, Any] | None = None




def select_late_value_indices(
    history: list[PublicHistoryStep], *,
    min_turn: int, max_player_hp: int, max_enemy_hp: int,
    max_samples: int, exclude: set[int],
) -> list[int]:
    """Select public-only endgame observations, with bounded turn coverage.

    This rule could run online: it inspects only the decision's own public
    observation/menu and earlier selected turns, never terminal outcomes.
    The eventual terminal reward is attached *after* real combat completion.
    """
    if (min_turn < 1 or max_player_hp < 0 or max_enemy_hp < 0
            or max_samples < 0):
        raise ValueError("Invalid H15 late value-sampling limits")
    chosen: list[int] = []
    sampled_turns: set[int] = set()
    sampled_lethal_turns: set[int] = set()
    for index, step in enumerate(history):
        if len(chosen) >= max_samples:
            break
        if step.chosen_action is None or not step.legal_actions:
            continue
        public = json.loads(step.observation.payload_json)
        combat = public.get("combat") if isinstance(public, dict) else None
        if not isinstance(combat, dict) or "h15_player_choice_kind" in public:
            continue
        turn, hp = combat.get("turn"), public.get("hp")
        if type(turn) is not int or turn < min_turn or type(hp) is not int:
            continue
        enemies = combat.get("enemies")
        if not isinstance(enemies, list):
            raise ValueError("Public late combat missing enemy roster")
        enemy_hps = [
            x["hp"] for x in enemies if isinstance(x, dict)
            and type(x.get("hp")) is int and x["hp"] > 0
        ]
        if not enemy_hps or (hp > max_player_hp
                and min(enemy_hps) > max_enemy_hp):
            continue
        # One decision when entering danger on each new turn, plus one
        # additional within-turn choice when a visible enemy is <= 12 HP.
        lethal = min(enemy_hps) <= 12
        new_turn = turn not in sampled_turns
        new_lethal = lethal and turn not in sampled_lethal_turns
        if not (new_turn or new_lethal):
            continue
        sampled_turns.add(turn)
        if lethal:
            sampled_lethal_turns.add(turn)
        # Do not apply the same realized return twice at a searched root.
        if index not in exclude:
            chosen.append(index)
    return chosen


def summarize_final_player_turn(
    history: list[PublicHistoryStep], teacher: H15CombatEvaluator,
) -> dict[str, Any]:
    """Explain the last *actually played* public turn using public-only logits.

    The turn is selected retrospectively, after the terminal combat result
    exists. This is an observational diagnostic, not a causal search root:
    the final outcome and future actions are never fed into predict().
    """
    actions: list[tuple[int, int, PublicHistoryStep, dict[str, Any]]] = []
    for i, step in enumerate(history):
        if step.chosen_action is None:
            continue
        public = json.loads(step.observation.payload_json)
        combat = public.get("combat") if isinstance(public, dict) else None
        if not isinstance(combat, dict):
            continue
        turn = combat.get("turn")
        if type(turn) is not int or turn < 1:
            raise ValueError("Missing public turn number in H15 final-turn trace")
        actions.append((i, turn, step, public))
    if not actions:
        raise ValueError("Completed H15 fight has no player decisions")
    final_turn = actions[-1][1]
    indices = [(i, st, state) for i, turn, st, state in actions if turn == final_turn]

    records: list[dict[str, Any]] = []
    for decision_index, step, state in indices:
        menu = step.legal_actions
        selected = step.chosen_action
        if not menu or selected not in menu:
            raise ValueError("Final-turn decision lacks a valid public action")
        prediction = teacher.predict(step.observation, menu)
        if len(prediction.logits) != len(menu):
            raise ValueError("Final-turn policy logits do not align with public menu")
        peak = max(prediction.logits)
        normalizer = math.fsum(math.exp(x - peak) for x in prediction.logits)
        index = menu.index(selected)
        selected_probability = math.exp(prediction.logits[index] - peak) / normalizer
        greedy = max(
            range(len(menu)),
            key=lambda j: (prediction.logits[j], menu[j].action_id),
        )
        combat = state["combat"]
        enemies = combat.get("enemies", [])
        payload = json.loads(selected.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("Invalid selected public action payload")
        instance_id = payload.get("CardInstanceId")
        hand = combat.get("hand", [])
        visible_card = next(
            (card for card in hand
             if isinstance(card, dict) and card.get("instance_id") == instance_id),
            None,
        ) if instance_id is not None and isinstance(hand, list) else None
        records.append({
            "decision_index": decision_index,
            "turn": final_turn,
            "player_hp": state.get("hp"),
            "player_block": combat.get("player_block"),
            "energy": combat.get("energy"),
            "enemy_hp": [
                {"instance_id": enemy.get("instance_id"),
                 "hp": enemy.get("hp")}
                for enemy in enemies if isinstance(enemy, dict)
            ],
            "selected_action": {
                "action_id": selected.action_id,
                "kind": selected.kind,
                "payload_json": selected.payload_json,
                "card_id": (
                    visible_card.get("card_id") if visible_card else None
                ),
                "card_instance_id": instance_id,
                "target_enemy_id": payload.get("TargetEnemyId"),
            },
            "greedy_action_id": menu[greedy].action_id,
            "legal_action_count": len(menu),
            "selected_policy_probability": selected_probability,
            "predicted_value_normalized": prediction.value,
            "predicted_encounter_reward": 2.0 * prediction.value,
            "neural_selection_fallback": (
                "h15_player_choice_kind" in state
            ),
        })
    return {"turn": final_turn, "actions": records}


def replay_native_public_history(
    backend: EmulatorBackend, recipe: CombatSnapshotRecipe,
) -> tuple[StateHandle, tuple[PublicHistoryStep, ...]]:
    """Replay H10's exact source recipe, recording what the player saw.

    The returned handle is owned by the caller. On any failure, release it.
    Snapshot hashes are verifier-only; the search actor never sees them.
    """
    recipe.validate()
    if (recipe.emulator_revision != backend.emulator_revision
            or recipe.ascension != 10):
        raise ValueError("Native source recipe revision/A10 mismatch")
    current = reset_training_run(
        backend, recipe.seed, NATIVE_OVERGROWTH, recipe.ascension,
    )
    trail: list[PublicHistoryStep] = []
    try:
        for action_id in recipe.action_ids:
            observation = backend.observe(current, POLICY)
            menu = tuple(backend.legal_actions(current))
            selected = next(
                (action for action in menu if action.action_id == action_id), None,
            )
            if selected is None:
                raise ValueError("Recorded macro/combat action missing from public menu")
            trail.append(PublicHistoryStep(observation, selected, menu))
            child = backend.step(current, selected).child
            backend.release_many([current])
            current = child
        _verify(backend, current, recipe)
        observation = backend.observe(current, POLICY)
        public = json.loads(observation.payload_json)
        if not isinstance(public, dict) or not isinstance(public.get("combat"), dict):
            raise ValueError("Native recipe did not arrive at a live combat")
        menu = tuple(backend.legal_actions(current))
        if not menu:
            raise ValueError("Native combat entry has no legal player action")
        trail.append(PublicHistoryStep(observation, None, menu))
        return current, tuple(trail)
    except BaseException:
        backend.release_many([current])
        raise


def collect_native_combat_search(
    backend: EmulatorBackend, recipe: CombatSnapshotRecipe, *,
    fair_sampler: FairContinuationSampler, teacher: H15CombatEvaluator,
    simulations: int = 32, max_decisions: int = 512, max_searches: int = 8,
    seed: int = 20261029, tree_depth: int = 16,
    rollout_depth: int = 512, leaf_mode: str = "terminal",
    search_schedule: str = "first",
    record_final_turn: bool = False,
    late_value_max: int = 0, late_value_min_turn: int = 7,
    late_value_player_hp: int = 25, late_value_enemy_hp: int = 40,
) -> NativeSearchCollection:
    """Search only selected moves; label all saved roots after actual combat.

    With the current H9 teacher, root visit counts guide policy improvements;
    the REAL completed fight supplies the common critic target. Remaining
    decisions use the frozen network's greedy legal action. A capped fight
    contributes no data. The fair sampler is mandatory and independent from
    the actor's actual native state handle.
    """
    if (min(simulations, max_decisions, tree_depth, rollout_depth) < 1
            or max_searches < 0 or seed < 0
            or leaf_mode not in ("terminal", "value")
            or search_schedule not in ("first", "turn_start")
            or late_value_max < 0 or late_value_min_turn < 1
            or late_value_player_hp < 0 or late_value_enemy_hp < 0):
        raise ValueError("Invalid native combat-search budget")
    # Reject unsupported samplers BEFORE replaying or owning a live native
    # state, so a missing capability cannot leak the source combat handle.
    require_fair_sampler(fair_sampler)
    current, initial_history = replay_native_public_history(backend, recipe)
    return collect_live_combat_search(
        backend, current, initial_history,
        fair_sampler=fair_sampler, teacher=teacher, source_seed=recipe.seed,
        entry_hash=recipe.exact_hash, encounter_tier=recipe.tier,
        simulations=simulations, max_decisions=max_decisions,
        max_searches=max_searches, seed=seed, tree_depth=tree_depth,
        rollout_depth=rollout_depth, leaf_mode=leaf_mode,
        search_schedule=search_schedule,
        record_final_turn=record_final_turn,
        late_value_max=late_value_max,
        late_value_min_turn=late_value_min_turn,
        late_value_player_hp=late_value_player_hp,
        late_value_enemy_hp=late_value_enemy_hp,
    )


def collect_live_combat_search(
    backend: EmulatorBackend, current: StateHandle,
    initial_history: tuple[PublicHistoryStep, ...], *,
    fair_sampler: FairContinuationSampler, teacher: H15CombatEvaluator,
    source_seed: str, entry_hash: str, encounter_tier: str,
    simulations: int = 32, max_decisions: int = 512, max_searches: int = 8,
    seed: int = 20261029, tree_depth: int = 16,
    rollout_depth: int = 512, leaf_mode: str = "terminal",
    search_schedule: str = "first",
    record_final_turn: bool = False,
    late_value_max: int = 0, late_value_min_turn: int = 7,
    late_value_player_hp: int = 25, late_value_enemy_hp: int = 40,
) -> NativeSearchCollection:
    """Consume a verified public combat entry and own/release its handle.

    Native A10 source recipes and independently initialized H8 reference
    fixtures both use this loop; the fair sampler is supplied explicitly.
    """
    if (min(simulations, max_decisions, tree_depth, rollout_depth) < 1
            or max_searches < 0 or seed < 0
            or leaf_mode not in ("terminal", "value")
            or search_schedule not in ("first", "turn_start")
            or late_value_max < 0 or late_value_min_turn < 1
            or late_value_player_hp < 0 or late_value_enemy_hp < 0):
        raise ValueError("Invalid H15 combat collection/evaluation budget")
    history = list(initial_history)
    adapter = H15PublicCombatAdapter(backend, sampler=fair_sampler)
    search = CombatPuct(
        adapter, teacher, seed=seed, max_depth=tree_depth,
        rollout_depth=rollout_depth, leaf_mode=leaf_mode,
    )
    saved: list[tuple[CombatFrame, Any]] = []
    search_diagnostics: list[dict[str, Any]] = []
    last_searched_turn: int | None = None
    count = 0
    try:
        while True:
            observation = history[-1].observation
            public = json.loads(observation.payload_json)
            if not isinstance(public, dict):
                raise ValueError("Invalid native public combat frame")
            if not isinstance(public.get("combat"), dict):
                score = H15PublicCombatAdapter._combat_result(public)
                if score is None:
                    raise RuntimeError("Combat exited without terminal score")
                outcome = "defeat" if public.get("terminal_outcome") == "defeat" or (
                    public.get("hp") == 0
                ) else "victory"
                return NativeSearchCollection(
                    source_seed=source_seed,
                    entry_hash=entry_hash,
                    encounter_tier=encounter_tier,
                    outcome=outcome,
                    reward=score,
                    exit_hp=public["hp"],
                    combat_decisions=count,
                    searched_decisions=len(saved),
                    searched_root_diagnostics=tuple(search_diagnostics),
                    final_player_turn=(
                        summarize_final_player_turn(history, teacher)
                        if record_final_turn else None
                    ),
                    targets=(
                        tuple(target_record(
                            root, stats, terminal_reward=score,
                            search_model_sha256=teacher.checkpoint_sha256,
                            source_seed=source_seed,
                            emulator_revision=backend.emulator_revision,
                        ) for root, stats in saved)
                        + tuple(value_target_record(
                            history[i].observation, history[i].legal_actions,
                            terminal_reward=score,
                            search_model_sha256=teacher.checkpoint_sha256,
                            source_seed=source_seed,
                            emulator_revision=backend.emulator_revision,
                            decision_index=i,
                        ) for i in select_late_value_indices(
                            history,
                            min_turn=late_value_min_turn,
                            max_player_hp=late_value_player_hp,
                            max_enemy_hp=late_value_enemy_hp,
                            max_samples=late_value_max,
                            exclude={
                                d["decision_index"] for d in search_diagnostics
                            },
                        ))
                    ),
                )
            if count >= max_decisions:
                raise RuntimeError(
                    "Actual native combat was censored; refusing incomplete targets"
                )
            menu = tuple(backend.legal_actions(current))
            if menu != history[-1].legal_actions:
                raise ValueError("Actual legal menu drifted from recorded public history")
            # H9's historical codec cannot represent some selections from
            # outside the hand. Search can traverse those public actions with
            # neutral priors, but they are not direct H15 training roots until
            # the selection-action codec is versioned and implemented.
            is_nested_choice = "h15_player_choice_kind" in public
            turn: int | None = None
            if search_schedule == "turn_start" and not is_nested_choice:
                turn = public["combat"].get("turn")
                if type(turn) is not int or turn < 1:
                    raise ValueError("H15 turn-start search needs public combat turn")
            search_here = (
                len(saved) < max_searches
                and not is_nested_choice
                and (search_schedule == "first" or turn != last_searched_turn)
            )
            if search_here:
                root = adapter.root(history)
                stats = search.search(root, simulations=simulations)
                if (leaf_mode == "terminal" and
                        (stats.neural_cutoffs != 0 or
                         stats.terminal_evaluations != simulations)):
                    raise ValueError("H15 search did not finish every sampled combat")
                index = max(
                    range(len(menu)),
                    key=lambda i: (stats.visits[i], menu[i].action_id),
                )
                # Measure actual search interventions against the *same*
                # checkpoint's raw greedy policy at this public decision.
                logits = teacher.predict(observation, menu).logits
                if len(logits) != len(menu) or not all(
                    math.isfinite(logit) for logit in logits
                ):
                    raise ValueError("H15 search root has invalid greedy logits")
                greedy_index = max(
                    range(len(menu)),
                    key=lambda i: (logits[i], menu[i].action_id),
                )
                # Work in log-space: tiny logits may underflow when
                # exponentiated, although a PUCT edge can still be visited.
                peak = max(logits)
                log_z = peak + math.log(math.fsum(
                    math.exp(logit - peak) for logit in logits
                ))
                visit_sum = sum(stats.visits)
                if visit_sum <= 0:
                    raise ValueError("H15 search root returned no visits")
                prior_kl = math.fsum(
                    (n / visit_sum) * (
                        math.log(n / visit_sum) - (logits[i] - log_z)
                    )
                    for i, n in enumerate(stats.visits) if n > 0
                )
                search_diagnostics.append({
                    "decision_index": count,
                    "public_observation_hash": observation.observation_hash,
                    "greedy_action_id": menu[greedy_index].action_id,
                    "search_action_id": menu[index].action_id,
                    "search_overrode_greedy": index != greedy_index,
                    "visit_policy_kl_from_prior": prior_kl,
                    "search_simulations": stats.simulations,
                    "search_neural_cutoffs": stats.neural_cutoffs,
                    "search_terminal_evaluations": stats.terminal_evaluations,
                    "root_visits": list(stats.visits),
                    "root_mean_action_values": list(stats.mean_action_values),
                    "root_mean_value": stats.root_mean_value,
                    "search_expanded_information_sets": stats.expanded_information_sets,
                    "search_transitions": stats.transitions,
                })
                saved.append((root, stats))
                if search_schedule == "turn_start":
                    last_searched_turn = turn
            else:
                logits = teacher.predict(observation, menu).logits
                index = max(
                    range(len(menu)), key=lambda i: (logits[i], menu[i].action_id),
                )
            chosen: LegalAction = menu[index]
            history[-1] = PublicHistoryStep(observation, chosen, menu)
            child = backend.step(current, chosen).child
            if child != current:
                backend.release_many([current])
            elif getattr(backend, "mutable_transition_capability_id", None) != (
                    "h15-h8-inplace-player-step-v1"):
                raise ValueError("Uncertified in-place native combat transition")
            current = child
            next_observation = backend.observe(current, POLICY)
            next_public = json.loads(next_observation.payload_json)
            if not isinstance(next_public, dict):
                raise ValueError("Invalid next public observation")
            next_menu = (
                tuple(backend.legal_actions(current))
                if isinstance(next_public.get("combat"), dict) else ()
            )
            history.append(PublicHistoryStep(next_observation, None, next_menu))
            count += 1
    finally:
        backend.release_many([current])
