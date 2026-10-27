"""Replay-checked real single-combat draw-only Nature episodes.

Everything after a selected outcome is executed by sts2-emulator's semantic
combat continuation. No hidden state, card order or synthetic combat actions
are created in Python. This is an engineering harness, not a training result.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from sts2_ai.emulator.semantic_p0_combat import strike_first_player
from sts2_ai.nature.combat_scenarios import (
    BASELINE_DECK,
    BASELINE_SCENARIO,
    DECK_PRESETS,
    SCENARIOS,
    CombatScenario,
)
from sts2_ai.nature.contextual_draw import ContextualDrawNature
from sts2_ai.nature.draw_only_p0 import (
    DRAW_ONLY_SCHEMA,
    DrawOnlyContractError,
    DrawOnlyRouter,
    SeededLawSampler,
)
from sts2_ai.nature.h4_reward import H4_REWARD, boss_progress_score, opening_enemy_health
from sts2_ai.nature.protocol import canonical_sha256
from sts2_ai.nature.two_site_opponent import PilotChoice, PilotEpisode, TwoSiteNature


class DrawOnlySession(Protocol):
    capabilities: dict[str, Any]

    def begin_seeded_semantic_route(
        self, seed: str, *, region: str,
        native_policy_baseline: bool,
    ) -> Any: ...

    def begin_seeded_semantic_combat(self, seed: str, **kwargs: Any) -> Any: ...

    def load_semantic_snapshot(self, snapshot: str) -> Any: ...
    def release_semantic_many(self, states: list[Any]) -> int: ...


@dataclass(frozen=True)
class DrawOnlyStep:
    phase: str
    support_id: str
    selected_id: str
    result_fingerprint: str
    law_id: str | None = None
    conditioning_fingerprint: str | None = None
    log_p: float | None = None
    log_q: float | None = None


@dataclass(frozen=True)
class DrawOnlyEpisode:
    schema: str
    emulator_revision: str
    source_seed: str
    reference_seed: int
    draw_mode: str
    player_model_id: str
    nature_revision: int
    source_opening_fingerprint: str
    first_public_fingerprint: str
    decisions: tuple[DrawOnlyStep, ...]
    nature_choices: tuple[PilotChoice, ...]
    reference_intent_count: int
    controlled_draw_count: int
    nontrivial_draw_count: int
    outcome: str
    exit_hp: int
    score: float
    terminal_fingerprint: str
    terminal_replay_verified: bool
    scenario_id: str = BASELINE_SCENARIO
    deck_preset_id: str = BASELINE_DECK
    reward_version: str = "hp_preservation_v1"
    opening_enemy_hp: int | None = None
    terminal_enemy_hp: int | None = None
    mcts_simulations: int = 0
    mcts_rollouts: int = 0
    mcts_search_boundaries: int = 0
    mcts_cutoff_rollouts: int = 0
    mcts_searched_decisions: int = 0
    mcts_greedy_fallback_decisions: int = 0

    @property
    def digest(self) -> str:
        return canonical_sha256(asdict(self))

    def for_nature_training(self) -> PilotEpisode:
        return PilotEpisode(player_return=self.score, choices=self.nature_choices)


def check_draw_only_capabilities(capabilities: dict[str, Any]) -> None:
    manifest = capabilities.get("stochasticCapabilities") or {}
    if (capabilities.get("semanticChanceSchemaId") !=
            "sts2-v111-semantic-playable-p0-v1"
            or capabilities.get("semanticPlayable") is not True
            or capabilities.get("semanticIntentSupportSchemaId") !=
            "sts2-v111-semantic-weighted-intent-v1"
            or not {"enemy_intent", "draw_card"}.issubset(set(
                capabilities.get("semanticControlledSites", [])
            ))
            or manifest.get("chanceLawSchemaId") != "sts2-v111-chance-law-v1"
            or capabilities.get("referenceNatureSessionSchemaId") !=
            "sts2-v111-reference-nature-session-v1"
            or capabilities.get("semanticSeededRouteEntryProviderId") !=
            "engine-owned-v111-native-weak-route-entry-unverified-v1"):
        raise DrawOnlyContractError("Emulator lacks unified draw-only law capabilities")


def _score(outcome: str, entry_hp: int, max_hp: int, exit_hp: int) -> float:
    # Mirror the existing v15 HP-preservation reward: no extra death bonus.
    if (type(entry_hp) is not int or type(max_hp) is not int
            or type(exit_hp) is not int or max_hp <= 0
            or entry_hp < 0 or entry_hp > max_hp or exit_hp < 0):
        raise DrawOnlyContractError("Invalid combat HP for player scoring")
    if outcome == "defeat":
        return 0.0
    if outcome != "victory":
        raise DrawOnlyContractError("Unknown genuine terminal outcome")
    return 1.0 - max(0, entry_hp - exit_hp) / max_hp



def fair_player_choice_actions(
    pending: dict[str, Any] | None, public_actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restrict a player's continuation to the *source* public choice menu.

    The engine owns a suspended card/hand-draw decision and publishes its
    exact action IDs. Player inference receives only fair action payloads,
    never the Nature-private event or its hidden deck state. Reject a mismatch
    between the pending source choice and the fair menu.
    """
    if not isinstance(pending, dict):
        raise DrawOnlyContractError("Player-choice boundary lacks pending choice")
    kind = pending.get("kind")
    if kind == "player_card_selection":
        candidates = pending.get("actions")
    elif kind == "draw_hook_action_choice":
        candidates = pending.get("candidates")
    else:
        raise DrawOnlyContractError(
            f"Unsupported semantic player-choice kind: {kind}"
        )
    if not isinstance(candidates, list) or not candidates:
        raise DrawOnlyContractError("Player-choice support is missing")
    if not isinstance(public_actions, list):
        raise DrawOnlyContractError("Fair player-choice action menu is missing")
    action_ids = [c.get("actionId") for c in candidates if isinstance(c, dict)]
    if (len(action_ids) != len(candidates)
            or any(not isinstance(x, str) or not x for x in action_ids)
            or len(set(action_ids)) != len(action_ids)):
        raise DrawOnlyContractError("Malformed player-choice candidate action IDs")
    fair_ids = [a.get("actionId") for a in public_actions if isinstance(a, dict)]
    if (len(fair_ids) != len(public_actions)
            or any(not isinstance(x, str) or not x for x in fair_ids)
            or len(set(fair_ids)) != len(fair_ids)):
        raise DrawOnlyContractError("Malformed fair player-choice action IDs")
    wanted = set(action_ids)
    if not wanted.issubset(fair_ids):
        raise DrawOnlyContractError(
            "Player-choice support is absent from the genuine fair action menu"
        )
    return [a for a in public_actions if a["actionId"] in wanted]


def play_draw_only_episode(
    session: DrawOnlySession, *, source_seed: str, reference_seed: int,
    nature: TwoSiteNature | ContextualDrawNature,
    emulator_revision: str,
    player_choose: Callable[[dict[str, Any], list[dict[str, Any]]], str] =
        strike_first_player,
    player_model_id: str = "test-only-strike-first",
    region: str = "Overgrowth", max_boundaries: int = 600,
    entry_mode: str = "source-route",
    draw_mode: str = "learned",
    replay: tuple[DrawOnlyStep, ...] | None = None,
    scenario: CombatScenario | None = None,
    deck_preset_id: str = BASELINE_DECK,
    fixture_spec: dict[str, Any] | None = None,
    score_reference_q: bool = False,
    reward_version: str = "hp_preservation_v1",
    mcts_simulations: int = 0,
    mcts_max_boundaries: int = 24,
    mcts_combat_budget: int = 4000,
) -> DrawOnlyEpisode:
    """Run actual supported first-room mechanics until a genuine terminal.

    A replay checks the source action menu, pending law IDs, selected outcomes,
    and every resulting state fingerprint; terminal snapshot round-trips.
    Reference RNG is a separate declared-law sampler, never the native cursor.
    """
    check_draw_only_capabilities(session.capabilities)
    if reward_version not in ("hp_preservation_v1", H4_REWARD):
        raise ValueError("Unknown draw-only reward version")
    if (mcts_simulations < 0 or mcts_max_boundaries < 1
            or mcts_combat_budget < 1):
        raise ValueError("Invalid H4 MCTS simulation budget")
    if mcts_simulations and (reward_version != H4_REWARD or replay is not None):
        raise ValueError("MCTS evaluation needs H4 reward and non-replay mode")
    if (draw_mode not in ("learned", "reference")
            or entry_mode not in ("source-route", "seeded-fixed-shrinker")
            or region not in ("Overgrowth", "Underdocks") or not source_seed
            or not emulator_revision or not player_model_id
            or max_boundaries <= 0):
        raise ValueError("Invalid draw-only fixture configuration")
    selected = SCENARIOS[BASELINE_SCENARIO] if scenario is None else scenario
    if fixture_spec is not None and (
        entry_mode != "seeded-fixed-shrinker" or deck_preset_id != BASELINE_DECK
        or selected.encounter_id != fixture_spec.get("encounter_id")
        or selected.native_entry != fixture_spec.get("native_encounter_entry")
        or selected.total_floor != fixture_spec.get("total_floor")
    ):
        raise DrawOnlyContractError("Incompatible structured research fixture")
    if deck_preset_id not in DECK_PRESETS:
        raise DrawOnlyContractError("Unsupported source-conditioned Silent deck preset")
    if selected.scenario_id not in SCENARIOS or SCENARIOS[selected.scenario_id] != selected:
        raise DrawOnlyContractError("Unpinned challenge-combat scenario")
    if entry_mode == "source-route" and deck_preset_id != BASELINE_DECK:
        raise DrawOnlyContractError("Route entry cannot use a research deck preset")
    if entry_mode == "source-route" and selected.scenario_id != BASELINE_SCENARIO:
        raise DrawOnlyContractError(
            "Source-route entries do not support conditioned encounter overrides"
        )
    router = DrawOnlyRouter(nature, SeededLawSampler(reference_seed))
    if entry_mode == "seeded-fixed-shrinker":
        if region != "Overgrowth":
            raise ValueError("Fixed Shrinker Beetle fixture requires Overgrowth")
        # Engine-owned source-shaped conditioned formation; no external
        # deck/HP/RNG mutation and no native encounter-probability claim.
        deck_kwargs = (
            {} if deck_preset_id == BASELINE_DECK else
            {"fixture_deck_preset_id": deck_preset_id}
        )
        if fixture_spec is not None:
            deck_kwargs["fixture_spec"] = fixture_spec
        state = session.begin_seeded_semantic_combat(
            source_seed, act=selected.act,
            encounter_id=selected.encounter_id,
            native_entry=selected.native_entry,
            total_floor=selected.total_floor,
            ascension=selected.ascension,
            first_monster_combat_id=selected.first_monster_combat_id,
            map_column=selected.map_column, map_row=selected.map_row,
            replay_opening_draw_hooks=True, native_policy_baseline=False,
            **deck_kwargs,
        )
    else:
        state = session.begin_seeded_semantic_route(
            source_seed, region=region, native_policy_baseline=False,
        )
    held = [state]
    decisions: list[DrawOnlyStep] = []
    choices: list[PilotChoice] = []
    opening = state.status().get("sourceOpeningFingerprint")
    first_public = None
    entry_hp: int | None = None
    max_hp: int | None = None
    opening_enemy_hp: int | None = None
    mcts_rollouts = mcts_search_boundaries = mcts_cutoff_rollouts = 0
    mcts_searched_decisions = mcts_greedy_fallback_decisions = 0
    draw_count = nontrivial_count = reference_intent_count = 0
    try:
        if not isinstance(opening, str) or not opening:
            raise DrawOnlyContractError("Missing source-authentic opening")
        for ordinal in range(max_boundaries):
            if state.terminal:
                break
            if state.boundary == "nature":
                if state.pending is None:
                    raise DrawOnlyContractError("Nature boundary lacks an event")
                # Nature sees precisely the player's public state *before*
                # this unresolved draw, plus its own legal Chance Law support.
                # The existing tabular baseline keeps its old protocol.
                public_observation = (
                    state.observe_for_nature()
                    if isinstance(nature, ContextualDrawNature)
                    and (draw_mode == "learned" or score_reference_q)
                    and state.pending.get("kind") == "draw_card"
                    else None
                )
                picked = router.choose(
                    state.pending, fingerprint=state.fingerprint,
                    episode_key=source_seed, ordinal=ordinal,
                    draw_mode=draw_mode,
                    public_observation=public_observation,
                    score_reference_q=score_reference_q,
                )
                if picked.operation == "enemy_intent":
                    reference_intent_count += 1
                    child = state.resolve_intent(picked.selected_outcome_id)
                else:
                    draw_count += 1
                    nontrivial_count += len(state.pending["candidates"]) > 1
                    if picked.nature_choice is not None:
                        choices.append(picked.nature_choice)
                    instance = int(picked.selected_outcome_id.removeprefix("card_instance:"))
                    child = state.resolve_draw_card(instance)
                decision = DrawOnlyStep(
                    phase=f"nature:{picked.operation}",
                    support_id=picked.law_id,
                    selected_id=picked.selected_outcome_id,
                    result_fingerprint=child.fingerprint,
                    law_id=picked.law_id,
                    conditioning_fingerprint=picked.conditioning_fingerprint,
                    log_p=picked.log_p, log_q=picked.log_q,
                )
            elif state.boundary in ("player", "player_choice"):
                phase = state.boundary
                if phase == "player" and state.pending is not None:
                    raise DrawOnlyContractError("Player boundary retained hidden chance")
                obs, actions = state.observe(), state.actions()
                if phase == "player_choice":
                    actions = fair_player_choice_actions(state.pending, actions)
                if first_public is None:
                    first_public = canonical_sha256(obs)
                    public_hp, public_max_hp = obs.get("hp"), obs.get("max_hp")
                    if (type(public_hp) is not int
                            or type(public_max_hp) is not int or public_max_hp <= 0):
                        raise DrawOnlyContractError(
                            "Missing public combat entry HP/max HP"
                        )
                    entry_hp, max_hp = public_hp, public_max_hp
                    if fixture_spec is not None and (
                        entry_hp != fixture_spec["hp"]
                        or max_hp != fixture_spec["max_hp"]
                    ):
                        raise DrawOnlyContractError(
                            "Engine public HP differs from structured fixture"
                        )
                    if reward_version == H4_REWARD:
                        opening_enemy_hp = opening_enemy_health(obs)
                        if opening_enemy_hp is None:
                            raise DrawOnlyContractError(
                                "Missing public opening enemy health for H4 reward"
                            )
                menu_hash = canonical_sha256(
                    [obs, actions] if phase == "player" else
                    [obs, actions, state.pending]
                )
                if replay is not None:
                    if ordinal >= len(replay) or replay[ordinal].phase != phase:
                        raise DrawOnlyContractError("Player replay phase drift")
                    action_id = replay[ordinal].selected_id
                elif mcts_simulations and len(actions) > 1:
                    # The budget is per *real combat*, not per player action.
                    # Once depleted, return to greedy policy rather than
                    # spawning unbounded full-fight searches.
                    remaining = mcts_combat_budget - mcts_search_boundaries
                    local_horizon = min(
                        mcts_max_boundaries,
                        remaining // mcts_simulations,
                    )
                    if local_horizon >= 4:
                        from sts2_ai.nature.h4_mcts import search_player_action
                        if opening_enemy_hp is None or type(max_hp) is not int:
                            raise DrawOnlyContractError(
                                "H4 MCTS needs public opening and player HP"
                            )
                        action_id, search_stats = search_player_action(
                            state, observation=obs, legal_actions=actions,
                            rollout_choose=player_choose,
                            simulations=mcts_simulations,
                            seed=f"{source_seed}:{reference_seed}:{ordinal}",
                            opening_enemy_hp=opening_enemy_hp,
                            player_max_hp=max_hp,
                            max_boundaries=local_horizon,
                        )
                        mcts_searched_decisions += 1
                        mcts_rollouts += int(search_stats["simulations"])
                        mcts_search_boundaries += int(
                            search_stats["simulated_boundaries"])
                        mcts_cutoff_rollouts += int(
                            search_stats["cutoff_rollouts"])
                        if mcts_searched_decisions == 1 or (
                            mcts_searched_decisions % 8 == 0
                        ):
                            print(
                                f"[h4-uct-worker] seed={source_seed} "
                                f"searched={mcts_searched_decisions} "
                                f"simulated_boundaries={mcts_search_boundaries}/"
                                f"{mcts_combat_budget}",
                                flush=True,
                            )
                    else:
                        action_id = player_choose(obs, actions)
                        mcts_greedy_fallback_decisions += 1
                else:
                    action_id = player_choose(obs, actions)
                if action_id not in {a["actionId"] for a in actions}:
                    raise DrawOnlyContractError("Player action outside genuine legal menu")
                child = state.step(action_id)
                decision = DrawOnlyStep(
                    phase=phase, support_id=menu_hash,
                    selected_id=action_id, result_fingerprint=child.fingerprint,
                )
            else:
                raise DrawOnlyContractError(
                    f"Unimplemented semantic boundary: {state.boundary}"
                )
            if replay is not None and (
                ordinal >= len(replay) or replay[ordinal] != decision
            ):
                raise DrawOnlyContractError("Draw-only replay transition mismatch")
            decisions.append(decision)
            held.append(child)
            state = child
        if not state.terminal:
            raise DrawOnlyContractError("CENSORED: no terminal combat before cap")
        if (state.outcome not in ("victory", "defeat") or
                type(state.exit_hp) is not int or state.boundary != "terminal"
                or state.pending is not None or type(entry_hp) is not int
                or type(max_hp) is not int
                or first_public is None):
            raise DrawOnlyContractError("Malformed genuine terminal combat")
        if replay is not None and len(replay) != len(decisions):
            raise DrawOnlyContractError("Replay ended at different boundary count")
        restored = session.load_semantic_snapshot(state.save_snapshot())
        held.append(restored)
        if restored.fingerprint != state.fingerprint:
            raise DrawOnlyContractError("Private terminal snapshot replay mismatch")
        terminal_enemy_hp: int | None = None
        if reward_version == H4_REWARD:
            if opening_enemy_hp is None:
                raise DrawOnlyContractError("H4 reward lacks opening enemy health")
            terminal_value = state.status().get("terminalEnemyHp")
            if state.outcome == "defeat":
                if type(terminal_value) is not int:
                    raise DrawOnlyContractError(
                        "H4 defeat lacks public terminal enemy health"
                    )
                terminal_enemy_hp = terminal_value
            score = boss_progress_score(
                state.outcome, player_exit_hp=state.exit_hp,
                player_max_hp=max_hp, opening_enemy_hp=opening_enemy_hp,
                terminal_enemy_hp=terminal_enemy_hp,
            )
        else:
            score = _score(state.outcome, entry_hp, max_hp, state.exit_hp)
        return DrawOnlyEpisode(
            schema=DRAW_ONLY_SCHEMA, emulator_revision=emulator_revision,
            source_seed=source_seed, reference_seed=reference_seed,
            draw_mode=draw_mode,
            player_model_id=player_model_id, nature_revision=nature.revision,
            source_opening_fingerprint=opening,
            first_public_fingerprint=first_public,
            decisions=tuple(decisions), nature_choices=tuple(choices),
            reference_intent_count=reference_intent_count,
            controlled_draw_count=draw_count,
            nontrivial_draw_count=nontrivial_count,
            outcome=state.outcome, exit_hp=state.exit_hp,
            score=score,
            terminal_fingerprint=state.fingerprint,
            terminal_replay_verified=True,
            scenario_id=selected.scenario_id,
            deck_preset_id=deck_preset_id,
            reward_version=reward_version,
            opening_enemy_hp=opening_enemy_hp,
            terminal_enemy_hp=terminal_enemy_hp,
            mcts_simulations=mcts_simulations,
            mcts_rollouts=mcts_rollouts,
            mcts_search_boundaries=mcts_search_boundaries,
            mcts_cutoff_rollouts=mcts_cutoff_rollouts,
            mcts_searched_decisions=mcts_searched_decisions,
            mcts_greedy_fallback_decisions=mcts_greedy_fallback_decisions,
        )
    finally:
        session.release_semantic_many(held)
