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
from sts2_ai.training.h15_search_supervision import target_record


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
    rollout_depth: int = 512,
) -> NativeSearchCollection:
    """Search only selected moves; label all saved roots after actual combat.

    With the current H9 teacher, root visit counts guide policy improvements;
    the REAL completed fight supplies the common critic target. Remaining
    decisions use the frozen network's greedy legal action. A capped fight
    contributes no data. The fair sampler is mandatory and independent from
    the actor's actual native state handle.
    """
    if (min(simulations, max_decisions, max_searches, tree_depth, rollout_depth) < 1
            or seed < 0):
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
        rollout_depth=rollout_depth,
    )


def collect_live_combat_search(
    backend: EmulatorBackend, current: StateHandle,
    initial_history: tuple[PublicHistoryStep, ...], *,
    fair_sampler: FairContinuationSampler, teacher: H15CombatEvaluator,
    source_seed: str, entry_hash: str, encounter_tier: str,
    simulations: int = 32, max_decisions: int = 512, max_searches: int = 8,
    seed: int = 20261029, tree_depth: int = 16,
    rollout_depth: int = 512,
) -> NativeSearchCollection:
    """Consume a verified public combat entry and own/release its handle.

    Native A10 source recipes and independently initialized H8 reference
    fixtures both use this loop; the fair sampler is supplied explicitly.
    """
    history = list(initial_history)
    adapter = H15PublicCombatAdapter(backend, sampler=fair_sampler)
    search = CombatPuct(
        adapter, teacher, seed=seed, max_depth=tree_depth,
        rollout_depth=rollout_depth, leaf_mode="terminal",
    )
    saved: list[tuple[CombatFrame, Any]] = []
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
                    targets=tuple(target_record(
                        root, stats, terminal_reward=score,
                        search_model_sha256=teacher.checkpoint_sha256,
                        source_seed=source_seed,
                        emulator_revision=backend.emulator_revision,
                    ) for root, stats in saved),
                )
            if count >= max_decisions:
                raise RuntimeError(
                    "Actual native combat was censored; refusing incomplete targets"
                )
            menu = tuple(backend.legal_actions(current))
            if menu != history[-1].legal_actions:
                raise ValueError("Actual legal menu drifted from recorded public history")
            if len(saved) < max_searches:
                root = adapter.root(history)
                stats = search.search(root, simulations=simulations)
                if (stats.neural_cutoffs != 0
                        or stats.terminal_evaluations != simulations):
                    raise ValueError("H15 search did not finish every sampled combat")
                index = max(
                    range(len(menu)),
                    key=lambda i: (stats.visits[i], menu[i].action_id),
                )
                saved.append((root, stats))
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
