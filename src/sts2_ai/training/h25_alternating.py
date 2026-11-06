"""H25 alternates full-Act macro reinforcement with offline full-combat PUCT.

The macro and combat models have independent parameters. Full-Act experience
uses an entirely frozen combat actor and no PUCT. Search replays an exact
observed action prefix into the native emulator and checks every public
observation before branching. Each selected fight searches *every* eligible
combat decision at full depth, with one fixed simulation budget per root.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.emulator.run_environment import NATIVE_OVERGROWTH, reset_training_run
from sts2_ai.training.h16_joint_act_pilot import (
    PUBLIC_POLICY,
    PilotEpisode,
    PilotPolicy,
    _sample,
    local_combat_puct,
    public_state,
)


class PrefixReplayMismatch(ValueError):
    """A recorded native rollout cannot be reproduced at this decision."""


@dataclass(frozen=True)
class CombatSearchExample:
    observation: Observation
    actions: tuple[LegalAction, ...]
    visits: tuple[float, ...]
    source_seed: str
    source_decision: int


@dataclass(frozen=True)
class CombatSearchResult:
    examples: tuple[CombatSearchExample, ...]
    visited_decisions: int
    searched_roots: int
    search_transitions: int
    complete_fight: bool


class SplitPilotPolicy:
    """Combine independent macro and combat actors with a combat HP prior.

    The macro critic is always the source of Act continuation values.
    The HP bonus applies to *combat search evaluation*, not to macro
    RL labels or the macro critic's calibrated clear probability.
    """

    def __init__(
        self, macro: PilotPolicy, combat: PilotPolicy, *,
        hp_bias: float = 0.15,
    ) -> None:
        if (not math.isfinite(hp_bias) or not 0 <= hp_bias <= 1
                or macro.temperature != combat.temperature):
            raise ValueError("Invalid H25 value mixing or policy temperature")
        self.macro = macro
        self.combat = combat
        self.hp_bias = hp_bias
        self.temperature = macro.temperature
        self.combat_credit_weight = combat.combat_credit_weight
        self.loss_progress_cap = combat.loss_progress_cap

    @property
    def inference_calls(self) -> int:
        return self.macro.inference_calls + self.combat.inference_calls

    @inference_calls.setter
    def inference_calls(self, value: int) -> None:
        if value != 0:
            raise ValueError("Only reset of H25 inference counters is allowed")
        self.macro.inference_calls = self.combat.inference_calls = 0

    @property
    def inference_seconds(self) -> float:
        return self.macro.inference_seconds + self.combat.inference_seconds

    @inference_seconds.setter
    def inference_seconds(self, value: float) -> None:
        if value != 0:
            raise ValueError("Only reset of H25 inference counters is allowed")
        self.macro.inference_seconds = self.combat.inference_seconds = 0.0

    def _combat_value(self, observation: Observation, continuation: float) -> float:
        state = public_state(observation)
        hp, maximum = state.get("hp"), state.get("max_hp")
        if type(hp) is not int or type(maximum) is not int or maximum <= 0:
            raise ValueError("H25 requires public HP and positive maximum HP")
        fraction = min(1.0, max(0.0, hp / maximum))
        return (1.0 - self.hp_bias) * continuation + self.hp_bias * fraction

    def infer(
        self, observation: Observation, actions: tuple[LegalAction, ...],
    ) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
        combat = isinstance(public_state(observation).get("combat"), dict)
        if not combat:
            return self.macro.infer(observation, actions)
        scores, _, offsets = self.combat.infer(observation, actions)
        _, macro_value, _ = self.macro.infer(observation, actions)
        return scores, self._combat_value(observation, macro_value), offsets

    def value_at(
        self, observation: Observation, actions: tuple[LegalAction, ...],
        *, terminal: bool = False,
    ) -> float:
        from sts2_ai.emulator.episode_goal import certified_native_act1_clear

        state = public_state(observation)
        if certified_native_act1_clear(state):
            return 1.0
        if terminal:
            return 0.0
        _, macro_value, _ = self.macro.infer(observation, actions)
        return self._combat_value(observation, macro_value)


def sample_fight_starts(
    episode: PilotEpisode, *, count: int, seed: int,
) -> tuple[int, ...]:
    """Deterministically sample whole encountered combats, not arbitrary turns."""
    if count < 0:
        raise ValueError("Negative H25 fight sampling budget")
    starts = [
        i for i in range(len(episode.decisions))
        if isinstance(episode.frames[i].get("combat"), dict)
        and (i == 0 or not isinstance(episode.frames[i - 1].get("combat"), dict))
    ]
    rng = random.Random(seed)
    rng.shuffle(starts)
    return tuple(sorted(starts[:count]))


def replay_to_combat_start(
    backend: Any, episode: PilotEpisode, decision_index: int, *,
    ascension: int = 0, map_profile_id: str | None = None,
) -> tuple[Any, Observation, tuple[LegalAction, ...]]:
    """Return an owned native handle only if full action-prefix replay matches.

    The caller must release the returned handle. This fails closed when
    native RNG or pending actions diverge; public states are not restart handles.
    """
    if (not 0 <= decision_index < len(episode.decisions)
            or not isinstance(episode.frames[decision_index].get("combat"), dict)):
        raise ValueError("Selected H25 state must be a recorded combat decision")
    handle = reset_training_run(
        backend, episode.seed, NATIVE_OVERGROWTH, ascension,
        map_profile_id=map_profile_id,
        compact_card_selection=map_profile_id is None,
    )
    try:
        for i in range(decision_index + 1):
            observed = backend.observe(handle, PUBLIC_POLICY)
            if public_state(observed) != episode.frames[i]:
                raise PrefixReplayMismatch(
                    f"Native public observation diverges at decision {i}"
                )
            legal = tuple(backend.legal_actions(handle))
            recorded = episode.decisions[i]
            if len(legal) != len(recorded.legal_actions) or any(
                (a.action_id, a.kind, a.payload_json)
                != (b.action_id, b.kind, b.payload_json)
                for a, b in zip(legal, recorded.legal_actions, strict=True)
            ):
                raise PrefixReplayMismatch(
                    f"Native legal action menu diverges at decision {i}"
                )
            if i == decision_index:
                return handle, observed, legal
            action = legal[recorded.chosen_index]
            child = backend.step(handle, action).child
            backend.release_many([handle])
            handle = child
    except BaseException:
        backend.release_many([handle])
        raise
    raise AssertionError("Unreachable H25 prefix replay")


def collect_searched_fight(
    backend: Any, episode: PilotEpisode, start_index: int,
    policy: SplitPilotPolicy, *, ascension: int,
    map_profile_id: str | None, simulations: int,
    seed: int, max_decisions: int = 256,
    max_simulation_steps: int = 512,
    on_root: Callable[[int, int, int], None] | None = None,
) -> CombatSearchResult:
    """Full-depth PUCT at every eligible root of one sampled complete fight."""
    if simulations <= 0 or max_decisions <= 0:
        raise ValueError("Invalid H25 search budget")
    handle, observation, actions = replay_to_combat_start(
        backend, episode, start_index, ascension=ascension,
        map_profile_id=map_profile_id,
    )
    rng = random.Random(seed)
    examples: list[CombatSearchExample] = []
    decisions = roots = transitions = 0
    complete = False
    try:
        while isinstance(public_state(observation).get("combat"), dict):
            if decisions >= max_decisions:
                break
            if not actions:
                raise ValueError("Native combat has no legal actions")
            scores, _, _ = policy.infer(observation, actions)
            if len(actions) >= 2:
                visits, work = local_combat_puct(
                    backend, handle, observation, actions, policy,
                    simulations=simulations, depth=None,
                    seed=rng.randrange(2**63),
                    max_simulation_steps=max_simulation_steps,
                )
                examples.append(CombatSearchExample(
                    observation, actions, visits, episode.seed, start_index,
                ))
                chosen = _sample(visits, rng)
                roots += 1
                transitions += work
                if on_root is not None:
                    on_root(roots, transitions, decisions)
            else:
                chosen = 0
            next_frame, = backend.batch_rollout_step_frame(
                ((handle, actions[chosen]),), PUBLIC_POLICY,
            )
            child = next_frame.transition.child
            backend.release_many([handle])
            handle = child
            observation, actions = next_frame.observation, next_frame.legal_actions
            decisions += 1
            if next_frame.transition.terminal:
                break
        complete = not isinstance(public_state(observation).get("combat"), dict)
        return CombatSearchResult(
            tuple(examples), decisions, roots, transitions, complete,
        )
    finally:
        backend.release_many([handle])


def fit_combat_search(
    model: Any, optimizer: torch.optim.Optimizer,
    examples: Sequence[CombatSearchExample], *,
    minibatch: int = 8, epochs: int = 1, temperature: float = 0.85,
) -> dict[str, float | int]:
    """Combat-only supervised PUCT distillation; no macro or HP regression."""
    if minibatch < 1 or epochs < 1 or temperature <= 0:
        raise ValueError("Invalid H25 distillation settings")
    if not examples:
        return {"examples": 0, "steps": 0, "loss": 0.0}
    device = next(model.parameters()).device
    model.train()
    total = 0.0
    steps = 0
    ordered = list(examples)
    rng = random.Random(25001 + len(examples))
    for _ in range(epochs):
        rng.shuffle(ordered)
        for start in range(0, len(ordered), minibatch):
            chunk = ordered[start:start + minibatch]
            encoded = [
                model.encode_decision(public_state(e.observation), e.actions)
                for e in chunk
            ]
            batch = model.collate(encoded).to(device)
            # Distillation only requires policy logits, so skip the
            # resource-neutral value backbone pass.
            logits = model.backbone(batch.policy)[0]
            logits = (logits / temperature).masked_fill(~batch.action_mask, -1e4)
            losses = []
            for i, e in enumerate(chunk):
                target = torch.tensor(
                    e.visits, dtype=logits.dtype, device=device,
                )
                if (len(target) != len(e.actions)
                        or not bool(torch.isfinite(target).all())
                        or bool((target < 0).any())
                        or abs(float(target.sum()) - 1) > 1e-4):
                    raise ValueError("Invalid H25 search distribution")
                losses.append(-(
                    target * F.log_softmax(logits[i, :len(target)], dim=-1)
                ).sum())
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("H25 combat distillation nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True,
            )
            optimizer.step()
            steps += 1
            total += float(loss.detach())
    model.eval()
    return {"examples": len(examples), "steps": steps, "loss": total / steps}


def fit_macro_reinforce(
    model: Any, optimizer: torch.optim.Optimizer,
    episodes: Sequence[PilotEpisode], *,
    minibatch: int = 8, epochs: int = 1, temperature: float = 0.85,
    entropy_weight: float = 0.001,
) -> dict[str, float | int]:
    """Act-clear REINFORCE and calibrated value BCE on macro decisions only."""
    if minibatch < 1 or epochs < 1 or temperature <= 0:
        raise ValueError("Invalid H25 macro optimization settings")
    samples = [
        (d, float(ep.outcome == "act_clear"))
        for ep in episodes if ep.complete for d in ep.decisions
        if not isinstance(public_state(d.observation).get("combat"), dict)
        and d.search_policy is None
    ]
    if not samples:
        return {"examples": 0, "steps": 0, "loss": 0.0}
    device = next(model.parameters()).device
    model.train()
    rng = random.Random(25002 + len(samples))
    total = 0.0
    steps = 0
    for _ in range(epochs):
        rng.shuffle(samples)
        for start in range(0, len(samples), minibatch):
            chunk = samples[start:start + minibatch]
            frames = [
                model.encode_decision(public_state(d.observation), d.legal_actions)
                for d, _ in chunk
            ]
            batch = model.collate(frames).to(device)
            policy_logits, value_logits, _ = model(
                batch, value_logits=True,
            )
            offsets = torch.zeros_like(policy_logits)
            for i, (decision, _) in enumerate(chunk):
                offsets[i, :len(decision.macro_offset)] = torch.tensor(
                    decision.macro_offset, dtype=policy_logits.dtype, device=device,
                )
            logits = ((policy_logits + offsets) / temperature).masked_fill(
                ~batch.action_mask, -1e4,
            )
            logp = F.log_softmax(logits, dim=-1)
            target = torch.tensor(
                [outcome for _, outcome in chunk],
                device=device, dtype=value_logits.dtype,
            )
            baseline = torch.sigmoid(value_logits.detach())
            pg = torch.stack([
                -(target[i] - baseline[i]) * logp[i, decision.chosen_index]
                for i, (decision, _) in enumerate(chunk)
            ]).mean()
            critic = F.binary_cross_entropy_with_logits(value_logits, target)
            entropy = -(logp.exp() * logp).masked_fill(
                ~batch.action_mask, 0,
            ).sum(-1).mean()
            loss = pg + critic - entropy_weight * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("H25 macro reinforcement nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True,
            )
            optimizer.step()
            steps += 1
            total += float(loss.detach())
    model.eval()
    return {"examples": len(samples), "steps": steps, "loss": total / steps}
