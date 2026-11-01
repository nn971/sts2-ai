"""H16 pilot: one Act-long joint actor/critic with local combat PUCT.

The reward is an *Act-level* binary terminal outcome. A future-floor HP
prediction supplies temporary policy bootstrapping and remains an auxiliary
target. Both combat and macro heads update from the same completed runs.

Experimental PUCT uses exact native-structure emulator forks. It is a
ROOT-CONDITIONED, potentially privileged search teacher, never a certified
public-history fair posterior. No hidden state is passed to the network.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from sts2_ai.emulator import FAIR_POLICY_ID, InformationPolicy, LegalAction, Observation
from sts2_ai.emulator.episode_goal import certified_native_act1_clear
from sts2_ai.emulator.protocol import EmulatorBackend, StateHandle
from sts2_ai.emulator.run_environment import (
    EXPERIMENTAL_FIXED_ROUTE_PROFILE,
    NATIVE_OVERGROWTH,
    reset_training_run,
)
from sts2_ai.models.phase_split import PhaseSplitNeuralModel
from sts2_ai.models.shared_transformer import (
    H7Config,
    SharedObjectTransformer,
    collate_h7,
)
from sts2_ai.search.combat_puct import (
    H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY,
    CombatFrame,
    CombatPuct,
)
from sts2_ai.training.h16_public_codec import H16_CODEC, encode_h16_decision

SCHEMA = "sts2-h16-act1-geometric-horizon-every-combat-puct-v2"
PUBLIC_POLICY = InformationPolicy(FAIR_POLICY_ID)


def public_state(observation: Observation) -> dict[str, Any]:
    if observation.policy_id != FAIR_POLICY_ID:
        raise ValueError("Only fair public observations may enter the H16 model")
    state = json.loads(observation.payload_json)
    if not isinstance(state, dict):
        raise ValueError("Public observation must contain an object")
    return state


def hp_fraction(state: dict[str, Any]) -> float:
    hp, max_hp = state.get("hp"), state.get("max_hp")
    if type(hp) is not int or type(max_hp) is not int or max_hp <= 0:
        raise ValueError("Missing public player HP")
    return min(1.0, max(0.0, hp / max_hp))


def floor_completed(state: dict[str, Any], floor: int) -> bool:
    visits = state.get("completed_rooms")
    return isinstance(visits, list) and any(
        isinstance(room, dict) and room.get("act") == 1
        and room.get("floor") == floor
        for room in visits
    )


def hp_at_future_floor(
    frames: Sequence[dict[str, Any]], decision_index: int, *,
    horizon: int,
) -> float:
    """HP after a later completed room, or zero on an earlier terminal loss.

    All frames in this function must be from a COMPLETE Act episode. Frames
    after an earlier death are absent; no fabricated continuation is used.
    """
    if horizon <= 0 or not 0 <= decision_index < len(frames) - 1:
        raise ValueError("Invalid full-Act future HP query")
    floor = frames[decision_index].get("floor")
    if type(floor) is not int or not 0 <= floor <= 16:
        raise ValueError("Invalid starting floor")
    target = min(16, floor + horizon)
    for frame in frames[decision_index + 1:]:
        if floor_completed(frame, target):
            return hp_fraction(frame)
    return 0.0


def hp_after_current_room(
    frames: Sequence[dict[str, Any]], decision_index: int,
) -> float:
    """Normalized HP at the next room-completion boundary after a decision.

    During an unfinished room, credit the HP on completing that room. At a
    map-choice state the previous room is already completed; credit the
    *next* room instead, so the target depends on the route/room decision.
    Neow's floor-zero opening likewise targets the first completed room.
    Death before the boundary contributes zero.
    """
    if not 0 <= decision_index < len(frames) - 1:
        raise ValueError("Invalid post-room HP query")
    starting = frames[decision_index]
    floor = starting.get("floor")
    if type(floor) is not int or not 0 <= floor <= 16:
        raise ValueError("Invalid starting floor")

    if floor == 16 and floor_completed(starting, floor):
        # Any outstanding Act-transition decision is already beyond the
        # final room. Preserve the actual terminal HP instead of assigning
        # a nonexistent floor 17 a value of zero.
        return hp_fraction(frames[-1])

    target = floor + 1 if floor == 0 or floor_completed(starting, floor) else floor
    for frame in frames[decision_index + 1:]:
        if floor_completed(frame, target):
            return hp_fraction(frame)
    return 0.0


def completed_room_hp(frames: Sequence[dict[str, Any]]) -> dict[int, float]:
    """First observed post-completion HP for each Act-1 floor.

    Take the first observable boundary at which the room is completed;
    subsequent healing, rewards or room travel must not overwrite it.
    """
    completed: dict[int, float] = {}
    for state in frames:
        visits = state.get("completed_rooms")
        if not isinstance(visits, list):
            continue
        for room in visits:
            if not isinstance(room, dict) or room.get("act") != 1:
                continue
            floor = room.get("floor")
            if type(floor) is int and 1 <= floor <= 16 and floor not in completed:
                completed[floor] = hp_fraction(state)
    return completed


def remaining_room_hp(
    state: dict[str, Any], completions: dict[int, float],
) -> tuple[float, ...]:
    """HP after each room strictly following the player's present decision."""
    floor = state.get("floor")
    if type(floor) is not int or not 0 <= floor <= 16:
        raise ValueError("Invalid observed floor for geometric HP target")
    first = floor + 1 if floor == 0 or floor_completed(state, floor) else floor
    return tuple(completions.get(next_floor, 0.0) for next_floor in range(first, 17))


def geometric_hp_return(
    room_hp: Sequence[float], act_clear: float, horizon_lambda: float,
) -> float:
    """Smooth post-room HP -> Act-clear target on a finite Act.

    (1-lambda) * sum_{k=0}^{R-1} lambda**k * hp[k]
        + lambda**R * act_clear, with R remaining room completions.
    lambda=0 gives next post-room HP; lambda=1 gives Act clear.
    Every missing future room on a defeated trajectory has HP zero.
    """
    if (not 0.0 <= horizon_lambda <= 1.0
            or not math.isfinite(horizon_lambda)
            or act_clear not in (0.0, 1.0)
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in room_hp)):
        raise ValueError("Invalid geometric-horizon target inputs")
    result = act_clear
    for hp in reversed(room_hp):
        result = (1 - horizon_lambda) * hp + horizon_lambda * result
    return result


def _softmax(logits: Sequence[float], temperature: float) -> tuple[float, ...]:
    if temperature <= 0 or not logits or not all(math.isfinite(x) for x in logits):
        raise ValueError("Invalid action logits or temperature")
    peak = max(logits)
    weights = [math.exp((x - peak) / temperature) for x in logits]
    norm = math.fsum(weights)
    return tuple(x / norm for x in weights)


def _sample(weights: Sequence[float], rng: random.Random) -> int:
    if not weights or not all(math.isfinite(x) and x >= 0 for x in weights):
        raise ValueError("Invalid action weights")
    if math.fsum(weights) <= 0:
        raise ValueError("Empty action distribution")
    return rng.choices(range(len(weights)), weights=weights, k=1)[0]


class JointActPilotNet(nn.Module):
    """H9/H7 shared backbone plus two small continuation-value heads."""

    def __init__(self, backbone: SharedObjectTransformer) -> None:
        super().__init__()
        self.backbone = backbone
        dim = backbone.config.width
        self.act_clear = nn.Linear(dim, 1)
        # The same public state has different values at different horizons.
        # Conditioning on lambda and lambda**2 allows the head to learn
        # interactions between horizon and state, not merely a scalar offset.
        self.horizon_head = nn.Sequential(
            nn.Linear(dim + 2, dim), nn.GELU(), nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.act_clear.weight)
        nn.init.constant_(self.act_clear.bias, -1.5)
        nn.init.zeros_(self.horizon_head[-1].weight)
        nn.init.zeros_(self.horizon_head[-1].bias)

    def _horizon_logits(
        self, context: Tensor, horizon_lambda: float | Tensor,
    ) -> Tensor:
        if isinstance(horizon_lambda, Tensor):
            if horizon_lambda.shape != (context.shape[0],):
                raise ValueError("Horizon lambda tensor must have shape [batch]")
            lam = horizon_lambda.to(device=context.device, dtype=context.dtype)
            if not bool((torch.isfinite(lam) & (lam >= 0) & (lam <= 1)).all()):
                raise ValueError("Invalid per-state geometric horizon")
            lam = lam.unsqueeze(-1)
        else:
            if not math.isfinite(horizon_lambda) or not 0 <= horizon_lambda <= 1:
                raise ValueError("Geometric horizon must belong to [0,1]")
            lam = context.new_full((context.shape[0], 1), horizon_lambda)
        return self.horizon_head(
            torch.cat([context, lam, lam.square()], dim=-1),
        ).squeeze(-1)

    def forward(
        self, batch: Any, *, value_logits: bool = False,
        horizon_lambda: float | Tensor = 0.0,
        auxiliary_lambda: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        logits, _, context = self.backbone(batch, return_context=True)
        act = self.act_clear(context).squeeze(-1)
        hp = self._horizon_logits(context, horizon_lambda)
        if auxiliary_lambda is not None:
            aux_hp = self._horizon_logits(context, auxiliary_lambda)
            if value_logits:
                return logits, act, hp, aux_hp
            return (logits, torch.sigmoid(act), torch.sigmoid(hp),
                    torch.sigmoid(aux_hp))
        if value_logits:
            return logits, act, hp
        return logits, torch.sigmoid(act), torch.sigmoid(hp)

    def save(self, path: Path) -> None:
        from dataclasses import asdict
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": SCHEMA,
            "codec": H16_CODEC,
            "config": asdict(self.backbone.config),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> JointActPilotNet:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("schema") != SCHEMA or
                checkpoint.get("codec") != H16_CODEC):
            raise ValueError("Expected an H16 joint Act checkpoint/codec")
        model = cls(SharedObjectTransformer(H7Config(**checkpoint["config"])))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.to(device)


@dataclass(frozen=True)
class PilotDecision:
    observation: Observation
    legal_actions: tuple[LegalAction, ...]
    chosen_index: int
    macro_offset: tuple[float, ...]
    search_policy: tuple[float, ...] | None = None


@dataclass(frozen=True)
class PilotEpisode:
    seed: str
    decisions: tuple[PilotDecision, ...]
    frames: tuple[dict[str, Any], ...]
    outcome: str
    searches: int
    search_transitions: int
    search_overrides: int = 0

    def __post_init__(self) -> None:
        if (self.outcome not in ("act_clear", "defeat", "censored")
                or len(self.frames) != len(self.decisions) + 1):
            raise ValueError("Malformed full-Act trajectory")

    @property
    def complete(self) -> bool:
        return self.outcome != "censored"


def targets_for_episode(
    episode: PilotEpisode, horizon: int,
) -> tuple[tuple[float, float], ...]:
    if not episode.complete:
        raise ValueError("Censored runs have no terminal Act-clear or future-HP target")
    win = float(episode.outcome == "act_clear")
    return tuple(
        (win, hp_at_future_floor(episode.frames, i, horizon=horizon))
        for i in range(len(episode.decisions))
    )


class PilotPolicy:
    """Frozen cohort policy. Macro teacher is a fixed soft prior, if supplied."""

    def __init__(
        self, model: JointActPilotNet, *, temperature: float,
        horizon_lambda: float, macro_teacher: PhaseSplitNeuralModel | None = None,
        macro_prior_weight: float = 0.5,
    ) -> None:
        if not 0 <= horizon_lambda <= 1 or temperature <= 0 or macro_prior_weight < 0:
            raise ValueError("Invalid pilot continuation/policy weighting")
        self.model = model.eval()
        self.temperature = temperature
        self.horizon_lambda = horizon_lambda
        self.macro_teacher = macro_teacher
        self.macro_prior_weight = macro_prior_weight

    def _offsets(
        self, observation: Observation, actions: tuple[LegalAction, ...],
    ) -> tuple[float, ...]:
        if (self.macro_teacher is None or
                isinstance(public_state(observation).get("combat"), dict)):
            return (0.0,) * len(actions)
        raw = self.macro_teacher.evaluate(observation, actions).action_logits
        if len(raw) != len(actions):
            raise ValueError("Macro reference prior is not aligned to legal actions")
        peak = max(raw)
        norm = math.log(math.fsum(math.exp(v - peak) for v in raw))
        return tuple(self.macro_prior_weight * (v - peak - norm) for v in raw)

    def infer(
        self, observation: Observation, actions: tuple[LegalAction, ...],
    ) -> tuple[tuple[float, ...], float, tuple[float, ...]]:
        if not actions:
            raise ValueError("Nonterminal inference needs a legal menu")
        encoded = encode_h16_decision(public_state(observation), actions)
        batch = collate_h7([encoded]).to(next(self.model.parameters()).device)
        with torch.inference_mode():
            logits, _, hp = self.model(
                batch, horizon_lambda=self.horizon_lambda,
            )
        offsets = self._offsets(observation, actions)
        scores = tuple(
            float(logits[0, i]) + offset
            for i, offset in enumerate(offsets)
        )
        value = float(hp[0])
        if not all(math.isfinite(x) for x in (*scores, value)):
            raise ValueError("Nonfinite joint inference")
        return scores, min(1.0, max(0.0, value)), offsets

    def value_at(
        self, observation: Observation, actions: tuple[LegalAction, ...],
        *, terminal: bool = False,
    ) -> float:
        state = public_state(observation)
        if certified_native_act1_clear(state):
            return 1.0
        if terminal:
            return 0.0
        return self.infer(observation, actions)[1]


class NativeRootConditionedSampler:
    """Exact-emulator-fork search: deliberately NOT a fair stochastic posterior."""

    fair_combat_capability_id = H16_EXPERIMENTAL_ROOT_CONDITIONED_CAPABILITY

    def __init__(
        self, backend: EmulatorBackend, root: StateHandle,
        policy: PilotPolicy,
    ) -> None:
        self.backend = backend
        self.root = root
        self.policy = policy
        self._active: StateHandle | None = None
        self._owned: list[StateHandle] = []

    def begin_simulation(
        self, root: CombatFrame, search_rng: random.Random,
    ) -> None:
        if self._active is not None:
            raise RuntimeError("Search simulation was already active")
        self._active = self.backend.fork(self.root)
        self._owned = [self._active]

    def step(
        self, frame: CombatFrame, action: LegalAction,
        search_rng: random.Random,
    ) -> CombatFrame:
        if self._active is None:
            raise RuntimeError("Search simulation has not begun")
        transition = self.backend.step(self._active, action)
        self._active = transition.child
        self._owned.append(transition.child)
        observation = self.backend.observe(transition.child, PUBLIC_POLICY)
        public = public_state(observation)
        inside_combat = isinstance(public.get("combat"), dict)
        terminal = transition.terminal
        actions = (() if terminal else tuple(self.backend.legal_actions(transition.child)))
        if not terminal and not actions:
            raise ValueError("Nonterminal emulator fork has no legal actions")
        boundary_value = None
        if terminal or not inside_combat:
            boundary_value = 2.0 * self.policy.value_at(
                observation, actions, terminal=terminal,
            )
            actions = ()
        return CombatFrame(
            information_key=frame.information_key + "/" + action.action_id,
            observation=observation, legal_actions=actions,
            terminal_reward=boundary_value,
        )

    def end_simulation(self) -> None:
        try:
            if self._owned:
                self.backend.release_many(self._owned)
        finally:
            self._owned = []
            self._active = None


def local_combat_puct(
    backend: EmulatorBackend, state: StateHandle, observation: Observation,
    actions: tuple[LegalAction, ...], policy: PilotPolicy, *,
    simulations: int, depth: int, seed: int,
) -> tuple[tuple[float, ...], int]:
    """Short combat search with learned Act/future-HP cutoffs.

    The result is an exploratory policy teacher, not an unbiased Q estimate.
    """
    if len(actions) < 2 or not isinstance(public_state(observation).get("combat"), dict):
        raise ValueError("Local PUCT needs an ordinary combat choice")
    root = CombatFrame(
        information_key="native-experimental-root:" + observation.observation_hash,
        observation=observation, legal_actions=actions,
    )
    adapter = NativeRootConditionedSampler(backend, state, policy)
    def evaluator(
        obs: Observation, menu: tuple[LegalAction, ...],
    ) -> tuple[tuple[float, ...], float]:
        scores, value, _ = policy.infer(obs, menu)
        return scores, value
    result = CombatPuct(
        adapter, evaluator, max_depth=depth, c_puct=1.5,
        tree_expansion_mode="full_depth", leaf_mode="value", seed=seed,
    ).search(root, simulations=simulations)
    return result.policy, result.transitions


def should_search_combat(
    state: dict[str, Any], actions: Sequence[LegalAction], *,
    searches: int, max_searches: int, search_probability: float,
    rng: random.Random,
) -> bool:
    """Every ordinary combat decision with >=2 moves when cap=0, p=1.

    A single legal action has no meaningful PUCT root distribution.
    Zero probability disables search, including held-out evaluations.
    """
    if (max_searches != 0 and searches >= max_searches) or len(actions) < 2:
        return False
    if not isinstance(state.get("combat"), dict) or search_probability <= 0:
        return False
    return search_probability >= 1 or rng.random() < search_probability


def collect_act_episode(
    backend: EmulatorBackend, policy: PilotPolicy, *,
    seed: str, ascension: int, max_decisions: int,
    search_probability: float, max_searches: int,
    simulations: int, search_depth: int,
    rng: random.Random,
) -> PilotEpisode:
    if not (0 <= search_probability <= 1 and max_searches >= 0
            and max_decisions > 0 and simulations > 0 and search_depth > 0):
        raise ValueError("Invalid Act collection settings")
    state = reset_training_run(
        backend, seed, NATIVE_OVERGROWTH, ascension,
        map_profile_id=EXPERIMENTAL_FIXED_ROUTE_PROFILE,
    )
    decisions: list[PilotDecision] = []
    frames: list[dict[str, Any]] = []
    searches = search_transitions = search_overrides = 0
    try:
        while True:
            observation = backend.observe(state, PUBLIC_POLICY)
            public = public_state(observation)
            frames.append(public)
            if certified_native_act1_clear(public):
                outcome = "act_clear"
                break
            if backend.is_terminal(state):
                outcome = "defeat"
                break
            if len(decisions) >= max_decisions:
                outcome = "censored"
                break
            actions = tuple(backend.legal_actions(state))
            if not actions:
                raise RuntimeError("Nonterminal run has no legal player actions")
            scores, _, offsets = policy.infer(observation, actions)
            search_policy = None
            should_search = should_search_combat(
                public, actions, searches=searches,
                max_searches=max_searches,
                search_probability=search_probability, rng=rng,
            )
            if should_search:
                search_policy, count = local_combat_puct(
                    backend, state, observation, actions, policy,
                    simulations=simulations, depth=search_depth,
                    seed=rng.randrange(2**63),
                )
                search_transitions += count
                searches += 1
                greedy = max(range(len(scores)), key=lambda i: scores[i])
                searched_greedy = max(
                    range(len(search_policy)), key=lambda i: search_policy[i],
                )
                search_overrides += int(searched_greedy != greedy)
                chosen = _sample(search_policy, rng)
            else:
                chosen = _sample(_softmax(scores, policy.temperature), rng)
            decisions.append(PilotDecision(
                observation, actions, chosen, offsets, search_policy,
            ))
            transition = backend.step(state, actions[chosen])
            backend.release_many([state])
            state = transition.child
        return PilotEpisode(
            seed, tuple(decisions), tuple(frames),
            outcome, searches, search_transitions, search_overrides,
        )
    finally:
        backend.release_many([state])


def train_joint_cohort(
    model: JointActPilotNet, optimizer: torch.optim.Optimizer,
    episodes: Sequence[PilotEpisode], *, horizon_lambda: float,
    temperature: float = 0.85, minibatch: int = 32,
    epochs: int = 1, entropy_weight: float = 0.001,
    search_weight: float = 0.5, hp_target: str = "geometric",
    auxiliary_horizon_weight: float = 0.5,
) -> dict[str, float | int]:
    """One shared optimizer for both policies and horizon-conditioned critic.

    For each observed state, the actor trains at the current slowly-ramped
    horizon. The critic also trains at a separate random lambda using the
    *same completed run*. No additional environment simulation is needed.
    Search-executed combat moves receive visit-target CE, not a falsely
    on-policy REINFORCE term.
    """
    if (hp_target not in ("geometric", "post_room")
            or not 0 <= horizon_lambda <= 1 or temperature <= 0
            or minibatch <= 0 or epochs <= 0 or search_weight < 0
            or auxiliary_horizon_weight < 0):
        raise ValueError("Invalid joint training configuration")
    current_lambda = horizon_lambda if hp_target == "geometric" else 0.0
    entries: list[tuple[PilotDecision, float, tuple[float, ...]]] = []
    for episode in episodes:
        if not episode.complete:
            raise ValueError("Censored episodes cannot train geometric returns")
        completed = completed_room_hp(episode.frames)
        act = float(episode.outcome == "act_clear")
        for i, decision in enumerate(episode.decisions):
            rooms = remaining_room_hp(episode.frames[i], completed)
            entries.append((decision, act, rooms))
    if not entries:
        raise ValueError("No complete Act decisions for joint learning")
    rng = random.Random(31001 + len(entries))
    device = next(model.parameters()).device
    model.train()
    sums = {"loss": 0.0, "policy": 0.0, "act": 0.0, "hp": 0.0}
    updates = 0
    phase_counts = {"combat": 0, "macro": 0}
    search_roots = 0
    for decision, _, _ in entries:
        phase = "combat" if isinstance(
            public_state(decision.observation).get("combat"), dict,
        ) else "macro"
        phase_counts[phase] += 1
        search_roots += int(decision.search_policy is not None)
    for _ in range(epochs):
        rng.shuffle(entries)
        for start in range(0, len(entries), minibatch):
            chunk = entries[start:start + minibatch]
            encoded = [
                encode_h16_decision(
                    public_state(decision.observation), decision.legal_actions,
                ) for decision, _, _ in chunk
            ]
            batch = collate_h7(encoded).to(device)
            aux_lambdas = torch.tensor(
                [rng.random() for _ in chunk],
                dtype=torch.float32, device=device,
            )
            logits, act_logit, hp_logit, aux_hp_logit = model(
                batch, value_logits=True, horizon_lambda=current_lambda,
                auxiliary_lambda=aux_lambdas,
            )
            # Report non-finite values *before* launching CUDA BCE kernels;
            # a failed CUDA device assertion invalidates the whole process.
            if (not bool(torch.isfinite(logits[batch.action_mask]).all())
                    or not bool(torch.isfinite(act_logit).all())
                    or not bool(torch.isfinite(hp_logit).all())
                    or not bool(torch.isfinite(aux_hp_logit).all())):
                phases = [enc.phase for enc in encoded]
                floors = [
                    public_state(decision.observation).get("floor")
                    for decision, _, _ in chunk
                ]
                raise FloatingPointError(
                    "H16 nonfinite batched forward at "
                    f"minibatch={start // minibatch}, phases={phases}, "
                    f"floors={floors}, action_lengths="
                    f"{[len(d.legal_actions) for d, _, _ in chunk]}"
                )
            act_hat = torch.sigmoid(act_logit)
            hp_hat = torch.sigmoid(hp_logit)
            aux_hp_hat = torch.sigmoid(aux_hp_logit)
            offsets = torch.zeros_like(logits)
            for i, (decision, _, _) in enumerate(chunk):
                offsets[i, :len(decision.macro_offset)] = torch.tensor(
                    decision.macro_offset, device=device, dtype=logits.dtype,
                )
            if not bool(torch.isfinite(offsets[batch.action_mask]).all()):
                raise FloatingPointError("H16 nonfinite public macro prior offsets")
            # The H7 backbone pads unused legal moves at float32's minimum.
            # Replacing masked positions with a moderate finite constant
            # prevents numerical overflow in log-probability entropy gradients.
            policy_logits = ((logits + offsets) / temperature).masked_fill(
                ~batch.action_mask, -1e4,
            )
            logp = torch.log_softmax(policy_logits, dim=1)
            probabilities = logp.exp()
            entropy = -(probabilities * logp).masked_fill(
                ~batch.action_mask, 0.0,
            ).sum(dim=1).mean()
            aux_values = aux_lambdas.detach().cpu().tolist()
            targets = torch.tensor(
                [[act,
                  geometric_hp_return(room_hp, act, current_lambda),
                  geometric_hp_return(room_hp, act, aux)]
                 for (_, act, room_hp), aux in zip(chunk, aux_values, strict=True)],
                dtype=logits.dtype, device=device,
            )
            target_act, target_hp, target_aux_hp = (
                targets[:, 0], targets[:, 1], targets[:, 2],
            )
            from torch.nn import functional as F
            # Logit-space BCE avoids the fragile CUDA probability-range
            # assertion and retains gradients at saturated predictions.
            act_loss = F.binary_cross_entropy_with_logits(
                act_logit, target_act,
            )
            hp_loss = (
                F.mse_loss(hp_hat, target_hp)
                + auxiliary_horizon_weight * F.mse_loss(aux_hp_hat, target_aux_hp)
            )
            mixed_target = target_hp
            mixed_value = hp_hat.detach()
            ordinary: dict[str, list[Tensor]] = {"macro": [], "combat": []}
            search_ce: list[Tensor] = []
            for i, ((decision, _, _), enc) in enumerate(zip(chunk, encoded, strict=True)):
                if decision.search_policy is None:
                    phase = "combat" if enc.phase == "combat" else "macro"
                    ordinary[phase].append(
                        -(mixed_target[i] - mixed_value[i]) *
                        logp[i, decision.chosen_index]
                    )
                else:
                    distribution = torch.tensor(
                        decision.search_policy, dtype=logits.dtype, device=device,
                    )
                    search_ce.append(
                        -(distribution * logp[i, :len(distribution)]).sum()
                    )
            # Equal phase weight prevents hundreds of combat micro-actions
            # from drowning out the much rarer macro decisions.
            actor_terms = [torch.stack(v).mean() for v in ordinary.values() if v]
            if search_ce:
                actor_terms.append(search_weight * torch.stack(search_ce).mean())
            policy_loss = sum(actor_terms) if actor_terms else logits.sum() * 0.0
            loss = policy_loss + act_loss + hp_loss - entropy_weight * entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"H16 nonfinite training loss at minibatch={start // minibatch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Crucial: the default clip_grad_norm_ silently multiplies all
            # model gradients by NaN if even one gradient is non-finite.
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                bad = [
                    name for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                ]
                raise FloatingPointError(
                    f"H16 nonfinite gradients before optimizer step: {bad[:16]}"
                ) from exc
            # Synchronize/log scalar losses before mutating parameters,
            # so an asynchronous device error cannot corrupt a checkpoint.
            for key, value in (
                ("loss", loss), ("policy", policy_loss),
                ("act", act_loss), ("hp", hp_loss),
            ):
                sums[key] += float(value.detach())
            optimizer.step()
            updates += 1
    model.eval()
    return {
        **{key: value / updates for key, value in sums.items()},
        "optimizer_steps": updates,
        "training_decisions": len(entries),
        "macro_decisions": phase_counts["macro"],
        "combat_decisions": phase_counts["combat"],
        "search_roots": search_roots,
        "horizon_lambda": current_lambda,
        "effective_horizon": (1.0 / (1.0 - current_lambda)
                              if current_lambda < 1.0 else None),
    }
