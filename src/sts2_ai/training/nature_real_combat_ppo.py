"""On-policy v15 residual PPO samples from *real* paired semantic combats.

Reuse the production v15 pooled/relational PPO optimizer without changing
its architecture. This conversion rejects stale policies, censored combats,
greedy/off-policy behavior, forged action menus and divergent q/logits.
Nature-only events are never treated as player decisions.

The training target is precisely v15's within-combat HP-preservation
objective, requiring public entry HP/max HP and genuine terminal exit HP.
The older P0 combat score is independently checked for provenance only.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from sts2_ai.emulator.protocol import LegalAction, Observation
from sts2_ai.emulator.semantic_p0_frozen_player import (
    FrozenP0Player,
    FrozenPolicyInferenceError,
)
from sts2_ai.models.neural import NeuralPolicyValueModel
from sts2_ai.models.pooled_objects import PooledObjectTorch
from sts2_ai.models.relational_objects import (
    RelationalObjectTorch,
    _action_tokens,
    prepare_relational_topology,
)
from sts2_ai.training.phase_split_ppo import gae_terminal
from sts2_ai.training.pooled_ppo_v15 import (
    PooledPpoSample,
    _prepared_decision,
)
from sts2_ai.training.selfplay import PublicDecision
from sts2_ai.nature.protocol import canonical_sha256

P0_PPO_SCHEMA = "sts2-ai-p0-source-combat-player-ppo-v1"


class RealPlayerPpoError(ValueError):
    """Behavior/action provenance does not permit an on-policy PPO update."""


def validate_recorded_action(
    raw: Mapping[str, Any], *,
    expected_model_id: str, expected_temperature: float,
    expected_scope: str,
) -> tuple[PublicDecision, float, float]:
    if raw.get("behavior_model_id") != expected_model_id:
        raise RealPlayerPpoError("Stale player behavior model identity")
    if (raw.get("action_scope") != expected_scope
            or expected_scope not in ("silent-basic-v1", "all-legal-v1")):
        raise RealPlayerPpoError("Behavior/legal-action scope mismatch")
    if (not isinstance(raw.get("temperature"), (int, float))
            or not math.isclose(raw["temperature"], expected_temperature, abs_tol=1e-12)):
        raise RealPlayerPpoError("Behavior temperature mismatch")
    obs = raw.get("observation")
    acts = raw.get("legal_actions")
    if not isinstance(obs, dict) or not isinstance(acts, list) or not acts:
        raise RealPlayerPpoError("Missing public observation or legal action menu")
    if obs.get("policy_id") != "prototype-fair-v0":
        raise RealPlayerPpoError("Player action uses a nonfair observation policy")
    try:
        payload = json.loads(obs["payload_json"])
        if not isinstance(payload, dict) or "semanticDeck" in payload or "pending" in payload:
            raise RealPlayerPpoError("Hidden/private state in player PPO observation")
        if obs.get("observation_hash") != canonical_sha256(payload):
            raise RealPlayerPpoError("Player observation hash corrupted")
        observation = Observation(
            obs["policy_id"], obs["payload_json"], obs["observation_hash"]
        )
        actions = tuple(
            LegalAction(a["action_id"], a["kind"], a["payload_json"])
            for a in acts
        )
        for action in actions:
            if not isinstance(json.loads(action.payload_json), dict):
                raise RealPlayerPpoError("Bad public legal action payload")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RealPlayerPpoError("Malformed public PPO action") from exc
    if len({a.action_id for a in actions}) != len(actions):
        raise RealPlayerPpoError("Duplicate public action IDs")
    chosen = raw.get("chosen_index")
    if type(chosen) is not int or not 0 <= chosen < len(actions):
        raise RealPlayerPpoError("Chosen action index not present in public menu")
    if raw.get("selected_id") != actions[chosen].action_id:
        raise RealPlayerPpoError("Chosen source-issued action identity drift")
    if (expected_scope == "silent-basic-v1"
            and any(a.kind not in ("end_turn", "play_card") for a in actions)):
        raise RealPlayerPpoError("Unmasked unsupported player action in P0 PPO")

    # The exact source-issued *unrestricted* public menu is part of the
    # emulator's replay support_identity. Re-derive both the public projection
    # and the restricted behavior menu, including unselected actions.
    source_observation = raw.get("source_observation")
    source_menu = raw.get("source_legal_actions")
    if not isinstance(source_observation, dict) or not isinstance(source_menu, list):
        raise RealPlayerPpoError("Missing complete source player menu provenance")
    try:
        source_identity = canonical_sha256([source_observation, source_menu])
        source_public, all_source_actions = FrozenP0Player._public_types(
            source_observation, source_menu
        )
        scoped = FrozenP0Player.scoped_actions(
            source_observation, all_source_actions, expected_scope
        )
    except (FrozenPolicyInferenceError, ValueError, TypeError, KeyError) as exc:
        raise RealPlayerPpoError("Malformed source player menu provenance") from exc
    if raw.get("source_support_identity") != source_identity:
        raise RealPlayerPpoError("Source action-menu provenance digest mismatch")
    if source_public != observation or scoped != actions:
        raise RealPlayerPpoError(
            "Recorded PPO observation/legal menu differs from issued source menu"
        )

    lp, value = raw.get("behavior_logprob"), raw.get("behavior_value")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (lp, value)):
        raise RealPlayerPpoError("Missing/nonfinite behavior log-probability or value")
    if lp > 1e-10:
        raise RealPlayerPpoError("Behavior log probability exceeds zero")
    return PublicDecision(observation, actions, chosen, "combat"), float(lp), float(value)


def prepare_real_combat_ppo_samples(
    paired_records: Sequence[Mapping[str, Any]], *,
    arm: str, base: NeuralPolicyValueModel,
    head: PooledObjectTorch | RelationalObjectTorch,
    behavior_model_id: str, behavior_sha256: str,
    temperature: float, action_scope: str,
    gamma: float = 0.995, gae_lambda: float = 0.98,
    inference_atol: float = 2e-3,
) -> tuple[list[PooledPpoSample], list[float]]:
    """Convert complete real source-combat episodes into ordinary v15 PPO data.

    Must run BEFORE any optimizer mutation after exporting the model to the
    rollout processes. Stored old log-probability is independently checked
    against the current Torch head; unlike off-policy importance weighting,
    any mismatch rejects the entire cohort.
    """
    if arm not in ("nature", "ordinary"):
        raise ValueError("Unknown combat training arm")
    if not (0 < temperature and math.isfinite(temperature)):
        raise ValueError("PPO actor sampling temperature must be positive")
    if not behavior_model_id or not behavior_sha256:
        raise ValueError("Require pinned v15 behavior model and SHA-256")
    if not paired_records:
        raise RealPlayerPpoError("No terminal paired combats to train on")
    if not 0 < inference_atol <= 0.01:
        raise ValueError("Invalid portable/Torch parity tolerance")

    torch = head.torch
    samples: list[PooledPpoSample] = []
    episode_scores: list[float] = []
    with torch.no_grad():
        for record in paired_records:
            if (record.get("same_opening") is not True or
                    record.get("same_initial_public_state") is not True or
                    record.get("ppo_behavior_model_sha256") != behavior_sha256):
                raise RealPlayerPpoError("Unpaired opening or stale PPO model digest")
            arm_trace = record.get(arm)
            collected = record.get("player_ppo", {})
            if not isinstance(arm_trace, dict) or not isinstance(collected, dict):
                raise RealPlayerPpoError("Missing genuine combat arm data")
            rows = collected.get(arm)
            if (arm_trace.get("outcome") not in ("victory", "defeat")
                    or type(arm_trace.get("exit_hp")) is not int
                    or not isinstance(rows, list) or not rows):
                raise RealPlayerPpoError("Unresolved, censored or empty combat segment")
            player_steps = [
                decision for decision in arm_trace.get("decisions", [])
                if decision.get("phase") == "player"
            ]
            if len(player_steps) != len(rows):
                raise RealPlayerPpoError("Player PPO record count differs from transcript")
            # Verify the legacy P0 published outcome/exit-HP reward,
            # then use the production v15 hp_preservation critic objective.
            published_score = float(arm_trace["player_return"])
            if not math.isfinite(published_score) or not math.isclose(
                published_score, float(arm_trace["outcome"] == "victory")
                + max(0, arm_trace["exit_hp"]) / 100.0, abs_tol=1e-10,
            ):
                raise RealPlayerPpoError("Combat terminal reward provenance mismatch")
            first_public = json.loads(rows[0]["observation"]["payload_json"])
            entry_hp, max_hp = first_public.get("hp"), first_public.get("max_hp")
            if (type(entry_hp) is not int or type(max_hp) is not int
                    or max_hp <= 0 or type(arm_trace["exit_hp"]) is not int):
                raise RealPlayerPpoError(
                    "v15 HP-preservation target requires public entry/max HP"
                )
            score = (
                1.0 - 0.5 * min(
                    1.0, max(0.0, (entry_hp - arm_trace["exit_hp"]) / max_hp)
                )
                if arm_trace["outcome"] == "victory" else 0.0
            )
            cache: list[tuple[Any, ...]] = []
            values: list[float] = []
            for raw, step in zip(rows, player_steps, strict=True):
                decision, old_lp, old_value = validate_recorded_action(
                    raw, expected_model_id=behavior_model_id,
                    expected_temperature=temperature, expected_scope=action_scope,
                )
                if decision.legal_actions[decision.chosen_index].action_id != step["selected_id"]:
                    raise RealPlayerPpoError("PPO selected action mismatches exact replay")
                if (not isinstance(step.get("support_identity"), str)
                        or not step["support_identity"]
                        or raw["source_support_identity"] != step["support_identity"]):
                    raise RealPlayerPpoError(
                        "Source player action-menu diverges from verified replay"
                    )
                objects, refs, baseline = _prepared_decision(decision, base)
                bindings = (
                    _action_tokens(decision.legal_actions, objects)
                    if isinstance(head, RelationalObjectTorch) else None
                )
                v, logits = head.forward_prepared(
                    objects, bindings if bindings is not None else refs, baseline,
                )
                lp = torch.nn.functional.log_softmax(
                    logits / temperature, dim=0
                )[decision.chosen_index].item()
                value = v.item()
                if (not math.isfinite(lp) or not math.isfinite(value)
                        or abs(lp - old_lp) > inference_atol
                        or abs(value - old_value) > inference_atol):
                    raise RealPlayerPpoError(
                        "Stale PPO head or portable-v15/Torch inference mismatch"
                    )
                topology = (
                    prepare_relational_topology(objects, bindings)
                    if bindings is not None else None
                )
                features = (
                    head.embedding.prepare(objects)
                    if isinstance(head, RelationalObjectTorch) else None
                )
                cache.append((
                    objects, refs, bindings, baseline, decision, old_lp,
                    topology, features,
                ))
                values.append(old_value)
            advantages, targets = gae_terminal(
                values, score, gamma=gamma, lam=gae_lambda,
            )
            for cached, adv, target in zip(cache, advantages, targets, strict=True):
                (objects, refs, bindings, baseline, decision, old_lp,
                 topology, features) = cached
                samples.append(PooledPpoSample(
                    objects, refs, baseline, decision.chosen_index, old_lp,
                    adv, target, bindings, topology, features,
                    1.0,
                ))
            episode_scores.append(score)
    return samples, episode_scores
