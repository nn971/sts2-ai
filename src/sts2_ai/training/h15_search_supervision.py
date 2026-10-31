"""H15 policy/value distillation from certified public-history combat search.

The actor learns root visit distributions. The critic learns *realized* H9
encounter total-non-minion HP return / 2, not the potentially biased search
bootstrap. Search results from oracle-exact state forks are forbidden.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.h15_combat_codec import (
    H15_COMBAT_SCHEMA,
    collate_h15,
    encode_h15_combat,
)
from sts2_ai.models.shared_game_objects import EncodedDecision
from sts2_ai.search.combat_puct import (
    FAIR_COMBAT_CAPABILITY,
    H15_DEEP_SEARCH_VERSION,
    H15_SEARCH_VERSION,
    ROOT_CONDITIONED_COMBAT_CAPABILITY,
    CombatFrame,
    CombatSearchResult,
)

SAMPLE_SCHEMA = "sts2-h15-combat-search-target-v1"
VALUE_SAMPLE_SCHEMA = "sts2-h15-combat-realized-value-target-v1"
REWARD_VERSION = "encounter_nonminion_total_hp_v1"
DEFEAT_MARGIN_REWARD_VERSION = "encounter_nonminion_total_hp_defeat_margin_beta050_v1"
SUPPORTED_REWARD_VERSIONS = (REWARD_VERSION, DEFEAT_MARGIN_REWARD_VERSION)


@dataclass(frozen=True)
class SearchTrainingSample:
    decision: EncodedDecision
    visits: tuple[int, ...]
    realized_reward: float
    search_model_sha256: str
    source_seed: str

    def __post_init__(self) -> None:
        if (self.decision.schema != H15_COMBAT_SCHEMA or
                self.decision.phase != "combat" or
                len(self.decision.actions) != len(self.visits) or
                not self.visits or any(type(n) is not int or n < 0 for n in self.visits)
                or sum(self.visits) <= 0 or
                not math.isfinite(self.realized_reward) or
                not 0 <= self.realized_reward <= 2 or
                len(self.search_model_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.search_model_sha256)
                or not self.source_seed):
            raise ValueError("Invalid H15 search training sample")




@dataclass(frozen=True)
class RealizedValueSample:
    """A public on-policy late decision labelled by its real terminal return.

    This carries *no* search-policy target, visit count or fake PUCT trace.
    """
    decision: EncodedDecision
    realized_reward: float
    search_model_sha256: str
    source_seed: str

    def __post_init__(self) -> None:
        if (self.decision.schema != H15_COMBAT_SCHEMA
                or self.decision.phase != "combat"
                or not self.decision.actions
                or not math.isfinite(self.realized_reward)
                or not 0 <= self.realized_reward <= 2
                or len(self.search_model_sha256) != 64
                or any(c not in "0123456789abcdef"
                       for c in self.search_model_sha256)
                or not self.source_seed):
            raise ValueError("Invalid H15 realized-value sample")


def value_target_record(
    observation: Any, legal_actions: tuple[LegalAction, ...], *,
    terminal_reward: float, search_model_sha256: str,
    source_seed: str, emulator_revision: str, decision_index: int,
    reward_version: str = REWARD_VERSION,
) -> dict[str, Any]:
    """Build value-only supervision from an actually played public decision."""
    if (reward_version not in SUPPORTED_REWARD_VERSIONS
            or not legal_actions or observation.policy_id != FAIR_POLICY_ID
            or not emulator_revision or type(decision_index) is not int
            or decision_index < 0):
        raise ValueError("H15 value sample needs a fair, live public decision")
    row = {
        "schema": VALUE_SAMPLE_SCHEMA,
        "sample_kind": "realized_value",
        "combat_reward_version": reward_version,
        "chance_capability": FAIR_COMBAT_CAPABILITY,
        "search_version": H15_SEARCH_VERSION,
        "search_model_sha256": search_model_sha256,
        "source_seed": source_seed,
        "emulator_revision": emulator_revision,
        "public_state": json.loads(observation.payload_json),
        "legal_actions": [
            {"action_id": a.action_id, "kind": a.kind,
             "payload_json": a.payload_json}
            for a in legal_actions
        ],
        "realized_reward": terminal_reward,
        "source_decision_index": decision_index,
    }
    decode_target(row)
    return row


def target_record(
    root: CombatFrame, result: CombatSearchResult, *,
    terminal_reward: float, search_model_sha256: str,
    source_seed: str, emulator_revision: str,
    reward_version: str = REWARD_VERSION,
) -> dict[str, Any]:
    """Serialize a completed ACTUAL trajectory's return, not a search estimate."""
    if (result.search_version not in (
                H15_SEARCH_VERSION, H15_DEEP_SEARCH_VERSION,
            ) or reward_version not in SUPPORTED_REWARD_VERSIONS
            or (result.search_version == H15_DEEP_SEARCH_VERSION) !=
               (result.tree_expansion_mode == "full_depth")
            or result.chance_capability not in (
                FAIR_COMBAT_CAPABILITY, ROOT_CONDITIONED_COMBAT_CAPABILITY,
            )
            or result.information_key != root.information_key
            or result.action_ids != tuple(a.action_id for a in root.legal_actions)
            or not emulator_revision):
        raise ValueError("Search result is not aligned with public combat root")
    row = {
        "schema": SAMPLE_SCHEMA,
        "combat_reward_version": reward_version,
        "chance_capability": result.chance_capability,
        "search_version": result.search_version,
        "search_tree_expansion_mode": result.tree_expansion_mode,
        "search_maximum_expanded_depth": result.maximum_expanded_depth,
        "search_model_sha256": search_model_sha256,
        "source_seed": source_seed,
        "emulator_revision": emulator_revision,
        "public_state": json.loads(root.observation.payload_json),
        "legal_actions": [
            {"action_id": a.action_id, "kind": a.kind, "payload_json": a.payload_json}
            for a in root.legal_actions
        ],
        "visits": list(result.visits),
        "realized_reward": terminal_reward,
        "search_simulations": result.simulations,
        "search_rollout_mode": result.rollout_mode,
        "search_terminal_evaluations": result.terminal_evaluations,
        "search_neural_cutoffs": result.neural_cutoffs,
    }
    decode_target(row)  # validate our own format before accepting the sample
    return row


def decode_target(row: dict[str, Any]) -> SearchTrainingSample | RealizedValueSample:
    if (row.get("schema") not in (SAMPLE_SCHEMA, VALUE_SAMPLE_SCHEMA)
            or row.get("combat_reward_version") not in SUPPORTED_REWARD_VERSIONS
            or row.get("chance_capability") not in (
                FAIR_COMBAT_CAPABILITY, ROOT_CONDITIONED_COMBAT_CAPABILITY,
            )
            or row.get("search_version") not in (
                H15_SEARCH_VERSION, H15_DEEP_SEARCH_VERSION,
            )
            or not isinstance(row.get("emulator_revision"), str)
            or not row["emulator_revision"]):
        raise ValueError("Incompatible or uncertified H15 search target")
    state = row.get("public_state")
    raw_actions = row.get("legal_actions")
    visits = row.get("visits")
    if (not isinstance(state, dict) or
            not isinstance(raw_actions, list) or
            (row["schema"] == SAMPLE_SCHEMA and not isinstance(visits, list))):
        raise ValueError("H15 target lacks public state, menu, or visits")
    try:
        actions = tuple(
            LegalAction(a["action_id"], a["kind"], a["payload_json"])
            for a in raw_actions
        )
    except (TypeError, KeyError) as exc:
        raise ValueError("Malformed public legal actions") from exc
    if (any(type(a.action_id) is not str or not a.action_id or
            type(a.kind) is not str or not a.kind or
            type(a.payload_json) is not str for a in actions)
            or len({a.action_id for a in actions}) != len(actions)):
        raise ValueError("Invalid public legal menu")
    if row["schema"] == VALUE_SAMPLE_SCHEMA:
        if (row["chance_capability"] != FAIR_COMBAT_CAPABILITY
                or row.get("sample_kind") != "realized_value"
                or type(row.get("source_decision_index")) is not int
                or row["source_decision_index"] < 0
                or any(k in row for k in (
                    "visits", "search_simulations", "search_rollout_mode",
                    "search_terminal_evaluations", "search_neural_cutoffs",
                ))):
            raise ValueError("Value-only target must not claim PUCT searches")
        return RealizedValueSample(
            decision=encode_h15_combat(state, actions),
            realized_reward=row.get("realized_reward"),
            search_model_sha256=row.get("search_model_sha256"),
            source_seed=row.get("source_seed"),
        )
    if row["search_version"] == H15_DEEP_SEARCH_VERSION:
        if (row.get("search_tree_expansion_mode") != "full_depth"
                or type(row.get("search_maximum_expanded_depth")) is not int
                or row["search_maximum_expanded_depth"] < 0):
            raise ValueError("M17 deep PUCT target lacks expansion provenance")
    elif row.get("search_tree_expansion_mode", "one_leaf") != "one_leaf":
        raise ValueError("H15 v2 target cannot claim deep expansion")
    sample = SearchTrainingSample(
        decision=encode_h15_combat(state, actions),
        visits=tuple(visits),
        realized_reward=row.get("realized_reward"),
        search_model_sha256=row.get("search_model_sha256"),
        source_seed=row.get("source_seed"),
    )
    if (row.get("search_rollout_mode") not in ("value", "terminal")
            or type(row.get("search_terminal_evaluations")) is not int
            or type(row.get("search_neural_cutoffs")) is not int):
        raise ValueError("Missing H15 search evaluation provenance")
    if (type(row.get("search_simulations")) is not int
            or row["search_simulations"] <= 0
            or sum(sample.visits) != row["search_simulations"]):
        raise ValueError("Root search visit count is inconsistent")
    if (row["search_rollout_mode"] == "terminal" and
            (row["search_neural_cutoffs"] != 0 or
             row["search_terminal_evaluations"] != row["search_simulations"])):
        raise ValueError("Terminal-rollout search cannot contain critic cutoffs")
    return sample


def search_policy_value_loss(
    model: Any,
    samples: Sequence[SearchTrainingSample | RealizedValueSample], *,
    value_weight: float = 1.0,
) -> tuple[Any, dict[str, float]]:
    """Train policy only on *real* PUCT visits; critic on all realized returns."""
    import torch
    from torch.nn import functional as F

    if not samples or not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("Invalid H15 distillation batch")
    batch = collate_h15([s.decision for s in samples]).to(
        next(model.parameters()).device,
    )
    logits, values = model(batch, known_phase=0)
    policy_indices = [
        i for i, sample in enumerate(samples)
        if isinstance(sample, SearchTrainingSample)
    ]
    # Masked legal-action logits may contain -inf for padding; use the
    # finite value head for the differentiable zero on critic-only batches.
    policy_loss = values.sum() * 0.0
    policy_entropy = logits.new_zeros(())
    policy_kl = logits.new_zeros(())
    if policy_indices:
        selected_logits = logits[policy_indices]
        selected_mask = batch.action_mask[policy_indices]
        counts = torch.zeros_like(selected_logits)
        for row, i in enumerate(policy_indices):
            visit_counts = samples[i].visits
            counts[row, :len(visit_counts)] = torch.tensor(
                visit_counts, dtype=logits.dtype, device=logits.device,
            )
        targets = counts / counts.sum(dim=1, keepdim=True)
        logp = F.log_softmax(selected_logits, dim=1).masked_fill(
            ~selected_mask, 0.0,
        )
        policy_loss = -(targets * logp).sum(dim=1).mean()
        log_targets = torch.where(
            targets > 0, torch.log(targets.clamp_min(1e-30)),
            torch.zeros_like(targets),
        )
        policy_entropy = -(targets * log_targets).sum(dim=1).mean()
        policy_kl = (targets * (log_targets - logp)).sum(dim=1).mean()
    real_returns = torch.tensor(
        [sample.realized_reward / 2.0 for sample in samples],
        device=values.device, dtype=values.dtype,
    )
    value_loss = F.mse_loss(values, real_returns)
    loss = policy_loss + value_weight * value_loss
    return loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "policy_target_entropy": float(policy_entropy.detach().cpu()),
        "policy_kl_to_search": float(policy_kl.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "policy_samples": float(len(policy_indices)),
        "value_only_samples": float(len(samples) - len(policy_indices)),
    }
