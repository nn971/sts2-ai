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

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.h15_combat_codec import (
    H15_COMBAT_SCHEMA,
    collate_h15,
    encode_h15_combat,
)
from sts2_ai.models.shared_game_objects import EncodedDecision
from sts2_ai.search.combat_puct import (
    FAIR_COMBAT_CAPABILITY,
    H15_SEARCH_VERSION,
    CombatFrame,
    CombatSearchResult,
)

SAMPLE_SCHEMA = "sts2-h15-combat-search-target-v1"
REWARD_VERSION = "encounter_nonminion_total_hp_v1"


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


def target_record(
    root: CombatFrame, result: CombatSearchResult, *,
    terminal_reward: float, search_model_sha256: str,
    source_seed: str, emulator_revision: str,
) -> dict[str, Any]:
    """Serialize a completed ACTUAL trajectory's return, not a search estimate."""
    if (result.search_version != H15_SEARCH_VERSION
            or result.information_key != root.information_key
            or result.action_ids != tuple(a.action_id for a in root.legal_actions)
            or not emulator_revision):
        raise ValueError("Search result is not aligned with public combat root")
    row = {
        "schema": SAMPLE_SCHEMA,
        "combat_reward_version": REWARD_VERSION,
        "chance_capability": FAIR_COMBAT_CAPABILITY,
        "search_version": H15_SEARCH_VERSION,
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
        "search_terminal_evaluations": result.terminal_evaluations,
        "search_neural_cutoffs": result.neural_cutoffs,
    }
    decode_target(row)  # validate our own format before accepting the sample
    return row


def decode_target(row: dict[str, Any]) -> SearchTrainingSample:
    if (row.get("schema") != SAMPLE_SCHEMA
            or row.get("combat_reward_version") != REWARD_VERSION
            or row.get("chance_capability") != FAIR_COMBAT_CAPABILITY
            or row.get("search_version") != H15_SEARCH_VERSION
            or not isinstance(row.get("emulator_revision"), str)
            or not row["emulator_revision"]):
        raise ValueError("Incompatible or uncertified H15 search target")
    state = row.get("public_state")
    raw_actions = row.get("legal_actions")
    visits = row.get("visits")
    if (not isinstance(state, dict) or
            not isinstance(raw_actions, list) or
            not isinstance(visits, list)):
        raise ValueError("Search target lacks public state, menu, or visits")
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
    sample = SearchTrainingSample(
        decision=encode_h15_combat(state, actions),
        visits=tuple(visits),
        realized_reward=row.get("realized_reward"),
        search_model_sha256=row.get("search_model_sha256"),
        source_seed=row.get("source_seed"),
    )
    if (type(row.get("search_simulations")) is not int
            or row["search_simulations"] <= 0
            or sum(sample.visits) != row["search_simulations"]):
        raise ValueError("Root search visit count is inconsistent")
    return sample


def search_policy_value_loss(
    model: Any, samples: Sequence[SearchTrainingSample], *,
    value_weight: float = 1.0,
) -> tuple[Any, dict[str, float]]:
    """Legal-only search cross-entropy + supervised real HP-return value."""
    import torch
    from torch.nn import functional as F

    if not samples or not math.isfinite(value_weight) or value_weight < 0:
        raise ValueError("Invalid H15 distillation batch")
    batch = collate_h15([s.decision for s in samples]).to(
        next(model.parameters()).device,
    )
    logits, values = model(batch, known_phase=0)
    mask = batch.action_mask
    counts = torch.zeros_like(logits)
    for i, sample in enumerate(samples):
        counts[i, :len(sample.visits)] = torch.tensor(
            sample.visits, dtype=logits.dtype, device=logits.device,
        )
    targets = counts / counts.sum(dim=1, keepdim=True)
    logp = F.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
    policy_loss = -(targets * logp).sum(dim=1).mean()
    real_returns = torch.tensor(
        [sample.realized_reward / 2.0 for sample in samples],
        device=values.device, dtype=values.dtype,
    )
    value_loss = F.mse_loss(values, real_returns)
    loss = policy_loss + value_weight * value_loss
    return loss, {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
    }
