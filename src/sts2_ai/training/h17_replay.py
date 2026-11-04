"""Bounded, reproducible H17 supervised replay from completed native trajectories.

Replay intentionally excludes REINFORCE on stale behavior actions. It trains
value/Act heads and optionally distills *observed PUCT visit distributions*
only. Every target is recomputed for the current horizon lambda.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_transformer import collate_h7
from sts2_ai.training.h16_episode_metrics import combat_results
from sts2_ai.training.h16_joint_act_pilot import (
    JointActPilotNet,
    PilotEpisode,
    combat_shaped_hp_return,
    completed_room_hp,
    remaining_room_hp,
)
from sts2_ai.training.h16_public_codec import encode_h16_decision
from sts2_ai.training.h17_adaptive import playable_card_types

SCHEMA = "sts2-h17-supervised-replay-v1"


def _weighted_sample(
    items: Sequence[dict[str, Any]], k: int,
    rng: random.Random, exposure: Mapping[str, int],
) -> list[dict[str, Any]]:
    if k <= 0 or not items:
        return []
    def weight(item: dict[str, Any]) -> float:
        cards = item["card_types"]
        rarity = max(
            (1 / math.sqrt(1 + exposure.get(card, 0)) for card in cards),
            default=0.15,
        )
        return (
            0.25 + 1.4 * rarity + 0.4 * min(item["floor"], 16) / 16
            + 0.7 * bool(item["search_policy"] is not None)
        )
    weighted = [
        (-math.log(max(rng.random(), 1e-15)) / weight(item), item)
        for item in items
    ]
    return [item for _, item in sorted(weighted, key=lambda pair: pair[0])[:k]]


def collect_replay(
    episodes: Sequence[PilotEpisode], *, source_round: int,
    exposure: Mapping[str, int], max_macro_per_run: int = 8,
    max_combat_per_run: int = 16,
    loss_progress_cap: float = 0.2,
    hp_reference: int | None = None,
) -> list[dict[str, Any]]:
    """Select varied, rare-card decision states with genuine completed labels."""
    if (source_round < 0 or min(max_macro_per_run, max_combat_per_run) < 0
            or (hp_reference is not None and (
                type(hp_reference) is not int or hp_reference <= 0
            ))):
        raise ValueError("Invalid H17 replay collection configuration")
    result: list[dict[str, Any]] = []
    rng = random.Random(1717 + source_round)
    for episode in episodes:
        if not episode.complete:
            continue
        room_hp = completed_room_hp(
            episode.frames, hp_reference=hp_reference,
        )
        act = float(episode.outcome == "act_clear")
        quality: dict[int, float] = {}
        for fight in combat_results(episode.frames, episode.outcome):
            for i in range(fight.start, fight.end):
                quality[i] = fight.quality(loss_progress_cap)
        phases: dict[str, list[dict[str, Any]]] = {"macro": [], "combat": []}
        for i, decision in enumerate(episode.decisions):
            state = episode.frames[i]
            phase = "combat" if isinstance(state.get("combat"), dict) else "macro"
            combat_quality = quality.get(i) if phase == "combat" else None
            if phase == "combat" and combat_quality is None:
                continue
            actions = decision.legal_actions
            record = {
                "seed": episode.seed, "source_round": source_round,
                "decision_index": i, "phase": phase,
                "floor": state.get("floor", 0),
                "observation": decision.observation.payload_json,
                "actions": [
                    [a.action_id, a.kind, a.payload_json] for a in actions
                ],
                "card_types": sorted(playable_card_types(state, actions)),
                "search_policy": (
                    list(decision.search_policy)
                    if decision.search_policy is not None else None
                ),
                "room_hp": list(remaining_room_hp(state, room_hp)),
                "act": act,
                "combat_quality": combat_quality,
            }
            phases[phase].append(record)
        result.extend(_weighted_sample(
            phases["macro"], max_macro_per_run, rng, exposure,
        ))
        result.extend(_weighted_sample(
            phases["combat"], max_combat_per_run, rng, exposure,
        ))
    return result


def merge_replay(
    previous: Sequence[dict[str, Any]], incoming: Sequence[dict[str, Any]],
    *, round_index: int, exposure: Mapping[str, int], capacity: int = 8192,
    max_age: int = 32,
) -> list[dict[str, Any]]:
    if round_index < 0 or capacity <= 0 or max_age < 1:
        raise ValueError("Invalid H17 replay retention settings")
    candidates = [
        row for row in [*previous, *incoming]
        if round_index - max_age < row["source_round"] <= round_index
    ]
    if len(candidates) <= capacity:
        return candidates
    rng = random.Random(52000 + round_index)
    return _weighted_sample(candidates, capacity, rng, exposure)


def sample_replay(
    rows: Sequence[dict[str, Any]], *, round_index: int,
    exposure: Mapping[str, int], samples: int = 256,
) -> list[dict[str, Any]]:
    if round_index < 0 or samples < 0:
        raise ValueError("Invalid replay training sampling configuration")
    rng = random.Random(33017 + round_index)
    fresh = [r for r in rows if r["source_round"] < round_index]
    macros = _weighted_sample(
        [r for r in fresh if r["phase"] == "macro"],
        samples // 2, rng, exposure,
    )
    combats = _weighted_sample(
        [r for r in fresh if r["phase"] == "combat"],
        samples - len(macros), rng, exposure,
    )
    wanted = samples - len(macros) - len(combats)
    seen = {(r["seed"], r["decision_index"]) for r in [*macros, *combats]}
    rest = _weighted_sample(
        [r for r in fresh if (r["seed"], r["decision_index"]) not in seen],
        wanted, rng, exposure,
    )
    chosen = [*macros, *combats, *rest]
    rng.shuffle(chosen)
    return chosen


def fit_replay(
    model: JointActPilotNet, optimizer: torch.optim.Optimizer,
    rows: Sequence[dict[str, Any]], *, horizon_lambda: float,
    replay_weight: float = 0.2, minibatch: int = 32,
    combat_credit_weight: float = 0.25,
) -> dict[str, float | int]:
    """Supervised heads only; no stale-policy REINFORCE or behavior logging."""
    if (minibatch <= 0 or not 0 <= horizon_lambda <= 1
            or not 0 <= replay_weight <= 1
            or not 0 <= combat_credit_weight <= 1):
        raise ValueError("Invalid H17 replay optimizer configuration")
    if not rows or replay_weight == 0:
        return {"examples": 0, "macro_examples": 0, "search_targets": 0,
                "steps": 0, "loss": 0.0}
    device = next(model.parameters()).device
    model.train()
    loss_sum = 0.0
    steps = 0
    for start in range(0, len(rows), minibatch):
        chunk = rows[start:start + minibatch]
        legal = [
            tuple(LegalAction(*item) for item in row["actions"]) for row in chunk
        ]
        encoded = [
            model.encode_decision(json.loads(row["observation"]), actions)
            for row, actions in zip(chunk, legal, strict=True)
        ]
        batch = model.collate(encoded).to(device)
        logits, act_logit, hp_logit = model(
            batch, value_logits=True, horizon_lambda=horizon_lambda,
        )
        hp_labels = torch.tensor([
            combat_shaped_hp_return(
                row["room_hp"], row["act"], horizon_lambda,
                combat_quality=row["combat_quality"],
                combat_credit_weight=combat_credit_weight,
            ) for row in chunk
        ], dtype=hp_logit.dtype, device=device)
        act_labels = torch.tensor(
            [row["act"] for row in chunk], dtype=act_logit.dtype, device=device,
        )
        hp_loss = F.mse_loss(torch.sigmoid(hp_logit), hp_labels)
        act_loss = F.binary_cross_entropy_with_logits(act_logit, act_labels)
        ce = []
        for i, row in enumerate(chunk):
            target = row["search_policy"]
            if target is None:
                continue
            if len(target) != len(legal[i]):
                raise ValueError("Replay PUCT policy/action menu mismatch")
            probs = torch.tensor(
                target, dtype=logits.dtype, device=device,
            )
            ce.append(
                -(probs * F.log_softmax(logits[i, :len(target)], dim=0)).sum()
            )
        ce_loss = torch.stack(ce).mean() if ce else logits.sum() * 0
        loss = replay_weight * (hp_loss + 0.25 * act_loss + 0.5 * ce_loss)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite H17 replay loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True,
        )
        optimizer.step()
        loss_sum += float(loss.detach())
        steps += 1
    model.eval()
    return {
        "examples": len(rows),
        "macro_examples": sum(row["phase"] == "macro" for row in rows),
        "search_targets": sum(row["search_policy"] is not None for row in rows),
        "steps": steps, "loss": loss_sum / steps,
    }


def load_replay(path: Path, *, fingerprint: str,
                expected_round: int) -> list[dict[str, Any]]:
    if not path.is_file():
        if expected_round == 0:
            return []
        raise ValueError("Missing H17 replay state on resume")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    if (document.get("schema") != SCHEMA
            or document.get("fingerprint") != fingerprint
            or document.get("round") != expected_round):
        raise ValueError("H17 replay state identity/round mismatch")
    return document["rows"]


def write_replay(
    path: Path, *, fingerprint: str, round_index: int,
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": SCHEMA, "fingerprint": fingerprint,
        "round": round_index, "rows": list(rows),
    }
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, delete=False,
        prefix="." + path.name + "-",
    ) as raw:
        name = Path(raw.name)
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3) as stream:
            stream.write(json.dumps(
                data, separators=(",", ":"), sort_keys=True,
            ).encode("utf-8"))
    os.replace(name, path)
