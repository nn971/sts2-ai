"""H24 opt-in H22-compatible joint model with a genuinely neutral Act-clear critic.

Policy consumes the original H21 public context. The optional legacy future-HP
head is disabled in the Act-only experiment.
The Act-clear head consumes a *separately encoded* resource-neutral H21 state
and certified nonnegative resource coordinates. Relic monotonicity is OFF.
Inference is more expensive than H22 (two backbone forwards); monitor TPS.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from sts2_ai.emulator.protocol import LegalAction
from sts2_ai.models.shared_game_objects import EncodedDecision
from sts2_ai.models.shared_macro_transformer_v2 import (
    H21GroundedMacroTransformer,
    collate_h21,
    collate_h21_packed,
)
from sts2_ai.models.shared_transformer import H7Config, H20Batch
from sts2_ai.training.h16_joint_act_pilot import (
    H22_SCHEMA,
    JointActPilotNet,
    encode_joint_decision,
)
from sts2_ai.training.h19_resource_codec import EMULATOR_PUBLIC_SCHEMA_V1
from sts2_ai.training.h21_macro_offer_codec import H21_CODEC, H21_OBJECT_SCHEMA
from sts2_ai.training.h24_applicability import ResourceConstraintSelection
from sts2_ai.training.h24_neutral_value import (
    H24_NEUTRAL_ENCODER,
    H24NeutralValue,
    encode_h24_neutral_value,
)
from sts2_ai.training.h24_resource_monotone import (
    SafeResourceVocabulary,
    build_h24_monotone_head,
)

H24_SCHEMA = "sts2-h24-h21-hard-resource-monotone-act-v1"


@dataclass(frozen=True)
class H24EncodedDecision(EncodedDecision):
    neutral_value: H24NeutralValue | None = None


@dataclass(frozen=True)
class H24Batch:
    policy: H20Batch
    neutral_value: H20Batch
    hp: Tensor | None
    gold: Tensor | None
    potions: Tensor | None
    relics: Tensor | None

    def to(self, device: str | torch.device) -> H24Batch:
        return H24Batch(
            self.policy.to(device), self.neutral_value.to(device),
            *(x.to(device) if x is not None else None
              for x in (self.hp, self.gold, self.potions, self.relics)),
        )

    def __getattr__(self, name: str) -> Any:
        # Retain the H22 minibatch/search/actor metrics interface.
        return getattr(self.policy, name)


def encode_h24_joint_decision(
    state: dict[str, Any], actions: Sequence[LegalAction], *,
    selection: ResourceConstraintSelection | None = None,
    vocabulary: SafeResourceVocabulary | None = None,
) -> H24EncodedDecision:
    """Worker-safe H24 RPC encoding without allocating a neural network."""
    normal = encode_joint_decision(state, actions, representation="h21")
    neutral = encode_h24_neutral_value(
        state, selection=selection, vocabulary=vocabulary,
    )
    return H24EncodedDecision(
        normal.schema, normal.phase, normal.objects, normal.actions,
        normal.relations, neutral,
    )


class H24JointActPilotNet(JointActPilotNet):
    """Checkpoint-isolated alternative to H22's unconstrained Act-clear head."""

    h24_enabled = True

    def __init__(
        self, backbone: H21GroundedMacroTransformer, *,
        selection: ResourceConstraintSelection | None = None,
        vocabulary: SafeResourceVocabulary | None = None,
        search_act_mix: float = 0.0,
        act_only: bool = False,
    ) -> None:
        if not isinstance(backbone, H21GroundedMacroTransformer):
            raise TypeError("H24 requires H21 grounded backbone")
        if not 0 <= search_act_mix <= 1:
            raise ValueError("Invalid H24 Act-clear search blend")
        super().__init__(backbone)
        self.selection = selection if selection is not None else ResourceConstraintSelection()
        self.vocabulary = vocabulary if vocabulary is not None else SafeResourceVocabulary()
        if act_only and search_act_mix != 1.0:
            raise ValueError("Act-only H24 must use the monotone Act critic for PUCT")
        self.search_act_mix = search_act_mix
        self.h24_act_only = act_only
        if act_only:
            for parameter in self.horizon_head.parameters():
                parameter.requires_grad_(False)
        self.monotone_act = build_h24_monotone_head(
            backbone.config.width, selection=self.selection,
            vocabulary=self.vocabulary,
        )

    @classmethod
    def warmstart_h22(
        cls, source: JointActPilotNet, *,
        selection: ResourceConstraintSelection | None = None,
        vocabulary: SafeResourceVocabulary | None = None,
        search_act_mix: float = 0.0,
        act_only: bool = False,
    ) -> H24JointActPilotNet:
        if (source.representation != "h21"
                or isinstance(source, H24JointActPilotNet)):
            raise ValueError("H24 warmstart requires an ordinary H22 model")
        model = cls(source.backbone, selection=selection, vocabulary=vocabulary,
                    search_act_mix=search_act_mix, act_only=act_only)
        model.horizon_head.load_state_dict(source.horizon_head.state_dict())
        model.act_clear.load_state_dict(source.act_clear.state_dict())
        # Reuse the H22 Act-clear linear head as the initial neutral base.
        # The change in input distribution still requires calibration.
        model.monotone_act.base.load_state_dict(source.act_clear.state_dict())
        return model

    def encode_decision(
        self, state: dict[str, Any], actions: Sequence[LegalAction],
    ) -> H24EncodedDecision:
        return encode_h24_joint_decision(
            state, actions, selection=self.selection,
            vocabulary=self.vocabulary,
        )

    def collate(
        self, decisions: list[H24EncodedDecision], *,
        packed: bool = False, **padding: Any,
    ) -> H24Batch:
        if (not decisions or any(not isinstance(d, H24EncodedDecision)
                                 or d.neutral_value is None for d in decisions)):
            raise ValueError("H24 batches require neutral-encoded decisions")
        fn = collate_h21_packed if packed else collate_h21
        policy = fn(decisions, **padding)
        neutral_values = [d.neutral_value for d in decisions]
        neutral = fn([value.encoded for value in neutral_values])

        def coords(field: str, count: int) -> Tensor | None:
            if count == 0:
                return None
            rows: list[list[float]] = []
            for value in neutral_values:
                original = getattr(value, field)
                if count == 1 and field in ("hp", "gold"):
                    if original is None:
                        raise ValueError("Missing H24 scalar resource")
                    rows.append([original])
                else:
                    if len(original) != count:
                        raise ValueError("H24 resource vocabulary dimension changed")
                    rows.append(list(original))
            return torch.tensor(rows, dtype=torch.float32)

        return H24Batch(
            policy, neutral,
            coords("hp", 1) if self.selection.hp else None,
            coords("gold", 1) if self.selection.gold else None,
            coords("potions", len(self.vocabulary.potion_ids)) if self.selection.potion else None,
            coords("relics", len(self.vocabulary.relic_ids)) if self.selection.relic else None,
        )

    def forward(
        self, batch: H24Batch, *, value_logits: bool = False,
        horizon_lambda: float | Tensor = 0.0,
        auxiliary_lambda: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(batch, H24Batch):
            raise TypeError("H24 requires paired policy/neutral-value batches")
        logits, _, policy_context = self.backbone(batch.policy, return_context=True)
        _, _, neutral_context = self.backbone(batch.neutral_value, return_context=True)
        act = self.monotone_act(
            neutral_context, hp=batch.hp, gold=batch.gold,
            potions=batch.potions, relics=batch.relics,
        )
        if self.h24_act_only:
            if auxiliary_lambda is not None:
                raise ValueError("Act-only H24 has no future-HP auxiliary target")
            if value_logits:
                return logits, act, act
            act_probability = torch.sigmoid(act)
            return logits, act_probability, act_probability
        hp = self._horizon_logits(policy_context, horizon_lambda)
        if auxiliary_lambda is not None:
            aux = self._horizon_logits(policy_context, auxiliary_lambda)
            if value_logits:
                return logits, act, hp, aux
            hp_prob = torch.sigmoid(hp)
            if self.search_act_mix:
                hp_prob = (1 - self.search_act_mix) * hp_prob + (
                    self.search_act_mix * torch.sigmoid(act))
            return (logits, torch.sigmoid(act), hp_prob, torch.sigmoid(aux))
        if value_logits:
            return logits, act, hp
        hp_prob = torch.sigmoid(hp)
        if self.search_act_mix:
            hp_prob = (1 - self.search_act_mix) * hp_prob + (
                self.search_act_mix * torch.sigmoid(act))
        return logits, torch.sigmoid(act), hp_prob

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": H24_SCHEMA,
            "parent_schema": H22_SCHEMA,
            "codec": H21_CODEC,
            "object_schema": H21_OBJECT_SCHEMA,
            "public_schema": EMULATOR_PUBLIC_SCHEMA_V1,
            "neutral_encoder": H24_NEUTRAL_ENCODER,
            "config": asdict(self.backbone.config),
            "selection": asdict(self.selection),
            "vocabulary": asdict(self.vocabulary),
            "search_act_mix": self.search_act_mix,
            "act_only": self.h24_act_only,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def from_checkpoint_data(
        cls, checkpoint: dict[str, Any], *, device: str = "cpu",
    ) -> H24JointActPilotNet:
        if (
            checkpoint.get("schema") != H24_SCHEMA
            or checkpoint.get("parent_schema") != H22_SCHEMA
            or checkpoint.get("codec") != H21_CODEC
            or checkpoint.get("object_schema") != H21_OBJECT_SCHEMA
            or checkpoint.get("public_schema") != EMULATOR_PUBLIC_SCHEMA_V1
            or checkpoint.get("neutral_encoder") != H24_NEUTRAL_ENCODER
        ):
            raise ValueError("H24 checkpoint and value encoder schema mismatch")
        model = cls(
            H21GroundedMacroTransformer(H7Config(**checkpoint["config"])),
            selection=ResourceConstraintSelection(**checkpoint["selection"]),
            vocabulary=SafeResourceVocabulary(**checkpoint["vocabulary"]),
            search_act_mix=checkpoint["search_act_mix"],
            act_only=checkpoint.get("act_only", False),
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model.to(device)
