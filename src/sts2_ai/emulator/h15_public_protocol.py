"""H15 experimental public-policy frames and emulator-owned Nature receipts.

Shared by source-entry diagnostics and independent-candidate conditioning.
The emulator remains the sole source of mechanics, laws and random outcomes.
No fair-search capability is exported by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sts2_ai.emulator.jsonl_backend import FAIR_POLICY_ID
from sts2_ai.emulator.protocol import LegalAction, Observation

NATURE_KINDS = frozenset({
    "enemy_action", "enemy_intent", "draw_card", "random_target",
})
REFERENCE_PROVIDER = "reference-nature-splitmix64-v1"


def public_player_frame(
    frame: Mapping[str, Any],
) -> tuple[Observation, tuple[LegalAction, ...]]:
    """Reproduce the exact public observation and *ordered* action menu."""
    if frame.get("boundaryKind") != "player":
        raise ValueError("H15 tactical inference requires a player boundary")
    if frame.get("observationPolicyId") != FAIR_POLICY_ID:
        raise ValueError("H15 status omitted the fair public observation policy")
    payload = frame.get("observationPayloadJson")
    digest = frame.get("observationHash")
    menu = frame.get("legalActions")
    if not isinstance(payload, str) or not isinstance(digest, str):
        raise ValueError("H15 status omitted the public observation")
    if not isinstance(menu, list) or not menu:
        raise ValueError("H15 player boundary has no legal actions")
    observation = Observation(FAIR_POLICY_ID, payload, digest)
    actions: list[LegalAction] = []
    seen: set[str] = set()
    for raw in menu:
        if (not isinstance(raw, dict)
                or not isinstance(raw.get("actionId"), str)
                or raw["actionId"] in seen
                or not isinstance(raw.get("kind"), str)
                or not isinstance(raw.get("payloadJson"), str)):
            raise ValueError("Invalid H15 public action menu")
        seen.add(raw["actionId"])
        actions.append(LegalAction(
            raw["actionId"], raw["kind"], raw["payloadJson"],
        ))
    return observation, tuple(actions)


def choose_reference_at_pending(
    backend: Any, handle: str, frame: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Advance one stochastic event using ReferenceNature inside the emulator.

    Verify receipt identity, including its outcome index, rather than
    implementing weighted-law roulette in Python. The caller decides whether
    the public continuation is complete, unsupported, or censored.
    """
    if frame.get("boundaryKind") not in NATURE_KINDS:
        raise ValueError("H15 ReferenceNature requires a Nature boundary")
    law = frame.get("chanceLaw")
    if (not isinstance(law, dict)
            or law.get("probabilityStatus") != "UNVERIFIED"
            or not isinstance(law.get("lawId"), str)
            or not isinstance(law.get("eventId"), str)
            or not isinstance(law.get("outcomes"), list)
            or not law["outcomes"]):
        raise ValueError("H15 requires an unverified declared Nature law")
    version = frame.get("version")
    if type(version) is not int or version < 0:
        raise ValueError("H15 Nature boundary omitted its version")
    advanced = backend.h15_experimental_reference_choose(handle, version)
    sampled = advanced.get("referenceNatureSelection")
    if not isinstance(sampled, dict):
        raise ValueError("ReferenceNature omitted its atomic selection receipt")
    index = sampled.get("outcomeIndex")
    if (type(index) is not int or not 0 <= index < len(law["outcomes"])
            or sampled.get("lawId") != law["lawId"]
            or sampled.get("eventId") != law["eventId"]
            or sampled.get("outcomeId") != law["outcomes"][index]["outcomeId"]
            or sampled.get("providerId") != REFERENCE_PROVIDER
            or advanced.get("version") != version + 1
            or advanced.get("h15Handle") != handle):
        raise ValueError("ReferenceNature receipt mismatches the advertised law")
    return advanced, sampled
