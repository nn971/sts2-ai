"""Version-owned H8 public actor boundary adapter (research M1).

This is a policy observation/choice/step bridge, not a full combat
episode runner. Nature requests, nested player choices, fixture entry,
terminal scoring and PPO replay are not yet routed through H8.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class H8Session(Protocol):
    def call(self, operation: str, **kwargs: Any) -> dict[str, Any]: ...


class H8PolicyContractError(RuntimeError):
    pass


FAIR_POLICY_ID = "prototype-fair-v0"


@dataclass(frozen=True)
class H8PublicDecision:
    observation: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    runtime_id: str
    boundary_version: int
    observation_hash: str

    def choose(self, actor_choose: Callable[
        [dict[str, Any], list[dict[str, Any]]], str,
    ]) -> str:
        selected = actor_choose(self.observation, self.legal_actions)
        if selected not in {
            action["actionId"] for action in self.legal_actions
        }:
            raise H8PolicyContractError(
                "H8 actor selected an action outside the source-public legal menu")
        return selected


def h8_public_decision(
    session: H8Session, status: dict[str, Any],
) -> H8PublicDecision:
    """Retrieve full genuine source-public H7 inputs, guarding the H8 token."""
    if (status.get("kind") != "player"
            or not isinstance(status.get("h8Handle"), str)
            or not isinstance(status.get("runtimeId"), str)
            or type(status.get("boundaryVersion")) is not int):
        raise H8PolicyContractError(
            "H8 public player boundary/status is absent")
    token = dict(
        h8_handle=status["h8Handle"],
        runtime_id=status["runtimeId"],
        boundary_version=status["boundaryVersion"],
    )
    packet = session.call(
        "h8_public_observe", **token, policy_id=FAIR_POLICY_ID,
    )
    if (packet.get("policyId") != FAIR_POLICY_ID
            or packet.get("runtimeId") != token["runtime_id"]
            or packet.get("boundaryVersion") != token["boundary_version"]):
        raise H8PolicyContractError(
            "H8 fair observation identity or source boundary drift")
    payload = packet.get("payloadJson")
    actions = packet.get("actions")
    if (not isinstance(payload, str) or not isinstance(actions, list)
            or not isinstance(packet.get("observationHash"), str)):
        raise H8PolicyContractError("H8 public observation wire is malformed")
    try:
        observation = json.loads(payload)
    except (ValueError, TypeError) as exc:
        raise H8PolicyContractError("Malformed source-public observation") from exc
    if not isinstance(observation, dict):
        raise H8PolicyContractError("Expected source-public observation object")
    if (not actions
            or any(not isinstance(a, dict)
                   or not isinstance(a.get("actionId"), str)
                   or not isinstance(a.get("kind"), str)
                   for a in actions)):
        raise H8PolicyContractError("H8 full legal action wire is malformed")
    ids = [a["actionId"] for a in actions]
    if len(set(ids)) != len(ids) or set(ids) != set(status.get("playerActionIds", ())):
        raise H8PolicyContractError("H8 full legal menu disagrees with status")
    return H8PublicDecision(
        observation, actions, token["runtime_id"],
        token["boundary_version"], packet["observationHash"],
    )


def h8_choose_and_step(
    session: H8Session,
    status: dict[str, Any],
    actor_choose: Callable[[dict[str, Any], list[dict[str, Any]]], str],
) -> dict[str, Any]:
    """One atomic H7-compatible policy decision on a live H8 session."""
    public = h8_public_decision(session, status)
    action_id = public.choose(actor_choose)
    result = session.call(
        "h8_step",
        h8_handle=status["h8Handle"],
        runtime_id=public.runtime_id,
        boundary_version=public.boundary_version,
        action_id=action_id,
    )
    nxt = result.get("result")
    if (not isinstance(nxt, dict)
            or nxt.get("runtimeId") != public.runtime_id
            or nxt.get("boundaryVersion") != public.boundary_version + 1):
        raise H8PolicyContractError(
            "H8 player step did not advance exactly one source boundary")
    return nxt
