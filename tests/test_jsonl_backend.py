from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sts2_ai.emulator import InformationPolicy, PrototypeJsonlBackend


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def request(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        op = payload["op"]
        if op == "hello":
            return {
                "ok": True,
                "wireSchemaId": "prototype-ai-jsonl-v0",
                "aiSchemaId": "prototype-ai-v0",
                "rulesetId": "prototype-silent-v0",
                "fairPolicyId": "prototype-fair-v0",
            }
        if op == "reset":
            return {"ok": True, "stateHandle": "s1", "terminal": False, "exactHash": "h1"}
        if op == "legal_actions":
            return {
                "ok": True,
                "actions": [
                    {
                        "actionId": "start:abc",
                        "kind": "start_run",
                        "payloadJson": "{}",
                    }
                ],
            }
        if op == "observe":
            return {
                "ok": True,
                "policyId": payload["policy_id"],
                "payloadJson": '{"phase":"RunStart"}',
                "observationHash": "oh1",
                "schemaId": "prototype-ai-v0",
            }
        if op == "step":
            return {
                "ok": True,
                "parent": "s1",
                "action": {
                    "actionId": payload["action_id"],
                    "kind": "start_run",
                    "payloadJson": "{}",
                },
                "child": "s2",
                "terminal": False,
                "exactHash": "h2",
            }
        if op == "fork":
            return {"ok": True, "child": "s3", "exactHash": "h1"}
        if op == "exact_hash":
            return {"ok": True, "exactHash": f"hash:{payload['state_handle']}"}
        if op == "is_terminal":
            return {"ok": True, "terminal": payload["state_handle"] == "terminal"}
        if op == "expand":
            return {
                "ok": True,
                "expansions": [
                    {
                        "parent": payload["state_handle"],
                        "action": {
                            "actionId": "a1",
                            "kind": "choice",
                            "payloadJson": '{"x":1}',
                        },
                        "child": "s4",
                        "terminal": True,
                        "exactHash": "h4",
                    }
                ],
            }
        if op == "release_many":
            states = payload["state_handles"]
            assert isinstance(states, list)
            return {"ok": True, "released": len(states)}
        raise AssertionError(f"Unexpected op {op!r}")

    def close(self) -> None:
        self.closed = True


def test_prototype_jsonl_backend_translates_wire_contract() -> None:
    client = FakeClient()
    backend = PrototypeJsonlBackend(client, emulator_revision="deadbeef")

    assert backend.binding_version == "prototype-ai-jsonl-v0"
    assert backend.emulator_revision == "deadbeef"
    assert backend.ruleset_id == "prototype-silent-v0"
    assert backend.fair_policy == InformationPolicy("prototype-fair-v0")

    state = backend.reset("seed")
    assert state == "s1"

    action = backend.legal_actions(state)[0]
    assert action.action_id == "start:abc"
    assert action.kind == "start_run"

    observation = backend.observe(state, backend.fair_policy)
    assert observation.observation_hash == "oh1"
    assert '"RunStart"' in observation.payload_json

    transition = backend.step(state, action)
    assert transition.parent == "s1"
    assert transition.child == "s2"
    assert not transition.terminal

    assert backend.fork(state) == "s3"
    assert backend.exact_hash("s2") == "hash:s2"
    assert backend.is_terminal("terminal")

    expansion = backend.expand(state)[0]
    assert expansion.child == "s4"
    assert expansion.terminal
    assert backend.release_many(["s2", "s3"]) == 2

    backend.close()
    assert client.closed
