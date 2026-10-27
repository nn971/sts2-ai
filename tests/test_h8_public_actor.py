"""Contract tests for the H8 -> frozen H7 public action adapter."""
import json
import unittest

from sts2_ai.emulator.h8_public_actor import (
    H8PolicyContractError, h8_choose_and_step, h8_public_decision,
)


class FakeSession:
    def __init__(self):
        self.operations = []

    def call(self, operation, **kwargs):
        self.operations.append((operation, kwargs))
        if operation == "h8_public_observe":
            return {
                "policyId": "prototype-fair-v0",
                "runtimeId": "abc",
                "boundaryVersion": 9,
                "payloadJson": json.dumps({"hp": 35, "max_hp": 70}),
                "observationHash": "source-hash",
                "actions": [
                    {"actionId": "play:1", "kind": "play_card",
                     "payloadJson": "{}"},
                    {"actionId": "end-turn", "kind": "end_turn",
                     "payloadJson": "{}"},
                ],
            }
        if operation == "h8_step":
            return {"result": {
                "runtimeId": "abc", "boundaryVersion": 10,
                "kind": "enemy_intent"}}
        raise RuntimeError(operation)


STATUS = {
    "kind": "player", "h8Handle": "h8-1",
    "runtimeId": "abc", "boundaryVersion": 9,
    "playerActionIds": ["play:1", "end-turn"],
    "endTurnActionId": "end-turn",
}


class H8PublicActorTests(unittest.TestCase):
    def test_public_observe_exact_wire_and_versioned_actor_step(self):
        s = FakeSession()
        obs = h8_public_decision(s, STATUS)
        self.assertEqual(obs.observation["hp"], 35)
        self.assertEqual(len(obs.legal_actions), 2)
        self.assertEqual(s.operations[0][0], "h8_public_observe")
        nxt = h8_choose_and_step(
            s, STATUS, lambda state, actions: actions[0]["actionId"])
        self.assertEqual(nxt["boundaryVersion"], 10)
        self.assertEqual(s.operations[-1][1]["action_id"], "play:1")

    def test_illegal_actor_choice_fails_before_emulator_mutation(self):
        s = FakeSession()
        with self.assertRaisesRegex(H8PolicyContractError, "outside"):
            h8_choose_and_step(s, STATUS, lambda *_: "invented")
        self.assertEqual([x[0] for x in s.operations], ["h8_public_observe"])

    def test_stale_source_menus_fail_closed(self):
        s = FakeSession()
        with self.assertRaisesRegex(H8PolicyContractError, "disagrees"):
            h8_public_decision(
                s, {**STATUS, "playerActionIds": ["end-turn"]})
        self.assertEqual(len(s.operations), 1)

    def test_nonplayer_cannot_request_public_actor_frame(self):
        s = FakeSession()
        with self.assertRaisesRegex(H8PolicyContractError, "boundary"):
            h8_public_decision(s, {**STATUS, "kind": "draw_card"})
        self.assertFalse(s.operations)


if __name__ == "__main__":
    unittest.main()
