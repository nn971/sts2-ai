from __future__ import annotations

from sts2_ai.emulator import EmulatorBackend, StateHandle

from .schema import ReplayScenario


def replay_scenario(
    backend: EmulatorBackend,
    scenario: ReplayScenario,
) -> StateHandle:
    """Reconstruct a scenario and verify its exact-state hash.

    The returned handle is owned by the caller and must eventually be released.
    """

    state = backend.reset(scenario.run_seed)
    try:
        for action_id in scenario.action_history:
            matches = tuple(
                action
                for action in backend.legal_actions(state)
                if action.action_id == action_id
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"Replay action {action_id!r} has {len(matches)} legal matches"
                )

            transition = backend.step(state, matches[0])
            previous = state
            state = transition.child
            backend.release_many((previous,))

        actual_hash = backend.exact_hash(state)
        if actual_hash != scenario.exact_state_hash:
            raise RuntimeError(
                f"Scenario {scenario.scenario_id!r} replay hash mismatch: "
                f"expected {scenario.exact_state_hash}, got {actual_hash}"
            )

        return state
    except Exception:
        backend.release_many((state,))
        raise
