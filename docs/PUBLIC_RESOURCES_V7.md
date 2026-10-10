# Public-resource combat encoder v7

This work is deliberately isolated from the 320-round PPO baseline (v6).
It uses the `agent/public-observations-v2` emulator submodule, preserving
the old experiment's artifacts and checkpoint.

## Player-observable input

- Full **unordered** draw-pile multiset (`combat.draw_pile`) rather than
  `draw_pile_count` alone. The encoder explicitly ignores instance IDs
  and array order, but uses card identity, multiplicity, upgrade level,
  temporary status and card-local metadata.
- Existing discard/exhaust/hand cards, now including upgrades and public
  per-card modifiers as well as card identities.
- Player-visible `combat.relic_counters` with per-trigger counts keyed by
  the owning relic. The top-level `relics[].state` is also encoded.
- Exact player-power stacks and exact enemy power/status magnitudes.
  The new tokens retain enemy ownership; the v6 displayed base/modified
  intent damage and reserved numeric coordinates are preserved.

## Safety and compatibility

- `public_resources_tactical_state_features` refuses observations lacking
  a draw-pile list or relic-counter list, and checks pile count agreement.
- These changes **do not** reveal the exact order of the draw pile, random
  seeds, future intents, or any emulator state handle.
- Combat model format `sts2-neural-policy-value-v7-public-resources-tactical`
  is distinct from v6. Existing v6 inference and models remain unchanged.
- When warm-starting a v7 combat model from a v6 checkpoint, state/value
  projections are reset rather than incorrectly treating changed features
  as the same coordinates. Compatible action/policy parameters are kept.
- The staged trainer pins the new emulator and starts a **new output directory
  and seed namespace**. It will refuse the old emulator checkout.
- Relic-count visibility may need a per-relic audit when expanding content
  beyond triggers derived from visible actions.

## Test commands (Fish shell)

```fish
git submodule update --init --recursive
python -m pytest -q tests/test_tactical_state_v7.py tests/test_tactical_state_v4.py \
    tests/test_phase_split_ppo.py tests/test_train_longrun_revision.py
(cd emulator; dotnet test tests/Sts2Emulator.Core.Tests \
    --filter FullyQualifiedName~PrototypeAiEnvironmentTests)
```

## Proposed next task

Change the **training episode terminal goal** from the full three-act
prototype to native Act 1 boss clear. This requires versioning the goal
and training checkpoint fingerprints, preserving uncensored boss-combat
transitions, and evaluating both versions on independent fixed seeds.
Do **not** simply relabel incomplete episodes as victories based on
`act >= 2` at arbitrary mid-run states without a certified transition.
