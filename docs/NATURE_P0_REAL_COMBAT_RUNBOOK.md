# Real two-site Nature combat — P0 implementation and acceptance

**AI PR:** [#79](https://github.com/nn971/sts2-ai/pull/79), stacked on Nature PR #78.
**Source-pinned emulator:** [integration commit `22e6114`](https://github.com/nn971/sts2-emulator/commit/22e6114db7c58990163892e28ce23abd2fa03024)
(merged issue #143 PR #172). The emulator code and existing gitlink remain unchanged.

## Actual interface (as implemented, NOT the earlier proposed semantic wire)

The source-backed `sts2-emulator` Python `Session` now advertises
`semanticChanceSchemaId=sts2-v111-semantic-playable-p0-v1`.
We use its actual methods:

- `begin_seeded_semantic_combat(seed,act,encounter_id,native_entry,total_floor,...)`;
  the engine commits source-shaped opening HP/intents/shuffle without a
  Nature-selected legacy Fisher–Yates permutation;
- `combat.observe()` and `combat.actions()` only at a `player`
  boundary, with unchanged `prototype-fair-v0` observation;
- `combat.step(actionId)` for legal player actions;
- `combat.resolve_intent(intentId)` using filtered semantic candidate
  `intentId` and `moveId`, post-filter `rawWeight` and
  `totalRawWeight` (all marked `UNVERIFIED` as audited source
  probabilities);
- `combat.resolve(selectedIndex)` for a `draw_card` selected
  from genuine eligible physical `cardInstanceId`/`cardModelId`;
- `combat.save_snapshot()`, session `load_semantic_snapshot()`,
  `release_semantic_many()` for private source-state restoration.

The prior `sts2-v111-semantic-chance-v1` adapter's proposed
`semantic_begin(fixture_id,seed)` fields are NOT how this emulator wire
works. This P0 adapter is a separate, exact-version consumer; previous
synthetic interfaces remain available for their earlier tests.

## Current AI implementation

- `emulator/semantic_p0_combat.py` negotiates the **actual** emulator
  handshake and source provider identities. It translates current
  weighted intent metadata / physical card IDs to the stable AI
  `SemanticEvent` and `TwoSiteNature` policy proposal interfaces,
  sends the exact intent ID or original draw candidate index, and records
  fair player actions and post-state fingerprints.
- `nature/real_combat_workers.py` launches independently owned emulator
  JSONL sessions in **15 Python OS workers by default**. Every complete
  real combat is played twice: once with legal policy proposals and
  again via strict scripted player/Nature selection, support identity
  and full mechanical transcript replay. Terminal snapshot restoration
  must match, and unrecognized/censored states fail rather than becoming
  artificial defeats. No hidden deck order enters the player policy.
- `tools/try_nature_real_combat.py` runs the real restricted Act 1
  Shrinker Beetle source-backed weak encounter with a fair
  *Strike→Defend→end_turn* structural player (not frozen v15).
  `mode=rollout` collects trace and outcome coverage;
  `mode=nature-smoke` performs tiny, genuine-terminal Nature
  REINFORCE updates on those combats, checkpointing the opponent.
  Neither mode compares trained player policies or asserts game strength.
- `.github/workflows/nature-p0-real-combat.yml` checks out/builds a
  *separate exact emulator SHA* and runs the AI's focused P0 contracts
  and first **15 independent JSONL workers**; CI smoke training uses
  two rounds purely to check actual gradient/update plumbing.

## WSL Fish acceptance command

Use a **separate checkout** of the exact emulator integration revision,
not the frozen all-Act main emulator. From `sts2-ai`:

```fish
git switch feature/nature-p0-real-combat-bridge
python -m pip install -e '.[dev]'

# Use your separately built pinned emulator checkout.
set -l emulator /tmp/sts2-nature-p0-emulator
set -l dll "$emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll"
set -l rev (git -C "$emulator" rev-parse HEAD)

# Actual real combat, 15 independent external JSONL processes and replay.
python -u tools/try_nature_real_combat.py \
    --emulator-dll "$dll" \
    --emulator-bindings "$emulator/bindings/python" \
    --emulator-revision "$rev" \
    --mode rollout --episodes 30 --workers 15 --seed 29 \
    --output results/nature/real-p0-coverage.json

# Optional end-to-end Nature optimizer **smoke** only.
python -u tools/try_nature_real_combat.py \
    --emulator-dll "$dll" \
    --emulator-bindings "$emulator/bindings/python" \
    --emulator-revision "$rev" \
    --mode nature-smoke --rounds 2 --episodes 15 --workers 15 \
    --output results/nature/real-p0-train-smoke.json
```

## Evidence standard and next training gate

See [mandatory evidence standard](https://github.com/nn971/sts2-ai/blob/main/docs/EXPERIMENT_EVIDENCE_STANDARD.md).
The above runs are **Level 1 engineering smoke tests**, independent
of their outcomes and durations. No Native RNG comparator exists for
the same matched initialized combat, so one cannot yet test the
**primary research claim** that Nature improves player strength.
Also, a first-legal/Strike-first structural player is not the v15
trained policy. Integrating v15 public observations/legal actions,
then a matched native RNG comparator and full training with at least
five independent seeds and 15 rollout workers, remains the next
research milestone.

The emulator's non-Nature cost/reward source assignments, full
entry/room effects and source conditional chance laws are **UNVERIFIED**.
`draw_card` is created only at an unresolved source-timed identity,
and already-known positions are consumed mechanically without an
additional Nature ticket. A hidden-source native shuffle and a Nature
deferred identity cannot both be counted for the same epoch.


## Verified AI→.NET integration (2026-10-16)

The complete [pinned emulator integration CI run #38878584942](https://github.com/nn971/sts2-ai/actions/runs/38878584942)
**passed**. The workflow separately checked out and built emulator
`22e6114db7c58990163892e28ce23abd2fa03024`, installed this AI
branch, ran the focused P0 unit tests, then drove real C# mechanics
through the external emulator Python/JSONL API:

- **15 independent OS workers / 15 seeded Act 1 Shrinker Beetle
  combats:** all 15 reached genuine `victory` under the
  legal Strike/Defend/end-turn player and the **untrained** two-site
  Nature proposal. Exactly 15 distinct worker processes were observed.
  Each combat's full interleaved player/Nature trace was replayed
  with scripted semantic selections; terminal fingerprint and saved
  private snapshot restoration matched.
- **Real Nature learning plumbing:** two REINFORCE optimizer updates,
  each on four complete terminal combats, ran successfully. The
  reported first-/second-round source-limited combat returns were
  1.5300 / 1.5450, win rate 1.000 / 1.000, and gradient norms
  0.10922 / 0.09495. These are **not native comparison returns**
  or evidence of strategic improvement.
- **Scope limitation:** this is an intentionally easy weak Act 1
  source-pinned encounter under a deterministic structural player,
  rather than frozen v15. The 15/15 victories provide a terminal/
  replay API test only. We need richer difficulty/encounters,
  source-run-disjoint data, frozen/v15 player actions, and a genuine
  matched native-RNG comparator before decision-grade training.

This first positive integration outcome moves the AI consumer from
synthetic-only to **real emulated combat**. It does not close the
native-reference comparison or game-source parity gaps in #143.
