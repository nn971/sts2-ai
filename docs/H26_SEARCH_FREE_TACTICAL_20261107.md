# H26 — Search-free combat learning with public tactical damage

**Status:** draft pilot, stacked on H25. PR #176. Generated Overgrowth,
Silent A0. Initialization: H25 v2 R3 macro/combat checkpoints or H22 R95.

## Purpose

Evaluate whether many search-free Act trajectories and direct actor–critic
updates teach stronger combat play per hour than H25's full-depth PUCT.
The main learning target remains Act-clear rate; dense supervised macro floor
progress and public combat reward shaping support early learning.

## Current implementation

- Frozen macro and combat actor snapshots during full-Act collection.
- **Zero PUCT**: no searched roots, no exact-prefix fight replays, no
  search-policy distillation. H26 asserts search count is zero.
- H26-only public combat feature token, derived strictly from the existing
  observable per-hit intent_damage and intent_hits, current player Block/HP,
  and number of alive enemies. Unknown and heterogeneous attacks contribute
  to an explicit unknown count. Missing damage is **not assumed zero**.
  The neutral H24 resource-monotone value representation is unchanged.
- Independent combat actor–critic: terminal combat win credit plus frozen
  macro Act-clear continuation, bounded defeat-progress credit, and
  gamma-consistent HP and enemy-HP potential shaping. No auxiliary future-HP
  regression.
- Macro actor updated each round with terminal Act-clear policy gradient
  and BCE, plus separately predicted graded future floor progress.
- Independent macro/combat parameters and optimizers; resumable round
  checkpoints with model and optimizer SHA-256 manifests.
- Search-free rollout bypasses the H25 redundant macro-value inference
  for every combat action. Macro continuation is evaluated only during
  training for observed, *completed* fights.
- Defaults: 64 training runs, 16 heldout runs, 15 workers, CUDA shared
  actors, 0.99 combat discount, up to 4,096 combat transition updates.

## Intent-law fidelity gate

The pinned emulator already implements committed current intents, their
projected attack numbers, and source-shaped semantic random intent support.
It also includes `ChanceLawEngineV111.QueryIntent`, whose native NextInt grid
weights are source-verified **for the stochastic selection boundary**.

A current committed intent and a *possible next-turn* intent distribution are
different queries. The latter is conditioned on the state after the current
turn, including any player action, enemy status changes, summons, move-state
continuations and deaths. The current player observation omits some internal
AI-state counters. We must either demonstrate that they are fully
reconstructible from public history, or exclude cases where the law would
expose information not available to the player.

Accordingly H26 v1 consumes **current displayed intent damage**; it does
not predict, sample or fabricate next-turn intent probabilities. The exact
next-turn law belongs in a future emulator public stochastic API, derived
from the source-defined chance law and audited for public observability.
This is an explicit outstanding fidelity gate for H26-C, not a learned
intent predictor.

## Cheap native smoke (fish)

    cd ~/projects/sts2-ai
    git fetch origin
    git switch research/h26-search-free-tactical-20261107
    set emu ~/projects/sts2-emulator
    set rev (git -C $emu rev-parse HEAD)
    set h22 results/h22-h21-packed-a0-long-v1/models/joint-r0095.pt
    set prior results/h25-a0-combat-first-8h-v2-p64/models
    set out results/h26-tactical-searchfree-smoke

    python -u tools/train_h26_searchfree_a0.py \
      --initial-h22 $h22 \
      --initial-h25-macro $prior/macro-r0003.pt \
      --initial-h25-combat $prior/combat-r0003.pt \
      --emulator-root $emu --emulator-revision $rev \
      --output-root $out \
      --rounds 1 --runs-per-round 4 --heldout-runs 2 \
      --workers 2 --minibatch 8 --max-combat-samples 256 \
      --macro-update-every 1 --device cuda --execute \
      2>&1 | tee results/h26-tactical-searchfree-smoke.log

First run on one or two rounds, inspect the logs and verify that
`combat_search.roots=0` and `combat_search.transitions=0`, both
optimizer step counts are positive, and independent checkpoints load.
Then enlarge to 64 train/16 heldout/15 workers and compare with H25.

## Monitoring

    python tools/monitor_h25.py --root results/h26-tactical-searchfree-smoke --once

H25's monitor uses compatible round summaries and recognizes the
zero-search counters. The model outputs should be assessed on independent
heldout seeds rather than on the collected training trajectories.

## H26-C: source-defined next-intent law (opt-in)

The companion emulator branch `research/h26-public-intent-law-20261107`
exports the **immutable, player-knowable enemy rule catalog** plus exact
Native `NextInt` grid masses (not a peek at future outcomes). Generate it
once from the same emulator source revision as training:

```fish
set emu ~/projects/sts2-emulator
git -C $emu fetch origin
git -C $emu switch research/h26-public-intent-law-20261107
set rev (git -C $emu rev-parse HEAD)
mkdir -p results/h26-intent
dotnet run --project $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj \
    -- public-intent-catalog-v111 \
    > results/h26-intent/native-source-catalog.json
```

Then run the existing smoke with the additional argument
`--intent-catalog results/h26-intent/native-source-catalog.json`
and **the new emulator revision** `--emulator-revision $rev` under a new
output root. This switches to a separate H26 v2 experiment fingerprint.

The v2 catalog exports each enemy's **full source-defined AI state
graph**: move states and their successors, weighted random branches,
repetition restrictions, and conditional branches. H26 reconstructs the
successor AI state from the uniquely visible current move and, when
necessary, a per-episode history of previously announced intents across
completed turns. This history is recorded with each rollout decision
and reused unchanged during actor-critic updates.

State-machine random branches use exact Native `NextInt` integer-ticket
masses. Conditional branches use only public HP, powers, and formation
facts; rules requiring unrepresented formation slots, off-balance flags,
summon budgets, or ambiguous prior state remain `UNAVAILABLE`.
The encoder adds enemy-owned probability tokens or one explicit
`UNAVAILABLE` marker. Existing H21 pointers, H24 monotone values,
and H25 rollouts are unchanged.

The law is **conditional on the current announced intent executing normally**
and the enemy surviving to select another move; it is not a prediction of
which enemy or action will remain after unexecuted player choices. Existing
H21 state/action pointers and H24 monotone value encoding are retained.

The output `progress.json` includes
`rounds[].intent_law_coverage.train` and `.heldout`, including
`living_availability_rate`, `by_enemy` with unavailable reasons, and
`next_turn_support_audit`: a conservative consistency probe against
subsequent publicly observed moves under unchanged visible conditions.
Review this before enlarging the experiment. Source/native parity for
exceptional stun, spawn, formation and phase transitions remains gated.

## Constraints and remaining verification

- The tactical damage token supplies current displayed damage; next-turn
  intent probabilities are exact only where public conditioning is resolved.
  This does not imply the future attack-damage distribution is computed.
- A 0.1-scale potential is dense but bounded. Reward ordering and calibration
  should be evaluated against complete Act outcomes and heldout HP.
- CPU unit tests and CLI compile are the first gate; an actual pinned native
  smoke is required before a long scheduled run.
- No untested claim of significant improvement in clear rate or throughput.
