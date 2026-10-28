# H10: jointly trained Silent A10 Act 1 Overgrowth agent

Status: **implementation started** on `research/h10-joint-act1-overgrowth-20261028`,
based on H9's 80-round structured-combat work (AI PR #133). The first
commit series implements measured tactical-opportunity priorities on the
existing native Act 1 snapshot + importance-PPO pipeline. It does **not**
yet run the H9 Torch combat checkpoint inside complete native Overgrowth
runs. No H9 or H6 checkpoint format is silently reinterpreted.

## Goal and objective

First measurable target: autonomous **Silent, Ascension 10, Act 1
Overgrowth boss clear** from real native-shaped run resets, outperforming
fixed baselines and eventually independently measured human runs on the
same emulator/game rules. The eventual objective is full-game A10.

A shared public-object encoder and distinct combat and macro policy/action
heads should be optimized for **Act-clear probability**. Combat also
optimizes short-horizon tactics; macro handles map, deck, relic, potion,
shop, event and campfire choices. The later combat boundary critic should
bootstrap from a frozen target macro continuation critic:

    G_combat = V_macro(public post-combat run state)

This avoids an unconditional HP-to-potion conversion rate. The outcome
predictor should represent a joint distribution over combat survival,
persistent HP, *named* potion inventory and other persistent effects.
Calibrate it on macro-generated entries and freeze/pin the tactical-policy
version generating each data cohort. The model does not reveal hidden
draw order, native RNG stream state or private emulator handles.

## Existing infrastructure to reuse

- H9 shared object/action Transformer, frozen Torch combat checkpoints
  and reference-law native complete-combat PPO
  (`docs/H9_NATIVE_STRUCTURED_TRAINING.md`).
- H6 phase-separated full-run learning, macro action semantics, fixed
  combat adaptation and run-level histories
  (`docs/H6_FROZEN_COMBAT_MACRO.md`).
- Authentic Act 1 combat snapshots: `evaluation/combat_snapshots.py`
  captures native reset seed + **public selected legal-action prefix**;
  restores/validates observation and exact-state fingerprints inside
  the emulator; forks combat entries. `snapshot_curriculum.py` already
  replays grouped prefixes and supports native isolated tactical PPO.
- `snapshot_importance.py` already supports full-support
  importance-weighted priority proposals. Do not duplicate it.
- Existing distributional outcome predictors are experimental and not
  yet conditioned on a deployed tactical-policy identity or calibrated
  for joint named-potion outcomes.

## Implementation stages

### M0 — tactical improvability for authentic macro histories (in progress)

Capture reachable native Act 1 snapshots using the existing snapshot
collector, on training-only macro seeds. Run the **same exact entry**
under the current tactical agent and a stronger candidate/reference.
Paired reports from `combat_snapshot_benchmark.py` are the evidence;
one exact RNG world is **not** a statistically reliable expected-regret
estimate. Priority labels are conservative:

- Current policy loses, reference wins: **reference_survives** (priority 5).
- Both win, reference has at least as much HP *and* retains a superset
  of the same named potion identities: **reference_dominates**
  (priority 1.5–2.5).
- Both lose: **unresolved** (priority 1), not `weak_deck`.
- Reference trades potions for HP: **resource_tradeoff_unresolved**
  (priority 1), pending strategic continuation values.
- Censored combat: priority 1; no fabricated defeat or win.
- If no reference does better, keep positive probability of sampling;
  periodically reconsider as the policy population improves.

The first implementation is in `h10_tactical_priorities.py` and
`tools/prepare_h10_tactical_curriculum.py`. A versioned JSON output is
matched to the **exact** training snapshot corpus hash, source seeds,
exact snapshot hashes and emulator revision. Ambiguous historical report
joins fail. Distinct reference policies can contribute independently;
the same current baseline must be used.

The opt-in `--snapshot-priorities` parameter of
`tools/train_pooled_objects_v15.py` passes those priorities into the
existing `build_importance_distribution` and on-policy PPO.
The target is preserved:

    p = mixture(natural snapshot frequency, balanced strata)
    q = (1-lambda)*p + lambda*normalized(p*priority)
    importance weight = p/q

There is a nonzero minimum sampling mass and the optimizer checkpoint
pins the complete priorities SHA. Existing training defaults, sample
weights and checkpoints remain unchanged without this option.

**Important scope:** This is a working *older v15 model* training bridge,
not yet the H9 Torch actor. H10's next integration should connect
H9-native decision encoding, batched inference and PPO to this same
verified source/priority machinery, without reverting to v15 capacity.
One repeated snapshot shares hidden initial RNG; new macro runs/source
seeds are required to obtain independent environment replicas.

### M1 — H9 combat within true native Act 1 runs

Integrate H9 policy inference with the full-run reset/step API. Preserve
all legal action queries, target pointers, potion and card choices.
Expose fair public observations and portable Torch checkpoints. Collect
actual combat-entry snapshots from H9-driven (or H9-combat/older-macro)
runs; all unsupported mechanics are coverage gaps, not fake victories.
Identify any remaining H9/H6 semantic incompatibilities directly;
do not modify training correctness to force acceptance.

### M2 — macro head and continuation critic

Attach a macro object/action-query adapter to the shared semantic
encoder. Verify different shop/event/card/upgrade/remove choices have
different input features and logits, map connectivity is explicit, and
treasure relic skip actions exist. Train macro PPO against real Act-clear
returns using a **frozen H9 combat pathway**. Keep independent optimizer
state and policy signatures. Censored runs have no artificial defeat label.

### M3 — joint survival/resource transition and macro lookahead

Fit a policy-versioned distribution over defeat versus joint exit
(HP, named potions, persistent deck/relic/gold changes). Preserve
correlation and calibrate on *run-seed-disjoint* source runs. Avoid using
on-policy outcome data as counterfactual evidence for a different tactic.
Use macro continuation value at the search frontier and simulate exact
known macro transitions wherever available. Compare macro PPO alone,
PPO+continuation critic, PPO+outcome-model lookahead, each with the same
frozen combat model.

### M4 — alternating or stabilized joint training

Iterate:
1. Frozen combat / exploratory macro: complete Act 1 runs; archive
   trajectories and authentic combat entries (including failed macro runs).
2. Train macro and policy-conditioned outcome model; run separate
   calibration/evaluation cohorts.
3. Freeze macro target critic; train combat from prioritized **entry
   states** with fresh on-policy actions, optionally using boundary
   continuation value. Preserve a natural-distribution portion and
   previous skills; remeasure reference gaps.
4. Refresh macro/outcome predictors against the new frozen combat policy.
   Never use a stale predictor as ground truth. Later unfreeze shared
   representations only with explicit stability gates.

Training/holdout partitions are by **originating full-run seed**, keeping
all entries from a run together. Held-out full-run A10 Act-clear rate,
boss-entry HP, potion inventory, deck quality and censoring are principal
metrics. Human comparison needs independently recorded same-rule runs.
Inspect representative decision histories for semantic failures rather
than inventing repeated microbenchmarks after negative results.


### M1 first integration hook — native H9/historical-macro hybrid collector

The opt-in `agents/h10_hybrid.py` uses H9's frozen Torch
`TorchH7PublicPolicy` only in live combat and the existing semantic
`PhaseSplitGreedyAgent` outside combat. It checks fair policy frames,
selects only supplied legal actions, and never exposes hidden RNG.
`tools/collect_h10_native_combat_entries.py` drives actual Silent A10
Act 1 Overgrowth runs, collecting native snapshot recipes containing
the source run's legal-action prefix and exact-state hash.

Example fish command (substitute an existing, compatible H6 macro
checkpoint; H9's R80 path is from your completed local run):

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
python -u tools/collect_h10_native_combat_entries.py \
  --macro-model results/your-phase-split-macro.json \
  --combat-checkpoint results/h9-structured-a10-r80/models/player-r0080.pt \
  --emulator-root $emu --emulator-revision $rev \
  --runs 16 --seed-prefix h10-a10-hybrid-train-v1 \
  --device cuda --output results/h10-snapshots/hybrid-entries.jsonl
```

This is an integration gate, **not yet validated on your local emulator**.
Verify an ordinary full-run combat is playable under the H9 codec; failure
on a new native card/mechanic is a concrete feature-coverage task.
Current collector is serial and runs greedy macro; add 15-worker parallel
collection only after the basic native codec proves compatible.
## First opt-in use (existing older v15 snapshot backend)

This stage requires **real native snapshot recipes** and paired reports
already generated by the existing benchmark tool. Neither an H9 Torch
model nor a public-only observation JSON by itself can be passed as an
exact emulator restart recipe.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h10-joint-act1-overgrowth-20261028
git pull --ff-only
source .venv/bin/activate.fish

# Use a TRAIN partition and compare current vs reference on those recipes.
# All inputs must have the same native emulator revision.
python -u tools/combat_snapshot_benchmark.py evaluate \
  --corpus results/h10-snapshots/train.jsonl \
  --baseline-model results/current-v15-model.json \
  --candidate-model results/reference-v15-model.json \
  --report results/h10-snapshots/current-vs-reference.json

python -u tools/prepare_h10_tactical_curriculum.py \
  --train-corpus results/h10-snapshots/train.jsonl \
  --comparison results/h10-snapshots/current-vs-reference.json \
  --output results/h10-snapshots/priorities.json
```

Then supply `--snapshot-corpus results/h10-snapshots/train.jsonl`,
`--snapshot-sampling importance`, and
`--snapshot-priorities results/h10-snapshots/priorities.json` to
`tools/train_pooled_objects_v15.py` along with its normal pinned base
model, vocabulary, optimizer, and checkpoint settings. This is a
**staging-only legacy bridge**, not a recommendation to invest in further
v15 training instead of integrating H9.

## Release gates

- Source runs/fingerprints reproducible; no evaluator handle or hidden
  RNG supplied to either agent; all candidate actions legal.
- Macro provenance and paired comparison must refer to the same
  configurations/emulator revision; ambiguity and mismatched priorities
  fail closed. Immutable result artifacts and new checkpoint fingerprints.
- Reference strength evaluated on separate source runs, not asserted from
  individual examples; one comparison world can suggest a tactical gap but
  cannot certify deck hopelessness.
- H9 full-run adapter tested for original game mechanics and action
  semantics; no fabricated terminal outcomes.
- Only complete native Act 1 runs count for Act-clear performance;
  separate selection and final confirmation by source seed. Preserve
  negative results and avoid needless successive tests of disproven ideas.

## H10 native run unblock: Power-card lifecycle (2026-10-28)

The first hybrid collector reached two full runs and failed on
`h10-native-a10-collector-v1-2` with
`InvalidOperationException: Combat card instance 17 is missing`.
The H9 emulator base revision `3420dde4b6eaeeb46f0925eda817acab3fe93eeb`
lacked the Power-card post-play fix already implemented separately
in emulator PR #189 / issue #188. That existing fix was ported
without changing the chance law to **emulator PR #221**:
`fix/h10-power-card-lifecycle-20261028`
(`e921af145c6cd0c69b5cff43428ee89dfc3342a8`).
The exact H10 crash seed has NOT yet been rerun, so this is a
well-supported candidate fix, not a completed end-to-end validation.

Switch the existing emulator checkout to the isolated fix branch only
if it has no uncommitted changes; the H10 Python collector's
`--emulator-revision` explicitly accepts this checkout. The bridge
uses `dotnet run --no-build`, so **rebuild** after switching.

```fish
cd ~/projects/sts2-emulator
git status --short
git fetch origin
git switch --track origin/fix/h10-power-card-lifecycle-20261028
dotnet test tests/Sts2Emulator.Core.Tests/Sts2Emulator.Core.Tests.csproj \
    -c Release --filter FullyQualifiedName~PrototypePowerCardCompletionLifecycleTests
dotnet build src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release

cd ~/projects/sts2-ai
source .venv/bin/activate.fish
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
python -u tools/collect_h10_native_combat_entries.py \
    --macro-model results/nature/h6c-macro-attention-r80.json \
    --combat-checkpoint results/h9-structured-a10-r80/models/player-r0080.pt \
    --emulator-root $emu --emulator-revision $rev \
    --runs 16 --seed-prefix h10-native-a10-collector-v1 \
    --device cuda --output results/h10-snapshots/hybrid-entries.jsonl
```

`--output` is created atomically only after all runs finish; interrupted
attempts do not create a trusted corpus. Never bypass an illegal card
transition or omit the failing run. If a different missing card error
persists, capture the selected legal action and the managed stack trace
rather than assuming this specific lifecycle bug explains every case.

### H10 next unblock: legacy generated event-gold request

On the same seed after the Power-card fix, an unrelated stochastic
interface mismatch raised:
`ArgumentException: Legacy site 'event.generated_gold.index_draw' requires
a bounded-integer request`.

The legacy event-gold helper previously used an enumerated support
`[minimum, ..., minimum + count - 1]` via `choose_index`. The
`StochasticSiteRegistryV111` requires `bounded_integer` for this site.
Emulator PR #221 now uses `minimum + ResolveLegacyBoundedInteger(count)`,
preserving the resulting discrete gold range and legacy modulo RNG
consumption/stream. It does not change native Chance Law ownership.

The H10 collector now accepts `--start-index 2 --runs 1` to
**replay the exact failing seed first**, and prints the last legal
action on exceptions:

```fish
cd ~/projects/sts2-emulator
git pull --ff-only
dotnet build src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release

cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
python -u tools/collect_h10_native_combat_entries.py \
    --macro-model results/nature/h6c-macro-attention-r80.json \
    --combat-checkpoint results/h9-structured-a10-r80/models/player-r0080.pt \
    --emulator-root $emu --emulator-revision $rev \
    --start-index 2 --runs 1 \
    --seed-prefix h10-native-a10-collector-v1 \
    --device cuda --output results/h10-snapshots/hybrid-entry-seed2.jsonl
```

Use a **different output** for the one-seed replay. Once it succeeds,
run the entire 16-run collector on the same revisions and original
seed prefix. Record any further real emulator exceptions and fix the
underlying call site rather than omitting a difficult seed.

### H10 Neow/HeftyTablet: legacy reward-shuffle registration

After the two earlier emulator fixes, the first three original H10
runs complete (3, 4 and 6 combat entries, each ending in defeat).
Seed `h10-native-a10-collector-v1-3` then failed at decision 0:

```text
event_choice: {"ChoiceId":"take_proto.native.neow.heftytablet"}
NotSupportedException: Native Fisher-Yates does not support shuffle
site 'run.random_pool_distinct.shuffle_swap_index'.
```

HeftyTablet generates a three-Rare-card choice. That follows
`PickDistinct` → `LegacyFisherYatesV111` → shared
`NativeFisherYatesV111.ShuffleInPlace`. The shuffle site was
already registered for legacy `reward` stream bounded draws,
but the shared algorithm checked the separate **native-only**
shuffle site list. Emulator PR #221 adds an explicit
`allowRegisteredLegacySite` pathway for the legacy adapter,
requiring exactly the registered site, stream, and operation;
the native provider's sites remain restricted.

Retry only the new failing source seed on the newly built emulator:

```fish
cd ~/projects/sts2-emulator
git pull --ff-only
dotnet build src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release

cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
python -u tools/collect_h10_native_combat_entries.py \
    --macro-model results/nature/h6c-macro-attention-r80.json \
    --combat-checkpoint results/h9-structured-a10-r80/models/player-r0080.pt \
    --emulator-root $emu --emulator-revision $rev \
    --start-index 3 --runs 1 \
    --seed-prefix h10-native-a10-collector-v1 \
    --device cuda --output results/h10-snapshots/hybrid-entry-seed3.jsonl
```

After this seed succeeds, run a fresh full 16-run corpus with the
same updated emulator revision. The original partial corpus was
discarded on the exception; prior provisional entries from older
revisions must not be concatenated with this one.
