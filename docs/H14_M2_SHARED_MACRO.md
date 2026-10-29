# H14 M2 — shared macro actor/value with frozen H9 combat

**Status:** working structural and training-path implementation, untrained.
**Architecture:** [H14 canonical design](HIERARCHICAL_MUZERO_ARCHITECTURE.md).
**Stack:** PR #144 on M1 PR #143; emulator receipts PR #225.

## Boundary and feature contract

- `SharedObjectTransformer` and H11's semantic macro-action codec have been
  selectively recovered from the H11 research branch. They use a **new H14
  object schema and checkpoint format**. No H11 negative-result weights,
  H12 transition predictor, or H12 standalone continuation critic are loaded.
- Persistent Pen Nib `AttacksPlayed % 10` and Pendulum `TurnsSeen % 3`
  are separate typed numeric tokens owned by their respective relic copy.
  The shared codec reads only known fair-player `relics[].state` fields.
  Unknown nested relic state and RNG cursors stay out of the model.
- The macro value head receives visible map room/floor aggregates at *all*
  macro decision phases. At map-choice menus, candidate-owned multistep
  public topology from H11 supplies additional legal-action semantics.
  Shop/reward/event options and concrete item targets use published identities;
  opaque join IDs never become embeddings.
- The H14 macro policy and long-term value **share a trainable Transformer**;
  the independently frozen H9 tactical policy uses its **original H7
  combat object/action codec** and original H9 checkpoint format. Renaming
  the legacy codec's schema label for the in-memory collator does not
  re-encode the historical H9 token values.
- Combat executes with H9 until the emulator reaches its first noncombat
  reward/macro state, as recorded by the M1 combat boundary observer.
  H14's actor then selects the actual card/potion/relic reward actions.

## Training interface

`tools/train_h14_macro.py` implements initial native Overgrowth,
single-player Silent, Ascension 10 full-run macro policy optimization.

- One independent source seed for each live emulator run, fair observation
  policy, source SHA-pinned H9 model, revision-pinned emulator, and SHA-pinned
  per-round H14 behavior checkpoint.
- Macro actor samples legal moves; H9 combat uses deterministic greedy
  inference. Fifteen ordinary emulator workers are the default.
- For observed Act-1 clearance, terminal training return is +1; for a
  certified Act-1 defeat it is -1. Runs censored before a certified clear
  contribute no samples. Each macro decision shares the corresponding
  measured run label. This is a **sparse whole-run objective**; no H12
  gold/reward predictor or desirability-weighted chance likelihood.
- Clipped on-policy proximal policy optimization (PPO) uses stored behavior
  log-probability, measured run-return advantage against the behavior
  macro value, and the *same sampling temperature* used by the actor.
  The critic learns the measured run outcome using MSE. The network has
  an optional legal search-policy distillation interface for later search.
- Immutable per-round behavior checkpoint, final macro/optimizer checkpoints,
  run seeds, returned outcomes, and digest-protected `latest.json`.
  Each round also saves `round-NNNN-history.jsonl` with per-run macro
  decision history (act, floor, public boss identity, action, map room type,
  event choice and reward action), plus a recorded SHA256. Resumption enforces
  training-identity and checkpoint matching.

This is the first training path. Macro advantage estimation is initially
a whole-episode Monte Carlo return; refined temporal-difference and
combat-exit bootstrapping should be introduced only after the macro critic
is calibrated on independent heldout native runs. Model validation must
measure Act-1 clears, ranking, and reliability against H6-C/H9.

## Local invocation (fish)

Use the existing checkouts, retaining their independently chosen branches.
Build the emulator on the pinned revision first:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h14-m2-shared-macro-value-20261029
source .venv/bin/activate.fish
python -m pip install -e '.[dev,neural]'

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
dotnet build $emu/Sts2Emulator.sln -c Release

set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set h9 results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt
set h9sha (sha256sum $h9 | cut -d' ' -f1)

python -m pytest -q tests/test_h14_macro_codec.py tests/test_h14_macro_actor_critic.py tests/test_h14_hierarchical_agent.py

# Strict pin/preflight, no training:
python -u tools/train_h14_macro.py \
  --h9-checkpoint $h9 --h9-sha256 $h9sha \
  --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
  --output-root results/h14-macro-a10-r80

# Substantial experimental budget (15 ordinary workers):
python -u tools/train_h14_macro.py \
  --h9-checkpoint $h9 --h9-sha256 $h9sha \
  --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
  --output-root results/h14-macro-a10-r80 \
  --rounds 80 --runs-per-round 64 --workers 15 \
  --device cuda --ppo-minibatch 8 --execute
```

Preflight now launches the pinned emulator bridge, verifies an actual
Ascension 10 native reset, and encodes the first legal Neow macro menu. The
emulator's A5+ initial curse is **Ascender's Bane**; A0 uses Restlessness.
The strict native-start guard accepts two audited generation profiles:
`native-overgrowth-map-structure-v0.111.0-v1` (older emulator gitlink)
and `native-overgrowth-map-native-rng-v0.111.0-v1` (your pinned
`2c46c703` revision). All other map profiles are rejected.

If the previous attempt failed before producing `latest.json`, rerun the
same `--execute` command without deleting the result directory: the trainer
safely restarts the uncommitted first round, with deterministic initialization
derived from the run identity. After at least one completed round, use
`--resume`. Use disjoint heldout source seeds for actual policy evaluation.

## Validation and next promotion gate

The source-fair codec tests and optional CPU PyTorch tests verify counter
residues, map-context preservation, H9/H14 checkpoint separation, policy
and critic gradients, and checkpoint roundtrips. GitHub Actions runs
Python static analysis and a separate CPU neural test job.

**No long native run has been executed from the remote GitHub session.**
Prior H11 full-run evaluation was negative; the migrated representation
and improved training contract are hypotheses until substantial native
A10 heldout comparison establishes better behavior. Next: execute a complete
pilot cohort, inspect policy/action statistics, then run a meaningful
multi-seed heldout comparison against H6-C macro + frozen H9 combat.
