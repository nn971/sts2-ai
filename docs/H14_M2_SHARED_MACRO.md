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
- The revised Act-1-only objective stops each episode at the first *observed*
  transition into Act 2, or certified defeat. The actor never collects
  post-Act-1 macro decisions; censored runs are excluded.
- Each macro transition, including an entire combat, earns a small
  `0.02 * newly_reached_floors` progress reward. The last transition
  also earns +1 for certified Act-1 clearance or -1 for defeat. Floor
  changes are computed between successive *public macro decisions*, with
  the terminal observed floor as the last boundary; no hidden combat
  outcomes or card/resource prices are shaped.
- A generalized advantage estimator (GAE; gamma=0.99, lambda=0.95) uses
  behavior-checkpoint value estimates of successor macro states, with
  zero bootstrap after terminal outcomes. Within each per-round on-policy
  cohort, GAE policy advantages are standardized once before PPO; the
  critic learns unnormalized TD(lambda) returns. Its output is linear,
  permitting targets above +1 from progress plus victory.
- The frozen H9 combat network retains its historical tanh value outputs.
  The network has an optional legal search-policy distillation interface
  for later search. H12 reward predictors and desirability-weighted
  chance likelihoods remain excluded.
- Immutable per-round behavior checkpoint, final macro/optimizer checkpoints,
  run seeds, returned outcomes, and digest-protected `latest.json`.
  Each round also saves `round-NNNN-history.jsonl` with per-run macro
  decision history (act, floor, public boss identity, action, map room type,
  event choice and reward action), plus a recorded SHA256. Resumption enforces
  training-identity and checkpoint matching.

This is the first shaped-return/GAE training path. A macro transition
may span an entire H9 combat, so the critic bootstraps at the next
macro decision. The bonus is deliberately small (at most 0.32 over 16
floors) relative to the two-point terminal win/loss gap. Treat shaping
as a temporary exploration aid; later reduce it if actual Act-1
performance improves. Value calibration and macro action ranking remain
independent heldout promotion gates.

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

python -m pytest -q tests/test_h14_macro_gae.py tests/test_h14_macro_codec.py tests/test_h14_macro_actor_critic.py tests/test_h14_hierarchical_agent.py

# Strict pin/preflight, no training:
python -u tools/train_h14_macro.py \
  --h9-checkpoint $h9 --h9-sha256 $h9sha \
  --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
  --output-root results/h14-macro-a10-gae-r80

# Substantial experimental budget (15 ordinary workers):
python -u tools/train_h14_macro.py \
  --h9-checkpoint $h9 --h9-sha256 $h9sha \
  --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
  --output-root results/h14-macro-a10-r80 \
  --rounds 80 --runs-per-round 64 --workers 15 \
  --gamma 0.99 --gae-lambda 0.95 --floor-reward 0.02 \
  --device cuda --ppo-minibatch 8 --execute
```

Preflight now launches the pinned emulator bridge, verifies an actual
Ascension 10 native reset, and encodes the first legal Neow macro menu. The
emulator's A5+ initial curse is **Ascender's Bane**; A0 uses Restlessness.
The strict native-start guard accepts two audited generation profiles:
`native-overgrowth-map-structure-v0.111.0-v1` (older emulator gitlink)
and `native-overgrowth-map-native-rng-v0.111.0-v1` (your pinned
`2c46c703` revision). All other map profiles are rejected.

**Preserve the old 23-round +/-1 cohort as a negative experiment.**
The revised trainer uses schema `sts2-h14-native-macro-ppo-gae-v2` and
checkpoint format `sts2-h14-macro-policy-value-v2`. Start it in a *new*
results directory as above. Do not `--resume` an earlier 23-round
checkpoint or reuse the old output directory: the reward semantics,
critic activation and training identity have changed. Once a new run
has completed a round, use `--resume` for that run only.
Use disjoint heldout source seeds for actual policy evaluation.

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
