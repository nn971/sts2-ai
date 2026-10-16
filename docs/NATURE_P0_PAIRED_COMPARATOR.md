# Paired two-site Nature vs ordinary emulator randomness — first frozen-player control

**Scope:** actual emulator source-routed Act 1 Silent combat, **two chance sites
only** for Nature: weighted eligible enemy intents and unresolved physical
card draws. Other randomness is engine-owned. Ordinary mode uses a private
committed hidden shuffle order and source-shaped intent RNG; Nature uses its
deferred draw provider. Both arms run the same legal combat rules and same
frozen public player. This is **not** a calibrated original-game chance law.

**AI branch:** `feature/nature-p0-paired-native-comparator`, stacked on
[PR #80](https://github.com/nn971/sts2-ai/pull/80).
**Emulator implementation pinned for the initial CI:**
[`8cbb15a`](https://github.com/nn971/sts2-emulator/commit/8cbb15ab32ba50dff5d40c214ef7a85266761694)
from `feature/nature-143-native-policy-comparator`. This is a local
Codex feature branch, **not** a merged stable integration contract.
The cross-repo test pins the exact SHA; update and rerun on any future
emulator semantic interface revision. No emulator source/gitlink changed.

## Exactly what is compared

1. An independently rebuilt **Nature** combat from source-route seed and
   Overgrowth/Underdocks region. Nature chooses `enemy_intent` by stable,
   filtered weighted intent ID and `draw_card` by eligible physical instance
   ID through the actual `semantic_*` protocol.
2. An independently rebuilt **ordinary emulator RNG** combat, started with
   `native_policy_baseline=True`, which privately commits one source-time
   Fisher–Yates order per shuffle epoch and uses its own intent RNG. This
   arm has no externally resolvable Nature tickets.
3. Require identical **`sourceOpeningFingerprint`** and first fair
   player observation after removing *only* public already-committed
   intent fields. Different visible intents, subsequent hands, actions,
   terminal outcome and exit HP are legitimate stochastic divergence.
4. In both arms use the **same player policy**: a pinned immutable exported
   v15 `model.json`, separate but identically seeded actor per arm, or a
   basic structural fair Strike/Defend policy for engineering smoke.
   The default v15 action scope is explicitly `silent-basic-v1` until
   additional card effects are supported by both execution modes.
5. Independently replay each arm from its own seed with every selected
   Nature action or player action ID, legal support fingerprint, public
   observation/menu fingerprint and resulting source-state fingerprint.
   Also save/load and compare exact terminal private snapshots.
6. Record individual terminal victory/defeat, exit HP, native shuffle
   counts, complete decision transcripts, Nature policy checkpoint,
   player model SHA-256, actor seed and **paired return delta**.
   Any unsupported operation or censored episode fails closed and is
   never reclassified as a loss.

The initial score is `1[victory] + max(exitHP,0)/100`, matching the
existing restricted two-site pilot. A pair's difference is diagnostic
of **the given frozen player's** exposure to alternative stochastic
outcomes. This does **not** measure whether *training* the player
against Nature improves strength.

## Fish commands (separate pinned emulator checkout)

```fish
set -l emulator /tmp/sts2-native-comparator
set -l dll "$emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll"
set -l rev (git -C "$emulator" rev-parse HEAD)

# First use five source-route smoke fixtures; no trained player required.
python -u tools/compare_nature_ordinary_combat.py \
    --emulator-dll "$dll" \
    --emulator-bindings "$emulator/bindings/python" \
    --emulator-revision "$rev" \
    --workers 15 \
    --output results/nature/paired-source-route-smoke.json

# Use your actual exported frozen v15 phase-split JSON checkpoint.
python -u tools/compare_nature_ordinary_combat.py \
    --emulator-dll "$dll" \
    --emulator-bindings "$emulator/bindings/python" \
    --emulator-revision "$rev" \
    --player-model /path/to/exported-v15/model.json \
    --player-action-scope silent-basic-v1 \
    --player-seed 2026 \
    --workers 15 \
    --output results/nature/paired-frozen-v15.json
```

The CLI also accepts `--player-sampling --player-temperature 1.0`.
Each arm starts with the same seeded player sampling generator; after
their observations diverge, selecting different actions is expected.

To define an explicit source-run-disjoint evaluation corpus, pass
`--source-fixtures /path/to/fixtures.jsonl` with one
`{"fixture_id":"...", "seed":"...", "region":"Overgrowth"}`
object per line. Every `(region,seed)` must occur at most once per
batch. Source-run groups must be split at the dataset level across
training/development/test. The five built-in test fixtures use emulator
source-route cases known to run in the native-comparator branch;
they are far too small for statistical judgment.

## Validation and evidence discipline

Dedicated `nature-paired-source-combat` CI builds the **exact feature
commit**, checks the Python contract unit tests, runs a genuine C#
comparison at **1 versus 15 workers**, and requires identical source-ordered
full transcript records/digest. Worker wall time is excluded from the
canonical transcript digest.

This is **Level 1 integration smoke** under
[the experiment evidence standard](https://github.com/nn971/sts2-ai/blob/main/docs/EXPERIMENT_EVIDENCE_STANDARD.md).
No significance, training benefit, native-game parity, or calibrated
reference-law claim follows from five combats, even if all paired
outcomes are victories. Substantial training, independent retraining
seeds, representative scenes, fair matched compute and a separate
source-run holdout remain mandatory for feature evaluation.

**Coordination boundary:** the user's emulator Codex continues the
unified stochastic architecture/Chance Law/ReferenceNature in
[issue #174](https://github.com/nn971/sts2-emulator/issues/174). This
AI branch consumes only the *currently implemented* comparator and
preserves the new contract's responsibilities. When ReferenceNature
and Chance Law arrive, their declared `p` becomes the reference
evaluation target and is independently validated. Native installed-game
fidelity remains the Inspector's separate concern.


## Verified first cross-repository acceptance (2026-10-16)

[Dedicated AI + pinned C# CI #38926378414](https://github.com/nn971/sts2-ai/actions/runs/38926378414) **passed**:
- Source-matched contract tests and Ruff; separately pinned emulator comparator built cleanly.
- Five source-route pairs completed **5 Nature victories and 5 ordinary
  comparator victories**, with matching pre-controller opening fingerprints,
  intent-normalized fair initial public views and both-arm scripted replay
  through genuine terminal states and private snapshot restoration.
- Running the same five independent source runs under a **one-worker pool**
  and a **15-worker-capacity pool** produced **identical complete pair
  records** and digest `a8ae5bb235e2293baaeafe1e6e0388adf8f872cc1ab30f36646912a3383ef9a9`.
  Because this smoke has five independent fixtures, the latter pool
  observed **five active worker processes**; it does not demonstrate
  15 simultaneously busy real combats. Test actual 15-worker saturation
  when a larger source-disjoint corpus is available.
- The structural fair player observed a mean Nature-minus-ordinary
  return of **−0.0100 synthetic pilot reward units** over those five
  real-emulator pairs, each under the declared unverified law.
- A separately generated **explicitly untrained** portable phase-split
  v15-shaped model completed five more real pairs using the frozen
  public-player adapter with `silent-basic-v1` action masking, including
  exact dual-arm replay. This is checkpoint *schema/inference* validation,
  not trained-v15 performance; its five-pair observed mean delta was
  −0.0300.

**Evidence level 1 only:** both microcomparisons are tiny and selected
for mechanical support. Their signs are not estimates of a Nature
training benefit and no feature judgment should be drawn. The
ordinary comparator also remains source-probability
`UNVERIFIED` and separate from pending Chance Law/ReferenceNature.
