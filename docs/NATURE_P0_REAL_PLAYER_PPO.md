# First real two-site Nature → v15 player PPO training boundary

**AI branch:** `feature/nature-p0-real-player-ppo`, stacked on
[paired ordinary/Nature comparator PR #81](https://github.com/nn971/sts2-ai/pull/81).
**Emulator:** pinned comparator feature commit
[`8cbb15a`](https://github.com/nn971/sts2-emulator/commit/8cbb15ab32ba50dff5d40c214ef7a85266761694).
Emulator source, native stream cursor fidelity, the user's unified stochastic
structure Codex work and existing v15 optimizer are untouched.

## Implemented boundary

The paired source-route worker can now run with
`capture_player_ppo=True` for a **sampled, immutable portable v15
phase-split** model. It records, for every actual player decision:

- The *fair public* observation and its content checksum;
- The exact **restricted source-issued** legal `action_id/kind/payload_json`
  menu evaluated by the actor (`silent-basic-v1`), together with the
  full original *public* source observation/menu and its replay support digest;
- Chosen action ID/index, temperature, behavior log-probability,
  critic value and behavior-model ID;
- Full separate Nature / ordinary-comparator source-state replay transcript,
  source opening, terminal victory/defeat and verified exit HP.

Chance events are **never player PPO actions**. Full stochastic private
state remains inside the emulator. Nature's `q` is tracked separately;
original-game conditional reference `p` is **UNVERIFIED**.

`prepare_real_combat_ppo_samples` requires exact behavior-model SHA-256,
behavior model ID, matching action scope, temperature, original public
observations and legal menus. It independently recomputes the complete source replay support digest
and checks it against the emulator's decision transcript, then reconstructs
the projected fair observation and restricted menu from the full
source-issued public inputs (including unchosen actions). It recomputes the
head's pre-update log-probs and values in Torch and compares them with the
saved behavior, failing closed on altered menus, stale policy parameters,
policy architecture mismatch or portable-vs-Torch inference discrepancy
(tolerance 0.002). It requires
complete terminal source-combat trajectories and rejects any unsupported,
censored or missing player-decision segment. The training target is
**the production v15 hp_preservation combat target**: defeat → 0;
victory → `1 − 0.5 × min(1, max(0, entryHP − exitHP)/maxHP)`, using
source-public entry/max HP and verified emulator terminal exit HP.

After verifying the cohort, the regular
`pooled_ppo_v15.ppo_update_pooled` executes clipped, minibatched
PPO+value updates for the genuine v15c pooled or v15d relational
residual head. Its original frozen v8/v13 base is unchanged.

The opt-in trainer may also call `TwoSiteNature.update` on the same
**completed** Nature-arm combat scores after the player PPO update.
Both optimizer states, portable player snapshots, chance-policy revisions,
training corpus and emulator DLL hashes are checkpointed. Resume
rechecks the exact experiment fingerprint, disjoint source groups and
architecture. No old PPO path is reinterpreted.

## Run with an actual trained portable v15 model (Fish)

Prepare a train-only source-run-disjoint JSONL file, one row:
`{"fixture_id":"train-0001","seed":"source-run-0001",
"region":"Overgrowth","partition":"train"}`.
The emulator currently supports restricted weak Act 1 source-routed scenes.

```fish
set -l emulator /tmp/sts2-native-comparator
set -l dll "$emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll"
set -l rev (git -C "$emulator" rev-parse HEAD)

python -u tools/train_nature_two_site_player_ppo.py \
  --initial-model /path/to/trained-portable-v15/model.json \
  --source-fixtures /path/to/train-only-source-runs.jsonl \
  --emulator-dll "$dll" \
  --emulator-bindings "$emulator/bindings/python" \
  --emulator-revision "$rev" \
  --training-arm nature --nature-update \
  --rounds 100 --episodes-per-round 128 --workers 15 \
  --output-root results/nature/p0-nature-player-seed29
```

Create an independent baseline run with `--training-arm ordinary` and
without `--nature-update`, starting from the same initial portable model
and source-run partition. **Research comparability additionally requires**
accounting for the extra Nature-optimization cost, identical selected
combat-action scope and a never-trained-on, source-run-disjoint
ordinary/reference evaluation set.

Only use `--resume` with exactly the same experiment specification,
pretrained model, source corpus and pinned emulator bytes. The trainer
keeps round-by-round portable behavior exports, Torch optimizer state,
source-provenance, and an updated portable final model.

## What is tested vs what remains

The focused tests cover public observation/menu integrity, exact behavior
log-probability, stochastic actor selection, checkpoint/source identity,
and stale/off-policy fail-closed behavior. Dedicated real-C# CI installs
CPU Torch, creates an **explicitly UNTRAINED test-only v15 pooled model**,
runs one Nature and one player update on five supported genuine combats,
and verifies that a normal v15 portable model is produced. This is a
**Level 1 optimizer-integration smoke only**.

For a decision-grade result we still need:
- a substantial distinct corpus of source-supported combats and a mature
  previously trained v15 checkpoint;
- expanded Silent card mechanics beyond `silent-basic-v1`;
- five or more separately trained replicas (adjust for power), genuine
  15-process saturation, equal budgets and independently evaluated
  training seeds;
- an immutable, source-run-disjoint ordinary / ReferenceNature holdout,
  stratified results, uncertainty over **both training seeds** and source
  runs, and learning-curve stability;
- harmonization with the upcoming independently verified Chance Law and
  ReferenceNature semantic contract.

See the mandatory
[experiment evidence standard](https://github.com/nn971/sts2-ai/blob/main/docs/EXPERIMENT_EVIDENCE_STANDARD.md).
Never infer improved Nature/player quality from one round, five encounters,
or a 20-round test. The underlying emulator original-game probabilities
remain `UNVERIFIED` until separately audited.


## Explicit count-only public draw piles (first real PPO integration finding)

The pinned semantic P0 fair observation can expose `combat.draw_pile`
as a **count** with physical card identities withheld, whereas the
legacy v15 public object encoder expected a complete array of known
card records. This is a genuine information-schema difference between
the two emulator interfaces, not a training numerical bug.

The new `semantic_p0_public_projection.py` converts a P0 count-only
zone to an **explicitly opaque** empty object list plus the **observed
count**, annotating `p0_opaque_public_fields`. It never invents
card-instance IDs, types, ordering or unseen effects. The v15 object
extractor recognizes this versioned marker and exposes
`p0_opaque:combat.draw_pile` as a categorical public-information
flag, preserving the original count as a scalar. Public count-only
discard/exhaust/play/choice zones likewise remain opaque, while their
observed counts are available as global numeric features. Contradictory
source count declarations fail closed. An older frozen checkpoint can
only use new numeric feature keys already present in its vocabulary. The strict legacy
invariant `draw_pile_count == len(draw_pile)` remains required outside
this explicitly versioned P0 projection.

Unprovided relic, potion or effect arrays are likewise marked as
**unobserved**, not asserted absent. The actor and PPO preparer both
use the **same recorded projected public observation** and thus the
same input features. This projection should eventually become a
native, versioned public-observation schema agreed with the user's
unified emulator stochastic interface. It is deliberately confined
to AI and does not alter the emulator game mechanics.


## Verified real-emulator PPO integration (2026-10-16)

[Focused cross-repository CI #38937309541](https://github.com/nn971/sts2-ai/actions/runs/38937309541)
**passed** with CPU Torch and independently built source-pinned C#
emulator `8cbb15a`:

- The new sampled public-PPO action collector and strict stale
  model/temperature/action-support tests passed. The P0 public projection
  tests verified the genuine hidden draw pile is **count-only**, with
  no invented card identities. Legacy v15 count/instance invariants
  remain enforced outside the projection.
- A **test-only, UNTRAINED** portable v15c pooled model with
  deliberately **nonzero residual policy/value gates** completed five
  genuinely simulated paired Nature/ordinary Act 1 combats, through
  five independently owned emulator processes. Nonzero gates force the
  portable v15 object encoder to exercise our public projection; the
  previous zero-gate test could bypass the encoder entirely.
- The Nature-arm cohort produced **123 actual player actions**, all
  verified against the source-issued action IDs and the *same*
  exported pre-update Torch residual policy. The production
  `ppo_update_pooled` performed **12 optimizer steps** with a finite
  reported loss. A separate Nature REINFORCE update completed on the
  same terminal HP-preservation scores. A new portable player checkpoint
  and atomic resumable Torch optimizer state were saved and validated.
- The observed five-combat mean v15 preservation target was **0.3886**;
  paired Nature-minus-ordinary pilot score was **−0.0080**. These are
  irrelevant as performance estimates at this scale. They are
  printed only to verify terminal reward plumbing and finite updates.

This is **Level 1 engineering/optimizer smoke**; no trained policy was
evaluated and no claimed player-strength improvement is supported.
The next research task is creating a substantial, strictly partitioned
source-run corpus and feeding an actual trained v15 checkpoint through
both ordinary/Nature arms for a matched-compute, replicated study.

## Candidate source splits and frozen-player holdout (2026-10-17)

The follow-up research instrument comprises these files:

- \`src/sts2_ai/nature/research_p0.py\`: homogeneous partitions,
  disjoint *source seeds across all regions*, production HP-preservation
  scores, full replay/menu provenance, and source-run-clustered bootstrap
  comparison on the ordinary comparator.
- \`tools/make_nature_p0_source_splits.py\`: reproducible, region-balanced
  train/development/evaluation **candidate** JSONLs and integrity manifest.
  Candidate generation is **not** source-support or fidelity verification:
  selected C# scenes must finish in the pinned emulator.
- \`tools/evaluate_nature_p0_player_ppo.py\`: immutable two-checkpoint
  source-run-disjoint holdout using the same sampled-action scope and a fixed
  untrained zero-mix Nature stress policy. The ordinary emulator arm is the
  **primary** outcome; the Nature arm is diagnostic. All actions are replay
  verified; the evaluator performs **zero gradient updates**.

Fish shell usage:

\`\`\`fish
python -u tools/make_nature_p0_source_splits.py \
  --seed 1731 --train 12800 --development 256 --evaluation 512 \
  --output results/nature/p0-source-splits-1731

# Run independent Nature and ordinary PPO jobs starting with the same
# pretrained v15 and the same train.jsonl, with matched player optimizer budget.

python -u tools/evaluate_nature_p0_player_ppo.py \
  --nature-trained results/nature/p0-nature-seed29/final-model.json \
  --ordinary-trained results/nature/p0-ordinary-seed29/final-model.json \
  --train-corpus results/nature/p0-source-splits-1731/train.jsonl \
  --holdout-corpus results/nature/p0-source-splits-1731/evaluation.jsonl \
  --emulator-dll "$dll" --emulator-bindings "$emulator/bindings/python" \
  --emulator-revision "$rev" --workers 15 \
  --output-root results/nature/p0-holdout-seed29
\`\`\`

The holdout difference is paired by *source-run seed*. Its bootstrap
reports conditional source-run uncertainty. Retraining-seed uncertainty,
multiple independent actor sampling replicates and matched total compute
(including Nature optimization) still require a replicated experiment.
The actor RNG seed includes its checkpoint digest: source scenes match,
but sampled player action noise is not common-random-number coupled.
The underlying source chance law and native-game fidelity remain
\`UNVERIFIED\`; reports explicitly set \`decision_grade: false\` and
\`research_verdict: NOT_ESTABLISHED\`.

CI exercises two previously unused C# routes and untrained test models
only to validate this protocol's plumbing, never to judge Nature.
