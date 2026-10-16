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
  menu evaluated by the actor (`silent-basic-v1`);
- Chosen action ID/index, temperature, behavior log-probability,
  critic value and behavior-model ID;
- Full separate Nature / ordinary-comparator source-state replay transcript,
  source opening, terminal victory/defeat and verified exit HP.

Chance events are **never player PPO actions**. Full stochastic private
state remains inside the emulator. Nature's `q` is tracked separately;
original-game conditional reference `p` is **UNVERIFIED**.

`prepare_real_combat_ppo_samples` requires exact behavior-model SHA-256,
behavior model ID, matching action scope, temperature, original public
observations and legal menus. It recomputes the head's pre-update log-probs
and values in Torch and compares them with the saved behavior, failing
closed on stale policy parameters, policy architecture mismatch or
portable-vs-Torch inference discrepancy (tolerance 0.002). It requires
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
