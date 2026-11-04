# P2/H22 opt-in packed H21 actor collation — 2026-11-05

## Evidence and status

The P2/H21 v4 offline public corpus contains 400 decisions from four generated-map
seeds; the paired input reports sample the same 128 decisions (105 combat,
23 macro), 31,737 objects, 13,433 typed relations, and 598 legal actions.
Both reports share the corpus SHA256
`acc22eaed62891cb595d82b4a28e9920721e836fe23a1a982c792fe7c2c8ebf9`,
public encoding SHA256
`b74d9c0af74efc6cd26c24b8b27ec6856a28e168619bdb8d6a5f3d9d556f3e10`,
and emulator DLL SHA256
`14589e3409aceaf14ce7fb2017fd9e7c19b26a5884ad46b3e9079313a9ea97b5`.
Original AI benchmark revision: `f0079afa47cdb8761ab2494648778ca6a83d839a`.

| Metric (batch 4, 96 timed batches / 384 decisions) | Reference | Packed |
| --- | ---: | ---: |
| Collate median per batch | 15.802 ms | 1.505 ms |
| Collate p95 per batch | 17.969 ms | 1.740 ms |
| Collate wall time summed | 1.512006 s | 0.144936 s |
| Timed decisions per second | 253.97 | 2649.45 |

This is a **10.43× CPU collation throughput ratio on one reference/packed
pair**, with exact tensor equality verified on matched public frames.
It is not yet an A/B/B/A confidence estimate or a full actor/search gain.

## Implementation boundary

This branch is stacked on H22 joint training PR #169. It carries the P2/H21
packed relational collator from PR #171 and introduces
`--actor-h21-collator reference|packed` for *shared inference*.
The default is `reference` and is omitted from the historical experiment
fingerprint; selecting `packed` is recorded in the new fingerprint.
The shared actor records `p2_h21_collator` in its statistics.

H16 and H18 still use their existing collation. The packed switch requires
`--representation h21 --actor-inference shared`, and the existing H21
fixed-shape batching exclusion remains in effect. Model architecture,
checkpoint format, optimizer collation, stochastic continuation, PUCT budget,
legal actions and RNG/Nature are unchanged. No new model is trained by this
code change.

The H22 v4 `prototype-ai-v1` public contract remains required; canonical-power
v5 `prototype-ai-v2` is a separately versioned schema and is not used by
this actor. The incomplete shop/later-floor coverage of the four-seed corpus
still limits model-wide performance generalization.

## Validation and acceptance

Run focused H22 actor/representation regression tests, plus the PR #171
reference-vs-packed tensor tests; inspect GitHub Actions results on this head.

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch perf/p2-h22-packed-collator-20261105
git pull --ff-only

python -m pytest -q tests/test_h22_h21_joint_training.py
```

For a *bounded integration smoke*, take the pinned emulator/checkpoints from
`docs/H22_H21_JOINT_TRAINING_STAGE_20261105.md` and run its existing
one-round command with `--actor-h21-collator packed` and a distinct output
directory. Complete at least one PUCT root and a representative macro
continuation; inspect actor statistics for `p2_h21_collator: packed`.

For speedup acceptance, run reference/packed/packed/reference **A/B/B/A**
cohorts on the same machine with an identical frozen H22 checkpoint,
same source seeds, search budgets, action menus, Nature mode, worker count,
and emulator binary. Record exact action/teacher replay checksums, actor
collate/transfer/forward wall, emulator and whole-cohort wall, PUCT roots,
transitions, completion/censoring, and CPU/GPU pressure. Repeat at
1/8/15 workers and actor batch sizes 1/4/8/16 where supported.
A short first-pass R/P pair is *diagnostic only*. Promote the packed
actor default only after full actor/search parity and consistent
nontrivial whole-cohort improvement. The pre-existing H16 fixed-shape
and fused-transport ratios are not applicable to H22/H21.

Training smoke approval still follows H22's separate source-compatible
emulator and real joint update gates; native-game parity is an independent
fidelity project.
