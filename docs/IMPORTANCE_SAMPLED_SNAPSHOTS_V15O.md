# v15o — importance-sampled combat snapshots (opt-in)

This stacked patch starts from v15n parallel snapshot PPO. **Existing v15l
full-run and v15m/v15n snapshot checkpoint semantics remain unchanged** for the
original `natural` and `balanced` modes. It does not claim that the observed
v15 200-round greedy combat policy is better than frozen v8; development
benchmark on 739 combats had v15 slightly lower exit HP and survival.

## What is implemented

`--snapshot-sampling importance` activates an empirical finite-corpus
target/proposal scheme. Let the training corpus contain N unique authentic
combat entries and K observed (progress, encounter tier) strata. If entry i
belongs to stratum h containing N_h entries, define

    p_i = (1-beta)/N + beta/(K * N_h)
    q_i = (1-lambda)*p_i + lambda*p_i*priority_i / sum_j(p_j*priority_j)
    w_i = p_i/q_i

Defaults: beta=0.5 (half natural empirical distribution, half
observed-stratum-balanced target); lambda=0.7 (70% priority proposal,
30% target proposal). Thus `q_i >= 0.3 p_i` and `w_i <= 10/3`.
There are **no invented missing strata**. The target p is explicitly over
the observed *training corpus*, not the inaccessible true game state
distribution. Mixed collector policies need their own coverage audit.

The initial, intentionally simple **proxy** priority is

    priority_i = 1 + tier_offset + 2*(1 - entry_hp/max_hp)
    tier_offset = {weak: 0, normal: 0.5, elite: 1, boss: 1.5}

This uses only recorded entry metadata, not a learned gradient estimate.
The pure sampler API also accepts external positive score overrides indexed by
(source seed, ascension, source policy, exact fingerprint), paving the way for
an independent RMS-gradient estimation job in v15p. Priority scores do not
change the target p.

Each sampled combat is an on-policy episode from a verified native source
prefix. The sampler returns p, q, priority and p/q for each draw; repeating
an entry uses different actor RNG but the same initial native hidden RNG.

The importance-weighted PPO path assigns `w_i / T_i` to each of the T_i
decisions in a resolved combat, so a 100-decision fight does not receive 10
times the weight of a 10-decision fight merely for being longer. Minibatch
actor, critic, and entropy losses use self-normalized weighted reductions;
advantages use weighted centering/scaling. This is **not an unbiased
estimator of an unclipped policy-gradient objective**: PPO clipping,
advantage normalization, minibatch normalization, and optional random
decision subsampling introduce approximations. It is a principled
importance-weighted PPO surrogate, not a proof of off-policy correction.

Importance mode refuses censored/unresolved combats instead of silently
dropping selected hard entries and biasing the target. Increase
`--max-decisions` if the collector encounters such fights.
Round reports include `importance_ess` and `importance_weight_max`.
Changing target/proposal controls changes the snapshot checkpoint
fingerprint; original unweighted fingerprints do not acquire these keys.

## Pilot: run a larger independent collection in WSL / Fish

First use 100 source runs for a smoke test, then a separate 1000-run
collection after verifying emulator CPU/throughput and resulting coverage.
**Collection is still serial** in `combat_snapshot_benchmark.py collect`;
v15n's parallelism applies to training/evaluation replay, not source-run
collection. A parallel resumable multi-policy collector is a distinct
follow-up patch; do not conflate the two.

~~~fish
git fetch origin
git switch agent/v15o-importance-sampled-combat
git submodule update --init --recursive

python -m pytest -q \
  tests/test_snapshot_importance_v15o.py \
  tests/test_snapshot_curriculum_v15m.py \
  tests/test_relational_ppo_v15.py

python -u tools/combat_snapshot_benchmark.py collect \
  --collector-model results/ppo-v8-batched-40/models/stage-0120.json \
  --runs 1000 --seed-prefix v15o-train-source \
  --output results/v15o-snapshots/all.jsonl

python -u tools/prepare_snapshot_curriculum.py \
  --corpus results/v15o-snapshots/all.jsonl \
  --output-dir results/v15o-snapshots/partition \
  --holdout-fraction 0.2
~~~

Inspect the partition manifest for unique runs, progress/tier and boss/elite
coverage before any training. Partition by *source run*, never by individual
combat. Reserve a new, uninspected seed prefix for final paired evaluation;
source-A is development data only. Scaling to 10,000 independent source runs
should be based on pilot yield and diversification, not a guaranteed number
of encounters.

## Three-way training ablation

Start all three pilots from the **same** frozen v8 base, compatible v15
relational architecture/vocabulary, optimizer settings and source-run-disjoint
training partition. A v15 round-200 warm-start is an additional separate
ablation; do not assume it is the best combat checkpoint. Example for the
importance arm:

~~~fish
python -u tools/train_pooled_objects_v15.py \
  --base-model results/ppo-v8-batched-40/models/stage-0120.json \
  --architecture relational --width 64 \
  --vocabulary results/ppo-v15j-relational-cpu64/vocabulary.json \
  --snapshot-corpus results/v15o-snapshots/partition/train.jsonl \
  --snapshot-sampling importance \
  --snapshot-target-balance 0.5 --snapshot-priority-mix 0.7 \
  --output-dir results/ppo-v15o-importance-pilot \
  --rounds 20 --episodes 32 --workers 8 \
  --max-decisions 512 \
  --ppo-batch-size 64 --ppo-epochs 3 --ppo-sample-limit 4096 \
  --snapshot-every 10 2>&1 | tee results/v15o-importance-pilot.log
~~~

Controls: `--snapshot-sampling natural` and
`--snapshot-sampling balanced`, with otherwise identical settings and new
output directories. Note that each mode defines/approximates a different
optimization target; to isolate *sampling efficiency* from target choice,
compare importance with lambda=0 and lambda=0.7 at the **same beta=0.5**.
Set `--snapshot-priority-mix 0` for the exact target baseline in this mode.

Do **not** select hyperparameters or the best milestone on reserved final
data. Report paired combat victory, exit HP, potion usage, enemy tiers, Act
progress, and clustered uncertainty. Also run fresh fixed-seed full Act 1
comparisons using the frozen macro policy.

## Known next steps

- Parallel, resumable, multi-policy source-run collection (currently serial).
- Stratification by build archetype and resource state, not only progress/tier.
- Independent policy-gradient RMS/variance priority estimates and cost
  adjustment using measured emulator work, with checkpointed priority versions.
- Investigation of importance-weighted learning at very small minibatch
  sizes (ESS, variance, clipping). Existing `sample_limit` selects decisions
  uniformly; avoid aggressive truncation in the first pilot.
- Exact native snapshot serialization/restoration for faster entry replay;
  current replay plus public/exact hash verification remains the oracle.
