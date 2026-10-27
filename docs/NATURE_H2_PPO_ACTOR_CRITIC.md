# H2 Nature PPO actor-critic — implementation and local training

Branch: `research/nature-h2-ppo-actor-critic-20261022`, based on contextual
Nature PR #90 and the **same** pinned emulator observation bridge
`c254bd99816ac2b89b350ec132fe1cfbb27dfac6` (emulator draft PR #180).
No changes to any emulator branch. The old tabular and REINFORCE
contextual Nature implementations remain selectable.

## Why

One 40×256 contextual REINFORCE trial (seed 29; frozen v8) showed
learned minus reference score `-0.00282857`, source-bootstrap 95% CI
`[-0.01004,+0.00446]`, and `KL(q||p)=0.00157083` on 1,000 held-out
worlds. It showed a working simulator interface and optimizer, **not**
a credible learned adversary. More REINFORCE seeds are lower priority
than testing the actual player's PPO training approach.

## Architecture and learning

`PpoDrawNature` uses `ContextualDrawNature`'s **complete player-visible
combat observation** feature encoder (256 hash coordinates) and a shared
24-dimensional tanh state embedding. Legal physical draw-card instances
are enumerated by the emulator's pre-outcome Chance Law. Each candidate
card model has a trainable embedding and bias; duplicate physical
instances remain separate possible outcomes. Actor:
`q(i|s) = (1-mix) p(i|s) + mix softmax(logit(s,card_i))`,
`mix=0.75`, retaining positive declared-law probability on every
eligible card. Enemy intents are still sampled according to the
emulator Chance Law, not controlled by Nature.

**New**: a trainable scalar state-value head predicts *Nature's*
expected return, `-player_score`, directly from that same state
embedding. Each on-policy Nature action records its full old q,
old value, chosen physical instance and visible-state features. Full
combats produce sparse terminal rewards. Reuse
`training.phase_split_ppo.gae_terminal` from the **player** trainer:
`gamma=1.0`, `lambda=0.98`. This gives Nature advantages and critic
regression targets. Normalize the advantages over the sampled cohort.
Apply **clipped PPO** with `epsilon=0.2`, entropy coefficient
`0.001`, value coefficient `0.5`, and PyTorch AdamW
`learning_rate=0.001`. Four epochs over at most 2,048 on-policy
decisions, minibatch 128 (each cohort has multiple optimizer steps).
Clip gradient norm at 1.0. Unlike our previous 40 REINFORCE updates,
40 PPO cohorts can have hundreds of optimizer updates.

Torch is needed **only in the training optimizer process**. The
15 independent emulator workers perform deterministic portable
Python inference and serve the same frozen v8 player. Parameter values
and **AdamW moments/steps** are JSON-checkpointed to resume identical
training. Versioned checkpoints cannot be confused with old contextual
REINFORCE checkpoints. Behavioral policy log q and critic predictions
are checked against the immutable rollout checkpoint before PPO update.

Diagnostics per training round include pre-update chance KL (existing),
PPO minibatch update count, PPO approximate behavior KL, clip fraction,
critic MSE, entropy, sampled decision count, and mean gradient norm.
Do not use the gradient norm as a measure of strategic competence.

This is a **PPO optimization experiment** with the existing compact
state encoder. It does not yet implement a transformer, reuse the
player's *network weights*, or provide counterfactual Q labels. If
PPO learns a nontrivial q but fails to worsen combat, the next
priority is feature quality and candidate-value supervision.
Full-fledged emulator branching for counterfactuals remains a separate
research extension.

## Reuse the existing H2 worktree

The following commands are Fish. Preserve local changes before
switching branches; the on-going H1 worktree is separate.

```fish
cd ~/projects/sts2-ai-h2
git status --short
git fetch origin
git switch --detach origin/research/nature-h2-ppo-actor-critic-20261022
git submodule update --init --recursive emulator
source .venv/bin/activate.fish
python -c 'import torch; print(torch.__version__)'
dotnet build emulator/Sts2Emulator.sln -c Release

set model ~/projects/sts2-ai/results/ppo-v8-batched-40/models/stage-0120.json
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set rev c254bd99816ac2b89b350ec132fe1cfbb27dfac6

python -u tools/probe_nature_h2_combat_profiles.py \
    --nature-model ppo \
    --scenarios mawler-a10,cubex-a10,vine-shambler-a10 \
    --emulator-dll $dll --emulator-bindings $bindings \
    --emulator-revision $rev

python -u tools/train_draw_only_nature_h1.py \
    --nature-model ppo --nature-lr 0.001 \
    --ppo-epochs 4 --ppo-batch-size 128 --ppo-sample-limit 2048 \
    --ppo-clip-epsilon 0.2 --ppo-value-weight 0.5 \
    --ppo-entropy-weight 0.001 --ppo-gamma 1.0 --ppo-gae-lambda 0.98 \
    --player-model $model \
    --emulator-dll $dll --emulator-bindings $bindings \
    --emulator-revision $rev --player-action-scope all-legal-v1 \
    --scenario-pool mawler-a10,cubex-a10,vine-shambler-a10 \
    --deck-pool silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1,silent-poison-kit-v1 \
    --training-seed 29 --actor-seed 31337 --reference-seed 23387 \
    --workers 15 --rounds 40 --episodes-per-round 256 \
    --eval-episodes 1000 \
    --output-root results/nature/h2-a10-ppo-v8-seed29 \
    --execute 2>&1 | tee h2-a10-ppo-v8-seed29.log
```

If PyTorch is absent: `python -m pip install -e '.[neural]'`.
If the process is interrupted after a completed round, repeat the
**exact same** training command with `--resume` appended. Do not
reuse an output root from a tabular or REINFORCE checkpoint.

Success criterion: large negative learned-minus-reference **paired**
player-score difference with source-bootstrap CI excluding zero on
held-out worlds, across further independent training seeds.
A strong training KL alone is not proof of adversarial competence.
Keep the full 40-round run; do not judge a new optimizer by a quick
20-round verdict. Fidelity calibration remains outside the training
gate, including when the report says `UNVERIFIED`.
