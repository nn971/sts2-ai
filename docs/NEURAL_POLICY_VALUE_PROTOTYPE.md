# First neural policy/value prototype

**Status: experimental, opt-in. No demonstrated strength gain yet.**

## Architecture

The portable sts2-neural-policy-value-v1 model scores player-visible observations
and arbitrary legal actions using existing hashed semantic features.

- Shared state encoder: ReLU of a learned affine state projection.
- Value head: tanh of a learned projection into [-1,1].
- Policy head: an action-dependent nonlinear score for each legal action.
- Defaults: 256 input features, 32 hidden units.

Training uses optional PyTorch; exported JSON is loaded for inference using the
Python standard library, without PyTorch. Files have validated shapes and finite
weights; model-content SHA-256 hashes appear in MCTS search fingerprints.

## Reproducible commands (fish)

~~~fish
cd /path/to/sts2-ai
git pull --ff-only
git submodule update --init --recursive
python -m pip install -e '.[dev,neural]'

sts2-ai export-training results/strategy.sqlite results/search-roots.jsonl
sts2-ai train-neural results/search-roots.jsonl results/neural-v1.json \
    --epochs 12 --dimension 256 --hidden 32 \
    --json-output results/neural-v1-validation.json
~~~

For neural-value-guided MCTS with *heuristic* rollout actions:

~~~fish
sts2-ai evaluate --agent mcts --budget 32 --rollout-depth 32 \
    --seeds 5 --seed-prefix heldout-neural \
    --cutoff-model results/neural-v1.json --learned-weight 0.25 \
    --strategy-db results/neural-value.sqlite
~~~

For neural-policy rollout actions with *handcrafted* cutoff values:

~~~fish
sts2-ai evaluate --agent mcts --budget 32 --rollout-depth 32 \
    --seeds 5 --seed-prefix heldout-neural \
    --neural-rollout-model results/neural-v1.json \
    --strategy-db results/neural-policy.sqlite
~~~

The two options can be combined; test them independently first.

## Two distinct supervised value objectives

Without continuation records, the root teacher trains both heads: MCTS visit
distributions for policy, searched best-action mean for value.

With labeled continuation records, the root teacher trains only the policy
head. The value head trains instead against actual terminal win/loss
under the specified heuristic continuation policy, starting from an exact
hidden-state MCTS cutoff. These **must not** be conflated with optimal value
or fair-observation expected value.

~~~fish
sts2-ai train-neural results/search-roots.jsonl \
    results/neural-cutoff.json \
    --cutoff-continuations results/cutoff-continuations.jsonl \
    --json-output results/neural-cutoff-validation.json
~~~

Censored, nonterminal continuations have no value targets and remain censored.
Continuations are split by source run seed, removing exact-state and visible
observation overlaps across cutoff partitions. Root training observations
overlapping the cutoff holdout are also excluded, and conversely cutoff
training observations overlapping root holdout are excluded. Requires at
least two source run seeds and some completed training continuations.

## Success criteria

1. Measure independent held-out policy cross-entropy, top-1 and target entropy.
2. Measure value RMSE on **the correct target distribution**, reporting
   outcome class balance and fraction of censored continuations.
3. Compare against heuristic and original MCTS on paired *unseen* run seeds.
   Compare both actual runtime and emulator transition budgets.
4. Do not claim a stronger agent based only on offline training metrics.
5. Remember that the existing MCTS still uses oracle-exact hidden-state forks.
   The learned model itself sees only fair observations and legal actions.

A small model is intentional: this is an engineering and generalization
baseline, not yet a large sequence architecture. A separate neural CPU CI job
checks actual PyTorch training while the default CI remains lightweight.
