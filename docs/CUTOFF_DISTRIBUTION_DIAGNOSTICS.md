# Diagnosing root-to-rollout-cutoff distribution shift

This is an **oracle-exact search diagnostic**, not a fair-agent or model-quality
claim. The search tree can see true hidden RNG. Captured observations use
`prototype-fair-v0`, but their *selection* follows oracle-exact search.

## Why collect cutoffs?

The first linear value model was trained to predict the best visited search
action mean at a searched root. When it is used in MCTS, it is evaluated on
**nonterminal states at the end of heuristic rollouts**, which may have a
different distribution of phases, HP levels and combat positions. Better RMSE
against searched-root labels has so far failed to improve gameplay.

A cutoff observation is **unlabeled**. The value supplied by the incumbent
cutoff function is a *prediction*, not a continuation-return target. To train
on these states, a subsequent experiment should independently estimate their
continuation values, preserving rollout policy, ruleset, emulator revision,
horizon, seed, and information regime.

## Reproducible capture (fish)

```fish
sts2-ai evaluate \
    --agent mcts --budget 32 --rollout-depth 16 \
    --seeds 5 --seed-prefix cutoff-audit \
    --cutoff-samples results/cutoff-observations.jsonl \
    --cutoff-sample-every 32 --cutoff-sample-limit 10000 \
    --strategy-db results/cutoff-audit.sqlite

sts2-ai export-training \
    results/cutoff-audit.sqlite results/cutoff-roots.jsonl \
    --min-budget 32

sts2-ai diagnose-training \
    results/cutoff-roots.jsonl \
    --cutoff-samples results/cutoff-observations.jsonl \
    --json-output results/cutoff-diagnostic.json
```

When a strategy database contains more than one search configuration, pass
`--search-version` to `export-training`. Keep the captured
`cutoff_value_id`, seed prefix and rollout settings with the exported data.

The cutoff sampler is opt-in, retains every Nth callback (global callback
order), and stores the first bounded set of unique observations, keeping a
count of repeated sampled occurrences. No new RNG draws, emulator transitions,
or state hashes are required for capture. Rows are per (run seed, information
policy, observation hash), sorted deterministically. Capping the number of
unique observations introduces a capture-order bias; compare uncapped,
different-rate collections before drawing population-level conclusions.

The report includes the teacher visit-distribution Shannon entropy, entropy
normalized by the number of legal actions, effective number of actions,
top-action mass and fractions of concentrated or nearly uniform multi-action
targets. For root-to-cutoff shift it compares observed phase distributions,
a phase total-variation distance, combat share, HP fraction, act, floor, and
the overlap of cutoff observation hashes with training-root hashes.

These are descriptive marginal diagnostics. The unit of comparison is the
**unique captured sample row**, so a common state visited many times is not
automatically many independent examples. Sample counts are included
separately. Observation-hash overlap does not indicate the underlying hidden
exact state. Future experiments should stratify by seed and run, and use
independent held-out seeds for downstream gameplay validation.

## Next milestone

Collect a multi-seed baseline capture, quantify the measured shift, then
sample cutoff states for **independently labeled** continuation experiments
(e.g. deterministic heuristic continuations in oracle-exact mode, or a
separately identified stochastic sampling scheme in fair mode). Evaluate
cutoff predictors on those *cutoff-domain* returns using seed/group-aware
splits before attempting another model-driven MCTS benchmark.


## Independently labeled cutoff continuations

The next opt-in experiment captures the exact rollout-cutoff handle **before**
it is released, forks that hidden state, and plays a *separate* heuristic
continuation. The resulting label is the **observed terminal win/loss under the
specified continuation policy**. The model being evaluated by UCT supplies no
part of the target.

Example (fish):

```fish
sts2-ai evaluate \
    --agent mcts --budget 32 --rollout-depth 16 \
    --seeds 5 --seed-prefix cutoff-labels \
    --cutoff-continuations results/cutoff-continuations.jsonl \
    --continuation-every 64 --continuation-limit 64 \
    --continuation-max-decisions 512 \
    --strategy-db results/label-strategy.sqlite

sts2-ai diagnose-continuations results/cutoff-continuations.jsonl
sts2-ai diagnose-continuations results/cutoff-continuations.jsonl \
    --cutoff-model results/linear-model.json
```

The JSONL schema is `sts2-ai-cutoff-continuation-v1`. Records retain
the fair observation, exact source-state hash (provenance only), originating
run seed, cutoff reason, continuation policy, decision cap, decisions taken,
outcome, observed terminal act/floor, and a value of exactly +1 for victory
or -1 for defeat. A continuation that stops at the decision cap, or has
no legal actions while nonterminal, is marked `truncated` or `stuck` and
its `terminal_value` is **null**. These records remain in diagnostics but
are excluded from terminal-value RMSE.

This is **oracle-exact data** because the continuation starts from the real
hidden state. The action-selection policy used inside the continuation reads
only observations, but the branch's sampled future is exact. Model validation
on this data must be grouped across original run seeds, with special care
for equal visible observations reached from different hidden states.
A deterministic policy's return is a well-defined outcome for the sampled
hidden state, yet one continuation may have very high variance as an estimate
of the *fair* observation value.

Full-run heuristic continuations may mostly lose. Such labels are valid
outcomes, but a nearly all-defeat dataset is a poor standalone source
of discriminative value targets. Measure class prevalence and the distribution
of terminal progress before training. Future training should combine
broader policies, diverse states, calibration, and independently held-out seeds.

Sampling every Nth cutoff with a unique-state cap bounds costs. Exact hashing
and extra forked transitions only occur after an observation is selected.
The extra computation is not part of the primary MCTS simulation budget;
compare **emulator transitions and wall-clock**, not just simulations, when
using these experiments.
