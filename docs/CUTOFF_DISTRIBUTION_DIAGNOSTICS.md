# Diagnosing root-to-rollout-cutoff distribution shift

This is an **oracle-exact search diagnostic**, not a fair-agent or model-quality
claim. The search tree can see true hidden RNG. Captured observations use
\`prototype-fair-v0\`, but their *selection* follows oracle-exact search.

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

\`\`\`fish
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
\`\`\`

When a strategy database contains more than one search configuration, pass
\`--search-version\` to \`export-training\`. Keep the captured
\`cutoff_value_id\`, seed prefix and rollout settings with the exported data.

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
