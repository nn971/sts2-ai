# First neural policy/value prototype

**Status: experimental, opt-in. No demonstrated strength gain yet.**

## Architecture

The portable sts2-neural-policy-value-v2-semantic-action model scores
player-visible observations and arbitrary legal actions. The legacy v1 model
remains loadable for reproducibility.

- Shared state encoder: ReLU of a learned affine state projection.
- Value head: tanh of a learned projection into [-1,1].
- Policy head: an action-dependent nonlinear score for each legal action.
- Defaults: 256 input features, 32 hidden units.
- V2 uses compact action-kind/card/target semantic features, with state context
  carried by the shared state encoder. V1 rehashed the entire state for every
  legal action and was over five times slower in the first MCTS pilot.

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

A first small real-emulator v1 pilot (40 teacher roots; 10 held-out roots) had
near-uniform policy targets (mean normalized entropy 0.995). All three
16-decision runs had reported frontier 1.0. MCTS time/run: baseline 4.10 s,
value-only 4.13 s, neural-policy-only 23.60 s. This is **not** evidence for a
strength improvement. V2 reduces redundant per-action state hashing and must
be independently benchmarked with the same workload.


## Follow-up v2 real emulator pilot

The v2 architecture was retrained from the same 40 searchable teacher roots
on the same pinned emulator and compared at fixed MCTS-12/rollout-depth-24
with 16 decisions per held-out seed:

| Configuration | v1 pilot (seconds/run) | v2 pilot (seconds/run) |
| --- | ---: | ---: |
| Handcrafted MCTS | 4.10 | 4.46 |
| Neural cutoff, 25% blend | 4.13 | 4.52 |
| Neural rollout policy | 23.60 | 6.95 |

The v2 learned rollout reduces measured execution time by about 3.4x relative
to the v1 policy pilot (different CI runners, so treat this as indicative).
Relative to the concurrent handwritten-policy baseline, the extra runtime
fell from ~5.75x to ~1.56x. All configurations retained the same reported
frontier progress of 1.00 on the single held-out truncated run; no win
or strategic improvement has been established. The v2 model's 10-root
holdout value RMSE was 0.05375, vs 0.05371 in the v1 pilot (not a
meaningful difference). Teacher normalized policy entropy remains 0.99488:
the visit supervision is essentially uniform, so stronger teacher data are
the next limiting factor.

V2 experiment and its saved checkpoint:
https://github.com/nn971/sts2-ai/actions/runs/37749836944

## Provenance-safe root validation (2026-10-08)

New search evidence records **every originating run seed** in a separate,
append-only SQLite table keyed by exact state and search configuration.
Existing databases are upgraded automatically when opened; historical roots
without source provenance continue to use the previous exact-state/fair-
observation grouped validation. New JSONL search-root examples also include
`source_run_seeds`, a possibly empty list.

The grouped train/validation split now joins all roots sharing an exact state,
a fair observation **or any originating run seed**, including transitive
connections. Consequently, samples from different floors of the same seed
cannot contaminate a held-out evaluation. If all states form one connected
component (for example because of repeated identical opening observations),
training/validation splitting intentionally fails rather than reporting a
misleading holdout score. Use additional independent seed sets, or explicitly
separate training and evaluation into different external datasets.

The matched-root shadow-teacher experiment's separate low/high search budgets
are not independent trajectories: both point to the same original run seeds.
The new Q-winner stability diagnostic in `compare-teachers` reports how
frequently the action with highest sampled mean value remains the same at
both search budgets, excluding ties and underexplored roots. It is an
uncalibrated consistency check, not a guarantee of optimal play.
