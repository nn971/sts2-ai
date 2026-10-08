# Stronger MCTS teacher and neural holdout

The initial neural-v2 pilot successfully trained a model but the 12-simulation
teacher's normalized action-visit entropy was 0.995, nearly uniform.
Increasing network size without improving supervision is unlikely to help.

## Action-semantics-aware teacher audit

The command **sts2-ai diagnose-training** now reports original visit entropy
and additional semantic-action metrics: literal/semantic action counts,
normalized semantic entropy, top semantic probability above uniform, minimum
visits per semantic action, value gap between top two estimated choices, and
agreement between the best-value and best-visited semantic choices.

Multiple copies of truly resolved identical decisions can be aggregated;
unknown semantics are deliberately **not** treated as equivalent, since
different shop purchases or event options can produce different outcomes.
No hidden state is used to resolve semantics; all classification uses the
observable frame plus legal action payloads.

A conservative signal-quality filter is available:

~~~fish
sts2-ai curate-teacher results/high-roots.jsonl \
    --min-budget 64 --min-visits-per-semantic-action 2 \
    --min-semantic-top-margin 0.08 \
    --min-semantic-value-gap 0.01 \
    --output results/curated-roots.jsonl \
    --json-output results/teacher-audit.json
~~~

The selected records retain their *original, unchanged* probabilities.
Quality filters are heuristics, not statistical confidence intervals or
guarantees that their chosen actions are optimal. Zero retained records is
a meaningful negative finding, not a reason to fabricate target sharpness.

## Automated experiment

The GitHub Actions workflow at
.github/workflows/teacher-quality-experiment.yml runs:

1. MCTS-12 vs MCTS-96 teacher generation with the same run seed prefixes.
   The root states may diverge after the first different decision.
2. Strict per-action and semantic-policy diagnostics and a named curation
   policy with fallback to all high-budget records if too few are accepted.
3. A small capped independently continued cutoff dataset spread across
   original source seeds; nonterminal continuations stay censored.
4. A neural-v2 checkpoint trained on the explicitly chosen teacher records.
5. Three *unseen* seed-matched MCTS-32 evaluations with equal depth: plain
   heuristic, learned value (25% blend), and neural rollout policy.
6. Seed-paired summaries of continuous frontier progress, victories,
   truncations, wall-clock and emulator transitions.

The workflow artifact is **stronger-teacher-neural-heldout**. It is a small
methodology pilot, not proof of improved gameplay. Follow-up experiments
need substantially more full-run heldout seeds and equal runtime budgets.

The emulator pin, default heuristic MCTS, and oracle-exact information-policy
boundary are unchanged.


## First completed measurement — October 8, 2026

The first larger experiment (https://github.com/nn971/sts2-ai/actions/runs/37752770078)
compared MCTS-12 to MCTS-96 on matching initial seed prefixes. Their game
trajectories can diverge, so aggregate teacher comparisons are confounded.

| Metric | MCTS-12 | MCTS-96 |
| --- | ---: | ---: |
| Searched roots | 71 | 68 |
| Mean normalized literal visit entropy | 0.99046 | 0.998997 |
| Mean semantic normalized entropy, multi-choice | 0.94751 | 0.95223 |
| Strictly curated high-budget roots | — | 9 |
| Time per run | 3.09 s | 15.48 s |

Since fewer than twelve high-budget records met the quality threshold, the
experiment recorded a fallback to all 68 high-budget roots. All six sampled
independent exact-state heuristic continuations terminated in **defeat**;
none were censored, but their value labels have no class discrimination.

Three unseen seed-matched MCTS-32 heldout runs, capped at 64 decisions each,
reported the following continuous frontier progress, rather than victories:

| Configuration | Mean progress | Mean time/run | Compared with baseline |
| --- | ---: | ---: | --- |
| Handcrafted MCTS | 3.67 | 11.79 s | baseline |
| Neural value (25% blend) | 2.67 | 12.97 s | behind on 3/3 |
| Neural rollout policy | 4.67 | 33.38 s | ahead on 2/3 |

All nine evaluations were truncated and none won. The value head performed
worse. The neural policy has an exploratory positive signal but costs 2.8x
runtime and is not established as stronger.

## Same-state shadow search

A later research hook allows exact-root teacher comparisons that do not
confound action quality with divergent run trajectories:

~~~fish
sts2-ai evaluate --agent mcts --budget 12 --shadow-budget 96 \
    --strategy-db results/low.sqlite \
    --shadow-strategy-db results/high-shadow.sqlite \
    --seed-prefix teacher-controlled --seeds 3 --max-decisions 28

sts2-ai export-training results/low.sqlite results/low.jsonl
sts2-ai export-training results/high-shadow.sqlite results/high.jsonl
sts2-ai compare-teachers results/low.jsonl results/high.jsonl
~~~

The higher-budget shadow runs from each exact live decision state before
the acting agent plays its selected action. The two exported datasets
match on source exact hash, fair observation, and legal actions. Shadow
searches do not change game decisions, but they consume extra diagnostic
compute, so the main run wall-clock is not a fair baseline benchmark.

For semantic curation, distinct map branches must never be equated simply
because they share a room type; only truly matching visible card plays
against the same target can be merged. Eligibility filters remain
uncalibrated heuristics, not proofs of correct action rankings.


## Why visit-count imitation is not necessarily the right policy objective

The acting MCTS agent selects the highest **estimated mean return**, using
visits as a tie-breaker, whereas the original policy head was trained on
the distribution of UCT **visit counts**. UCT allocates visits partly to
exploration, so its distribution can remain flat even when the mean-value
ranking is nonuniform.

A separate opt-in distiller now exports softmax policies over each identifiable
semantic action's *visit-weighted action Q estimate*. It distributes group
mass equally over truly identical visible card plays, but does not merge
different map paths or unresolved purchases.

~~~fish
sts2-ai distill-q-teacher results/high-roots.jsonl \
    results/q-roots.jsonl \
    --min-budget 64 --min-semantic-visits 2 \
    --min-value-gap 0.05 --uncertainty-scale 0.25 \
    --temperature 0.15 \
    --json-output results/q-target-report.json
~~~

An example is **rejected**, not artificially sharpened, if: the budget is
too low, some semantic action is underexplored, fewer than two actions
are distinguished, or the leading action Q values are not separated by
the configured gap and standard-error proxy. The proxy is deliberately
not presented as a calibrated confidence interval because tree-search
samples are correlated. Different configurations are encoded in the
policy-target-mode field in every exported training example.

For an experiment that trains only the policy (keeping handcrafted MCTS
cutoffs), use:

~~~fish
sts2-ai train-neural results/q-roots.jsonl results/q-policy.json \
    --root-value-weight 0 --json-output results/q-validation.json
sts2-ai evaluate --agent mcts --budget 32 \
    --neural-rollout-model results/q-policy.json \
    --seed-prefix q-heldout --seeds 5
~~~

The model's **value head remains untrained** in this regime. Do not use it
as a cutoff estimator. The matched-root workflow tests the conservative
Q target and a clearly labeled exploratory weaker gate. If neither
produces enough records, it does not train a new model at all.


## Measured same-root and Q-policy experiments

The first passive MCTS-12/MCTS-96 comparison used **131 identical exact
decision roots**, preserving the acting MCTS-12 trajectory:

https://github.com/nn971/sts2-ai/actions/runs/37753667685

The mean high-minus-low semantic entropy change was **+0.00450**, with
semantic entropy decreasing on 62.6% of matched multi-action roots, and
38 of 131 high-budget roots satisfying the broader audit filter.
The stricter curation criterion kept 26. This resolves the differing-route
confound but still does not show that simply increasing MCTS budget makes
the teacher much sharper.

The first Q-target pilot then used the older, exploratory Q-distiller
implementation:

https://github.com/nn971/sts2-ai/actions/runs/37754267114

- Conservative action-Q evidence threshold: only **3/131** roots accepted.
- Exploratory threshold (Q gap 0.02, no uncertainty multiplier): **43/131**
  accepted, allowing policy-only training.
- Two independent seed-matched MCTS-16 runs, capped at 48 decisions:
  mean frontier progress increased by 0.337 for Q-policy rollouts, one
  ahead and one behind; neural policy took 25.7 s/run versus 9.9 s/run.
- This is insufficient evidence to claim an improvement, and the
  exploratory gate is not statistically calibrated.

The stricter **Q-softmax-v2** format introduces a representability check:
if two different game actions produce identical neural-v2 action features
but imply conflicting policy targets, the root is rejected. For example,
two map branches of the same room type are not interchangeable.
Checkpoint metadata now also marks policy-only value heads untrained,
and the learned cutoff loader rejects using them as evaluators.
A rerun with those guards is the authoritative v2 experiment; the first
pilot remains archived for comparison.

## Exact-root Q-rank stability (2026-10-08)

The `compare-teachers` report now also distinguishes **Q-value rankings**
from UCT-visit target entropy. For identical exact roots at the two budgets,
we aggregate sampled action means by conservative semantic action grouping,
weighting each by its actual visit count. A winner is *resolved* only when at
least two semantic actions were sampled and the best and runner-up values
are not tied. The report includes the number of resolved pairs, Q-winner
agreement fraction, winner flip count, high-budget unresolved winners given
a resolved low-budget winner, and mean top-two Q gaps at each budget.

This makes the teacher's **directional consistency** measurable even if visit
counts are flat. It does not establish statistical significance: rollouts are
correlated, the high budget may change its preferred action for valid reasons,
and Q differences remain noisy. A confident policy should not be trained
solely because UCT visits are concentrated; inspect the matched-root
Q-winner consistency and Q-target rejection counts together.

## Six-seed replication and agreement-gated Q teacher (October 8, 2026)

The six held-out run-seed replication of the exploratory Q-softmax-v2
policy is complete:

https://github.com/nn971/sts2-ai/actions/runs/37755895476

| Six capped runs | Handcrafted MCTS-16 | Exploratory neural rollout |
| --- | ---: | ---: |
| Mean frontier progress | 3.5907 | 3.1894 |
| Mean wall seconds/run | 11.55 | 31.23 |
| Full-run victories | 0 | 0 |

The learned policy was ahead on 2 seeds and behind on 4, with mean frontier
difference **-0.4012**. All runs hit the 64-decision truncation limit, so
none of this establishes completed-run strength. The earlier two-seed positive
comparison did not replicate.

The same matched-root diagnostic found 113 roots at which both MCTS-12 and
MCTS-96 had a resolved best semantic action. The winner agreed on 67 and
**flipped on 46**, only **59.29% agreement**. The teacher is likely too unstable
for unfiltered action-Q imitation to deliver a reliable policy.

A new *consensus teacher* explicitly gates evidence on both search budgets:
it requires identical exact-root and legal-action provenance, enough
low-budget visits to cover all semantic alternatives, a non-tied low-budget
Q winner, agreement with the high-budget winner, and then the existing
high-budget Q gap / uncertainty / neural-action-representability checks.
Labels are copied from the high-budget softmax only when **all** gates pass;
they are never sharpened just to satisfy the filter.

Reproduce the consensus audit on an existing matched-root pair using fish:

~~~fish
sts2-ai distill-consensus-q-teacher \
    results/shadow/low-roots.jsonl \
    results/shadow/high-roots.jsonl \
    results/shadow/q-consensus.jsonl \
    --min-budget 64 --min-semantic-visits 2 \
    --min-value-gap 0.02 --uncertainty-scale 0 \
    --min-low-value-gap 0.01 \
    --json-output results/shadow/q-consensus-audit.json
~~~

This is a **correlated-search agreement check**, not an independent label,
confidence interval, or proof that the high-budget choice is optimal. Low and
high searches use the same hidden state and heuristic cutoff. If insufficient
roots survive, we should not automatically relax the thresholds. The
matched-root workflow now conditionally trains and evaluates a
consensus-gated policy only when at least twelve roots survive, and compares
it on the same six untouched run seeds. If fewer survive, the evidence
shortfall remains the experiment's legitimate result.

Longer-term improvements will need independent action-return measurements,
better exploration and broader complete-run coverage, rather than arbitrary
reweighting of one noisy teacher.
