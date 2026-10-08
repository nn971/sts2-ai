# Self-improving stochastic PUCT: bounded variance roadmap

**Status:** approved design direction / implementation plan, **not implemented**.
**Scope:** single-player Silent; prioritize native Act 1 Overgrowth coverage in
the pinned emulator, then expand to longer runs. Multiplayer-only and
cross-character reward mechanics remain out of scope.
**Baseline date:** 2026-10-08.

This is the next strategic-AI development plan. It supersedes the *ordering*
of the earlier search-first / handcrafted-teacher roadmap, without deleting
the earlier oracle experiments or their evidence. The new objective is a
**teacher-free (no expert demonstrations required), AlphaZero-inspired**
policy/search/learning loop with correct player-visible information and a
**small, bounded mean/variance auxiliary value model**.

Our baseline already has a JSONL emulator bridge, exact-state UCT, persistent
search evidence, a portable 256-feature/32-hidden-unit neural policy/value
prototype, seed-paired evaluation, and optional PyTorch training. It does
**not** yet have fair hidden-randomness sampling, PUCT selection, a
self-improvement loop, or trained variance predictions. No claim of a
stronger learned agent is justified yet.

## 0. Non-negotiable decisions

1. **Knowledge of the rules is allowed; foreknowledge of RNG is not.**
   The agent may use the known probability laws for shuffles, draws, rewards,
   and other stochastic mechanics, conditioned on the complete *observable
   history*. It must not get the actual seed, internal RNG cursor/state, or
   counterfactual future outcomes from the current live run.
2. **Use known chance laws in the search engine, not a learned RNG model.**
   Enumerate small discrete chance supports exactly; otherwise sample
   conditionally correct futures with reproducible *search-side* random seeds.
   Merely deleting the seed field from a neural observation is insufficient:
   forking the true hidden emulator state still gives oracle-exact search.
3. **Start with bounded, uniformized variance, not raw second moments and
   not full distributional RL.** Predict auxiliary outcome mean and variance
   directly in a normalized representation. Do not compute the predicted
   variance by subtracting two separately predicted raw moments.
4. **Keep the terminal goal primary.** Auxiliary progress/health may accelerate
   early training, with annealing. Evaluate against a fixed primary goal, not
   against the changing shaped reward.
5. **Keep search/data provenance strict.** Oracle-exact research results,
   fair stochastic experiments, incomplete/censored runs, and different
   emulator/reward specifications must not be mixed silently.
6. **No requirement for a strong external teacher.** The current network
   guides PUCT; searches and actual simulated outcomes train the next
   network. Handcrafted heuristics remain optional warm starts and
   competitive baselines, not compulsory supervision.

## 1. Precisely define "uniformized variance"

For a *bounded scalar auxiliary outcome* \(X\in[L,U]\), with fixed,
versioned \(L<U\), put

\[
Y=\frac{X-L}{U-L}\in[0,1],\qquad
\mu=\mathbb E[Y\mid h],\qquad
v=4\operatorname{Var}(Y\mid h)\in[0,1].
\]

Here \(h\) denotes information a player can legitimately possess: the
public observation and any remembered observations/actions. The factor
four follows the universal bound \(\operatorname{Var}(Y)\le 1/4\).
Both targets and outputs are dimensionless on comparable scales.

**Chosen implementation:** predict \(\hat\mu=\operatorname{sigmoid}(z_\mu)\)
and a dispersion \(d=\operatorname{sigmoid}(z_d)\), then define

\[
\hat v=4\hat\mu(1-\hat\mu)d.
\]

This construction guarantees \(0\le\hat v\le 4\hat\mu(1-\hat\mu)\le1\),
so the predicted mean/variance always admit some distribution on
\([0,1]\). It is not a beta-distribution assumption and does not
impose a Gaussian shape. Crucially, the variance output is **not**
formed as \(\widehat{\mathbb E[Y^2]}-\hat\mu^2\), which can suffer from
cancellation or yield negative estimates.

**Targets:** for \(n\) independently sampled, information-consistent
returns \(Y_1,\ldots,Y_n\) from the same public root and downstream
policy, estimate mean and variance using **Welford's stable online
algorithm**, not by subtracting accumulated squares. For tiny samples,
mask variance loss until \(n\ge 8\); store \(n\), estimator type,
sampling policy, and uncertainty. The unbiased sample variance may
slightly exceed the population theoretical upper bound; clamp the
resulting \(4s^2\) to \([0,1]\) *with a recorded clamp counter*.
Never fabricate a variance target from one trajectory. For exact
chance enumeration, calculate mixture mean and variance with the law
of total variance:

\[
\mu=\sum_i p_i\mu_i,\quad
v=\sum_i p_i v_i+4\sum_i p_i(\mu_i-\mu)^2.
\]

**Initial auxiliary outcome definitions (version: \`aux-v1\`):**

- **Progress:** final deepest progress in the declared *Act 1 Overgrowth*
  episode, mapped monotonically and with fixed episode-end bounds to
  \([0,1]\). Include completion as the upper endpoint; do not normalize
  each training example by its achieved floor. The exact floor/room
  mapping must be published alongside the ruleset and checked against
  generated maps rather than assuming an undocumented length.
- **HP:** player HP fraction \(HP/\max(1,HP_{\max})\) at the **end of
  the next room**, with death assigned zero. Use the HP *at that
  endpoint* and cap to \([0,1]\); if the episode terminates before a
  room boundary, use its terminal HP. This is a separate horizon and
  prediction head from final progress.
- **Success:** a separate Bernoulli probability of *clearing Act 1
  Overgrowth* in the Act-1-only phase. Later experiments must add or
  switch explicitly to full-run victory; never label Act-1 success as
  full-game victory. For binary success \(W\), variance is already
  \(p(1-p)\): **no redundant win-variance head**.

The meanings of progress and HP must be tied to a versioned target
schema. Censored episodes may provide already-observed transitions
but not fictitious final success, final progress, or next-room labels.
Search-side true player chance distributions must be distinguished from
model uncertainty caused by few samples. Do not reward risk merely
because it is intrinsically random.

### Learning objective and stability controls

Use a shared encoder, variable-legal-action policy head, primary success
head, and two pairs of bounded mean/dispersion heads: progress and HP.
Keep the existing tiny model (256 hashed input features, 32 hidden
units) as the first benchmark. Export a new **v3 model format**; keep
v1/v2 loading intact and reject absent/untrained heads if requested
as search cutoffs.

Initial loss (weights are configurable, not hidden constants):

\[
\mathcal L_k=
  \mathcal L_{\mathrm{search-policy}}
 +\lambda_w\mathcal L_{\mathrm{success-BCE}}
 +\lambda_a(k)\bigl(
    \mathcal L_{\mathrm{progress-mean}}
   +\mathcal L_{\mathrm{HP-mean}}\bigr)
 +\lambda_v(k)\bigl(
    \mathcal L_{\mathrm{progress-var}}
   +\mathcal L_{\mathrm{HP-var}}\bigr).
\]

Use Huber losses for bounded auxiliary means and uniformized variances,
BCE for success, and cross-entropy or KL for the search policy. Masks
remove censored, unavailable, or under-sampled targets. Log gradient
norms and each component separately; cap gradients and reject NaN /
out-of-range output. An optional heteroscedastic likelihood is an
*ablation only*; do not assume a Gaussian return law by default.

Default annealing experiment: \(\lambda_w=1\);
\(\lambda_a(k)=0.15+0.85\exp(-k/5)\);
\(\lambda_v(k)=0.05+0.20\exp(-k/5)\), with \(k\) the completed
self-improvement iteration (not individual SGD step). These are
**starting hyperparameters to test**, not established optimal values.
Maintain a **no-auxiliary** and **mean-only** control. If auxiliary
*reward shaping* is also used to help bootstrap PUCT in early
iterations, anneal its coefficient separately to zero and continue
selecting the best checkpoint by the unshaped success objective.

## 2. Ordered implementation milestones

Work in \`sts2-ai\` unless a small, mechanically testable chance-sampling
primitive truly belongs in \`sts2-emulator\`. Keep the default agents
unchanged until an alternative wins a fair benchmark.

### S0 — Freeze baselines and targets (first)

**Primary files:** \`docs/INFORMATION_POLICY.md\`,
\`docs/EMULATOR_CONTRACT.md\`,
\`src/sts2_ai/evaluation/run.py\`, new versioned experiment config.

- [ ] Freeze a reproducible checkpoint of heuristic, UCT, and neural
  rollout baselines against the pinned emulator and Act-1 goal.
- [ ] Specify \`aux-v1\` progress/HP horizons and bounds and check
  them against Overgrowth map/terminal behavior.
- [ ] Add clearly differentiated result fields:
  \`act1_cleared\`, \`full_game_victory\`, \`censored\`, and
  \`episode_goal_version\`; preserve historical file compatibility.
- [ ] Log exact seed partition, emulator revision, search regime,
  runtime, simulator transitions, and fixed decision caps.

**Done when:** same seed/config reproduces baseline decisions/results,
goal metrics and censoring are not confused, and a pinned CI smoke
verifies both win and defeat labeling.

### S1 — Known-rule chance model and fair search roots (critical gate)

**Primary files:** new \`src/sts2_ai/emulator/chance.py\` interface,
\`src/sts2_ai/emulator/protocol.py\`,
\`jsonl_backend.py\`; possibly a minimal new emulator JSONL operation.

- [ ] Inventory stochastic mechanics of the *pinned* Silent Overgrowth
  episode; classify each as visible, hidden-but-inferable, or hidden,
  and record its correct conditional probability law.
- [ ] Implement sampling of independent, **observation/history-consistent**
  continuations that does *not* reuse the live state's hidden RNG.
  Respect deck-without-replacement and any correlated RNG streams.
- [ ] Enumerate chance outcomes when the support is small; otherwise
  use batched samples. Begin with 8 or 16 samples per uncertain root,
  increase adaptively for consequential/high-uncertainty choices.
- [ ] Support **common random numbers** for fair paired-action
  comparisons only where the coupling preserves both actions'
  correct marginals. Keep search RNG independent from game RNG.
- [ ] If consistent hidden-state sampling is not yet supported by the
  emulator, expose an explicit blocked capability/error. Do **not**
  label exact-state forks with new seeds as a fair implementation
  until conditional correctness is demonstrated.
- [ ] Keep existing \`oracle-exact\` unchanged as a diagnostic only.

**Done when:** empirical branch frequencies match analytically known
probabilities within a specified binomial/multinomial test tolerance
(e.g. 10,000 deterministic-test samples for small supports); identical
visible histories but different hidden seeds give the same chance law
up to sampling error; no action can detect the live hidden seed; and
chance branches obey full emulator mechanics.

### S2 — PUCT and stochastic backup

**Primary files:** new \`src/sts2_ai/search/puct.py\` and chance-search
tests; reuse \`SearchBudget\` / \`SearchResult\`.

- [ ] Implement legal-action masked priors \(P(a\mid h)\), edge visits,
  means, and the single-player PUCT rule
  \[
    a=\arg\max_a\{Q(h,a)+c_{\rm puct}P(a\mid h)
           \sqrt{N(h)}/(1+N(h,a))\}.
  \]
  No two-player sign alternation.
- [ ] Make stochastic branching explicit: one public history/belief
  must aggregate different possible hidden continuations; do **not**
  merge future outcomes solely by the exact hashes of a single
  sampled seed.
- [ ] Back up primary success and auxiliary means/variances separately.
  For sampled outcomes use stable Welford statistics; for enumerated
  branches use exact probability weights and total variance.
- [ ] Initially use **expected primary success** for action selection;
  expose optional risk-sensitive scoring only behind an experiment
  flag. A larger variance does *not* make a bad action good.
- [ ] Allow small batches, reusable neural state embeddings, and
  inference caching by fair information key and model version;
  profile wall time per decision and transitions per decision.

**Done when:** toy Bernoulli-chance trees have exact known mean and
uniformized variance recovered within Monte Carlo confidence intervals;
PUCT converges toward the known better arm as budget increases;
deterministic toy trees match a trusted dynamic-programming solution;
no private-state lookup is used in fair mode.

### S3 — Tiny v3 neural model and multi-outcome data

**Primary files:** \`src/sts2_ai/models/neural.py\`,
\`src/sts2_ai/training/neural.py\`, new self-play/target schema and
associated tests.

- [ ] Keep the compact shared encoder/action features for the first
  iteration; add Bernoulli success and progress/HP
  sigmoid-mean/sigmoid-dispersion heads.
- [ ] Train direct \(v=4\operatorname{Var}(Y)\) **targets** from multiple
  independent fair sampled outcomes per information set; use the
  feasibility-coupled prediction above. Store sample count and
  sampling-policy identifier.
- [ ] Keep independent train/validation/evaluation **run-seed
  groups** and fair history/state deduplication. Keep exact-state
  labels separate from sampled fair labels.
- [ ] Preserve existing model format readers, refuse untrained
  value heads when used as cutoffs, and test portable stdlib inference
  against optional PyTorch within numerical tolerance.
- [ ] Collect and report Brier score / log loss for success,
  MAE/RMSE for auxiliary means, variance-target error and bin-wise
  dispersion calibration. Report sample size and censoring rate.

**Done when:** synthetic bounded distributions (constant, Bernoulli,
uniform, two-point skew) have analytically checked normalized targets;
head outputs always satisfy feasibility bounds; inference is
deterministic and finite; the additional heads have measured negligible
parameter overhead and a reported actual CPU/runtime overhead.

### S4 — Teacher-free iterative learning

**Primary files:** new \`src/sts2_ai/training/self_improve.py\`,
\`experiments/\` manifest, opt-in CLI command and GitHub workflow.

- [ ] Start from a randomly initialized network; allow zero expert
  demonstrations. Use the current checkpoint to generate stochastic
  PUCT policies and episode outcomes; train a **new** checkpoint.
  Never mutate the acting checkpoint inside the data-collection batch.
- [ ] Record full terminal successes/failures when available, policy
  targets, auxiliary multi-sample labels, model/search versions,
  chance-law version, and source seeds. Do not label truncated
  episodes as defeats.
- [ ] Use temperature-controlled search targets; measure visit
  entropy and JS divergence between independent equal-budget
  searches and between increasing budgets. Include action-value
  sensitivity and regret against an **independently sampled**
  reference when affordable.
- [ ] Start a small reproducible CPU pilot: 4 learning iterations,
  128 training seeds/iteration, held-out evaluation seeds not seen
  during training, PUCT budgets 16 and 64 (then 128 if throughput
  permits), chance samples 8 and 16, fixed model size 256/32.
  Record throughput before increasing scale.
- [ ] Compare *no auxiliary*, *mean-only*, and *mean + uniformized
  variance* using the same generation/training budgets and seeds.
  Compare no annealing versus scheduled annealing only after
  the basic loop works. This ablation is mandatory: auxiliary
  heads should earn their complexity by faster learning.

**Done when:** a single command produces four fully versioned
checkpoints and manifests; replay is deterministic given full research
seeds; the evaluation does not contain training seeds; checkpoints
have independently measured gameplay and compute results; a failure
to improve is reported as such, without automatic threshold tuning
on held-out seeds.

### S5 — Strength and compute gate before promotion

- [ ] Run *uncapped* Act-1 episodes on **at least 100 previously
  untrained seeds**; add more seeds if rare clears make intervals too
  wide. Report clear rate, defeat/censor count, deepest floor, HP,
  compute time, simulator transitions, and peak memory.
- [ ] Paired-seed comparisons: heuristic, oracle UCT (clearly labeled
  unfair), **fair** PUCT without learned heads, neural PUCT mean-only,
  neural PUCT mean+variance. Distinguish *matched simulations* from
  *matched wall-clock budgets*.
- [ ] Publish paired uncertainty intervals for clear-rate difference;
  if there are too few clears, progress and survival remain
  **exploratory proxies**, not proof of win-rate improvement.
- [ ] Add robustness tests on 0.5x/2x chance-sampling budgets and
  run-seed changes, plus sanity checks on the tail of bad outcomes.
- [ ] Promote a checkpoint/agent to default only if its fixed-objective
  held-out performance justifies its added compute. Otherwise retain
  the heuristic/default model and iterate on fair chance sampling,
  coverage, and search quality first.

## 3. Practical performance and memory guardrails

Do not assume that extra variance outputs cost substantial memory:
each auxiliary scalar head adds only \(O(H)\) parameters for hidden
width \(H\). Two extra *mean/dispersion* pairs add approximately
\(4H+4\) linear-head parameters, excluding shared features, versus
\(O(KH)\) for \(K\)-quantile outputs.

**However, fair chance-branching is the real expensive change.**
Measure branch count, live emulator handles, per-state network
inference, and peak search memory before raising sampling depth.
Do not materialize all possible hidden worlds or retain full outcome
distributions by default. Maintain per-edge visit count, running mean,
Welford \`M2\`, and sample count (\(O(1)\) statistics per edge).
Bound handle pools, release unused states, use existing JSONL batches,
and cache only player-legitimate inference input.

Initial benchmarking guardrails (diagnostic, not hard failures):
hold training network width at 32, test 8 vs 16 chance samples, and
report throughput/CPU + RAM before adding any network capacity.
Set a baseline for mean-only versus mean+variance within the same
implementation to isolate the overhead of the variance heads.

## 4. Experiment manifest and version keys

Required for each dataset/checkpoint/run:

- emulator commit, game/ruleset/episode variant, observation schema,
  information-policy ID, and **public-history schema/version**;
- search algorithm version, PUCT constants, search budget and runtime
  budget, chance sampling/enumeration version and search-side RNG seed;
- model checkpoint ID, head schema version, reward/aux target version,
  outcome censoring, annealing iteration and coefficients;
- root run-seed provenance, training/validation/test partition IDs,
  stochastic label sample count, and optional reference-policy ID.

Any change of outcome bounds, history conditioning, chance-law model,
or variance normalization requires a new target/schema version.
Legacy oracle data may be examined but must not silently populate the
fair self-improvement training set.

## 5. Immediate first PR / implementation slice

**Implement S0 plus a minimal S1 proof of concept**, not everything at
once. In order:

1. Record \`aux-v1\` target normalization and add numerical tests for
   bounded mean/variance, stable Welford updates, chance-mixture
   variance, near-constant outcomes, and near-boundary predictions.
2. Add a read-only stochastic-law contract + one independently testable
   chance mechanic (e.g. a public known-card draw distribution);
   establish whether the pinned emulator can sample consistent hidden
   branches without a new operation.
3. Add a seed-blind fairness regression: two true hidden states with
   the same public history induce equal **action probability laws**,
   not necessarily identical realized samples.
4. Then implement fair PUCT behind an opt-in experiment flag, with
   oracle UCT and heuristic left unchanged.

Only after that boundary is correct should we spend simulation budgets
training the uniformized-variance heads and the four-iteration loop.

## 6. Related documents and current evidence

- [Information discipline](INFORMATION_POLICY.md)
- [Emulator consumer contract](EMULATOR_CONTRACT.md)
- [Existing small neural model](NEURAL_POLICY_VALUE_PROTOTYPE.md)
- [Teacher-quality experiments](TEACHER_QUALITY_EXPERIMENT.md)
- [AI baseline roadmap](AI_PROTOTYPE_ROADMAP.md)

The latest **six-seed Q-consensus** experiment succeeded as a workflow
but did not establish better playing strength:
https://github.com/nn971/sts2-ai/actions/runs/37758041017 .
It accepted **22/131** exactly matched roots, and its neural rollout
still had no full-run victories among six 64-decision-capped runs.
Its mean frontier difference vs handwritten MCTS-16 was roughly
\(-0.105\), with the neural model ahead on five seeds but worse on
one; that was not a statistically supported improvement and used
approximately 2.8x the time/run. This is a reason to measure
self-improvement from actual stochastic experience rather than
making agreement-filtered oracle imitation a prerequisite.
