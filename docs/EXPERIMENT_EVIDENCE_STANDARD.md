# Experiment evidence standard — project-wide research rule

**Applies to:** all `sts2-ai` model, training, search, sampling, emulator-interface,
and Nature changes. **Status:** project policy, 2026-10-16. This document
governs the wording of future PRs, experiment reports and assistant summaries.

## Training is independent of original-game fidelity (binding policy)

**Never make installed-game fidelity, exact native RNG reproduction,
Fidelity Inspector issue #122, N5 source-law calibration, or broad S3/S4
equivalence prerequisites for agent training.** A declared-law emulator
experiment may proceed when the chosen mechanical subset compiles, plays to
real terminal states, uses legal supported outcomes and fair observations,
fails closed on unsupported stochastic sites, and replays exactly.

For all training and decision-grade *emulator-relative* claims below,
"native-RNG" or "ordinary/reference-RNG" evaluation means held-out trials
using the **emulator's engine-owned reference stochastic policy** (including
`ReferenceNature` for declared Chance Laws); it does not imply verified
agreement with the installed game's RNG. Keep uncalibrated laws labeled
`UNVERIFIED` as probability provenance; that status is **not a training
failure or stop condition**. Internal support/weight consistency is still
required. Original-game accuracy and unbiased native-game importance-weighting
are *separate, later claims* requiring appropriate source evidence.

The evidence-level rules below (multi-seed evaluation, meaningful budgets,
holdout separation, uncertainty) remain mandatory **within the chosen
declared emulator environment**, irrespective of its original-game
fidelity status.

## Core rule: an integration smoke test is not an assessment of a new feature

**Never judge, accept, reject, or promote a training or algorithmic feature
from a 20-round experiment or any comparably short smoke run.**
A new component that passes tests or appears to improve a handful of rounds
has shown that its code can run, not that its learning or strategic effect is
established. Likewise, an apparent regression in a small run is a signal to
investigate, not grounds to declare the idea ineffective.

The project is large, highly stochastic and nonstationary. Training behavior
may depend on how the player and opponent evolve, on encounter rarity, on
training seed, and on the distribution of combat states. Evaluation must
cover these sources of uncertainty.

## Evidence levels and permitted claims

| Level | Typical work | Permitted claim | Claims not permitted |
| --- | --- | --- | --- |
| **0. Contract/unit** | Finite oracle, exact replay, CI, lint, short synthetic test | Interfaces, algebra, test invariants or replay work in the tested setting | Feature improves/worsens real gameplay |
| **1. Smoke/integration** | 1–20 rounds or similarly small compute; a few supported combats | Mechanisms execute; logs/checkpoints/metrics are well formed; obvious bugs and blockers identified | Training convergence; feature utility; promotion/rejection |
| **2. Diagnostic pilot** | Expanded multi-seed runs, learning curves and representative engine-owned reference-policy holdout | Preliminary signals and plausible hypotheses explicitly tagged *provisional* | Definitive improvement or generalization |
| **3. Decision-grade experiment** | Predeclared substantial training budget, independent replicas, matched control, confidence analysis and untouched evaluation | Supported comparison for the declared environment, scope and compute | Claims beyond tested game fidelity, mechanics or heldout distribution |
| **4. Promotion** | Repeated decision-grade result, fresh independent holdout and engineering/regression acceptance | Deploy/promote the feature within the validated scope | Blanket performance guarantees |

The scale distinguishing levels 1–3 must be selected from the current
baseline's convergence rate, computational cost and observable learning
curve. **There is no magic threshold such as “20”, “40” or “100” rounds**
that makes an experiment decisive. In particular, a 20-round synthetic
Nature experiment is **Level 1**, regardless of how consistent its apparent
result looks. Small-run confidence intervals or a 3-seed smoke are not
substitutes for substantial training.

## Required design for feature-value claims

Before calling a new training feature better or worse:

1. **Predeclare the comparison.** State the primary metric (normally
   engine-owned reference-RNG player survival/win and exit HP), chosen environment/game
   revision, source-run-disjoint evaluation corpus, training checkpoints,
   reward objective, controlled variables, and minimum detectable effect
   or practical target. Separate source-authentic game evidence from
   synthetic-toy and emulator-interface tests.
2. **Train at meaningful scale.** Use a substantial budget justified
   against a mature existing baseline and enough checkpoints to inspect
   learning-curve stability, later-stage behavior, and diminishing returns.
   Expand or revisit the budget if curves are still changing materially.
   Record full training history; do not select only the best intermediate
   checkpoint without independent validation.
3. **Replicate independent training seeds.** Use **at least five**
   independent player/Nature training seeds as an initial decision-grade
   target; increase replication or perform a prospective power analysis
   when model-to-model uncertainty is large. Three-seed experiments are
   still *diagnostic* by default. Independent evaluation encounters from
   one checkpoint do not replace independent retraining.
4. **Use a matched comparator.** Hold emulator revision, player
   architecture/initialization, snapshot corpus, optimizer settings,
   environment rules and major training-budget choices fixed. For Nature
   explicitly account for *both* opponent rollout cost and player
   optimization opportunity cost; report total simulator episodes,
   sampled decisions, optimizer steps and wall/GPU time. Document
   unavoidable asymmetries rather than hiding them.
5. **Evaluate only under the intended target distribution.** For player
   strength, use independent **emulator reference RNG**, with frozen public-only
   player observations and a fresh source-run-disjoint holdout. Report
   weak/normal/elite/boss and relevant Act/floor/deck stages rather than
   relying only on an overall aggregate. Keep training/development/test
   boundaries and supported-feature coverage explicit.
6. **Quantify both uncertainties.** Report per-training-seed differences,
   means/medians and effect sizes, confidence/credible intervals suitable
   for training-seed variability, and within-seed source-run-clustered
   evaluation uncertainty. A bootstrap over combat snapshots alone
   estimates **evaluation sampling variation**, not the variance across
   separately trained models. Describe correlated or paired comparisons.
7. **Check correctness and leakage.** Audit replay, legality, chance
   support, hidden-player-information boundaries, censoring, checkpoint
   provenance, chance-law calibration status and unsupported mechanics.
   Treat truncated episodes as censored, never defeats; label toy/declared-law results as emulator-relative and
   validate original-game accuracy as a separate claim.
8. **Report uncertainty and failure honestly.** Include negative and
   inconclusive outcomes, seed variance, confidence intervals, benchmark
   scope, known emulator fidelity gaps, and what remains untested.
   If training diverges or learning curves are inconclusive, extend the
   diagnostic or leave the result explicitly unresolved.

**Default full-machine parallelism for new experiments: 15 rollout
workers**, with 1/15 worker determinism checks where relevant. Worker
count measures throughput; it does not substitute for the independent
training seeds required above.

## Nature-specific rule

Three questions must be reported **separately**:

- **Opponent learnability:** can learned Nature lower a frozen player's
  return compared with an untrained Nature proposal under the same
  mechanics? This tests adversarial learning, *not player improvement*.
- **Player benefit:** does Nature-trained player improve relative to
  **equal-compute reference-trained** player on a fresh, independent,
  **engine-owned reference-RNG** holdout? This is the primary strength claim.
- **Probability validity (if applicable):** does Nature have an audited
  conditional reference law `p` and a full-support `q`, permitting
  a justified importance-weighted original-game estimator? Without
  audited `p`, label training as **unweighted adversarial/stress**.
  Do not combine the adversarial and source-expectation objectives.

**Nature pilot PR #78:** the 20-round, three-seed synthetic comparison
is a *smoke/diagnostic demonstration* only. Its opponent difficulty and
native-toy A/B numbers **must never be presented as a verdict** on the
value of Nature for Slay the Spire 2, even if a single number appears
positive or negative. The first real-combat experiment begins only
after the external playable semantic intent/draw interface and verified
terminal returns are available.

## Reporting checklist

Every future training PR/experiment summary should identify its
**evidence level**, exact environment/revision and training budget,
independent training-seed count, source-run-disjoint holdout, primary
reference-evaluation metric, uncertainty calculation, compute accounting,
CI/unit-test status, and any supported-mechanics or fidelity limitations.

If evidence is Level 0/1, use language such as **“smoke passed”, “API
wired”, “optimizer direction tested”, “preliminary observation”**.
Reserve **“better”, “worse”, “effective”, “ineffective”, “promote” or
“reject”** as model-quality judgments for decision-grade experiments.
A small-run directional numerical comparison can be reported precisely,
but **never converted into a judgment of feature quality**.
