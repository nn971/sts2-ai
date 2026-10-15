# sts2-ai — repository instructions for development agents

This file is project policy for AI assistants, coding agents and reviewers.

## Mandatory research evidence discipline

Read [docs/EXPERIMENT_EVIDENCE_STANDARD.md](docs/EXPERIMENT_EVIDENCE_STANDARD.md)
before designing training experiments, interpreting benchmarks, updating
feature statuses or recommending promotion.

**Do not judge a new model, trainer, sampling method, search method or Nature
feature from a 20-round run, a small pilot or any similarly short experiment.**
Such a run is an **integration smoke**, showing only that the pipeline executes
and its local invariants appear correct. Apparent positive *or negative*
numbers are diagnostics, never a verdict on feature quality.

When asked whether a feature works or is better, require substantive
training-curve coverage, a mature matched baseline, meaningful budget and
replication, an untouched source-run-disjoint native-RNG holdout, and
uncertainty across independently trained models as well as evaluation
source runs. **At least five independent training seeds** is an initial
decision-grade target, not a substitute for power analysis or longer
training when learning remains nonstationary. Use the project's preferred
**15 rollout workers** for new full-machine experiments and compare compute
fairly (simulations, optimizer steps and wall-time). Never infer
between-training-seed confidence from bootstrap resampling of episodes
produced by one model.

Tag reports **contract/unit**, **smoke/integration**, **diagnostic pilot**,
**decision-grade**, or **promotion**, as defined in the evidence standard.
Make synthetic experiments visibly synthetic. Report outcomes and uncertainty
without concluding that Nature or another new feature is effective/ineffective
from short runs.

For Nature specifically, distinguish (a) whether the opponent learns to
make a fixed player face harder outcomes, (b) whether the Nature-trained
player improves under *native* RNG for equal compute, and (c) whether
any reference probability law has been independently audited. Only (b)
is the primary ordinary-game strength comparison. The 20-round three-seed
toy Nature test in PR #78 is **smoke only**.

## Repository and emulator boundaries

- Preserve the frozen Silent game baseline and existing v15 checkpoints.
- Keep new methods opt-in; use explicit versioned RNG/Nature contracts.
- The emulator owns game mechanics, source legality, stochastic
  continuation and hidden deck constraints; the AI owns player/Nature
  policies and research objectives.
- Never expose hidden emulator data to the player observation.
- Preserve separate native committed shuffles and Nature deferred-card
  identity representations; do not double-count both as independent draws.
- Explicitly label unsupported mechanics, censored trajectories and
  unverified source chance probabilities.

When changing experiments or interpreting empirical results, cite
the evidence standard in the relevant PR/report and keep it visible in
summary conclusions.
