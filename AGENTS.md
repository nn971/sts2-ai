# sts2-ai — repository instructions for development agents

## Current H15 M18 throughput decision (2026-10-31)

The user rejected manual search-root allocation and approved PUCT
at EVERY encodable multi-action public combat decision, initially
32 simulations, full-depth expansion with strict depth 8 and neural
value leaves. This is a new M18 research branch
`research/h15-m18-every-decision-value-puct-20261031`,
superseding M17's **search schedule** while preserving M17 historical
training/checkpoints, beta=0.5 defeat-margin reward, existing
root-conditioned approximate teacher, emulator-owned ReferenceNature,
and policy/value architecture. See the M18 section of the canonical
`docs/H15_SEARCH_TRAINING_PLAN.md`.

The one-round 15-worker throughput smoke completed (15 combats, 643
searched roots, 500.43 seconds, no heldout evaluation). The user
subsequently approved an 80-round M18 run initialized from the
unchanged M17 R10 checkpoint, with 128 combats/round, 64-case heldout
evaluations every 10 rounds, 15 workers, 32 PUCT simulations per root,
depth 8, value leaves and beta=0.5 defeat reward. Use the separate
M18 long-run seed prefix/output root in the canonical plan. Treat the
single training run as research evidence; do not claim model promotion
without replicated evaluations. Actual R10
checkpoint and pinned H8 emulator binaries are available on the
user's machine, not in GitHub CI; do not claim they were executed
without a local run receipt. Nested player-selection menus remain
unsupported for direct search distillation under the current action
codec. Avoid additional replay/optimizer revisions until after the
throughput experiment to keep the M18 search-coverage variable isolated.

## Current H15 M17 execution decision (2026-10-31)

The user approved a **new** training experiment following M16's
unimproved round-20 evaluation: beta=0.5 applied on the **defeat**
branch of the terminal H9 objective (victory HP preservation unchanged),
and **more frequent/deeper PUCT expansion**. See the M17 section in
`docs/H15_SEARCH_TRAINING_PLAN.md`. Keep earlier M16 artifacts,
reward-version identities, frozen source checkpoints, and already
running experiments untouched. M17 lives on the separate
`research/h15-m17-victory-margin-puct-20261031` research branch.

All new M17 data must carry explicit terminal reward version and
deep-search version/provenance. Full-depth root-conditioned H8 PUCT
is an **approximate privileged teacher**, not a fair posterior;
network observations remain public. No new Chance Law or stochastic
provider is authorized. Evaluation should compare win rates and
surviving HP on matched fresh ReferenceNature holdouts, since mean
rewards under different beta versions are not directly comparable.
This M17 decision updates the research configuration without changing
the accepted overall goal of a complete search-guided policy/value
feedback cycle, the no-fidelity-gate rule, or evidence standards.


This file is project policy for AI assistants, coding agents and reviewers.

## Mandatory H15 implementation priority and conflict disclosure (2026-10-30)

**Current accepted H15 execution plan:** [docs/H15_SEARCH_TRAINING_PLAN.md](docs/H15_SEARCH_TRAINING_PLAN.md). Read it **before any H15 code change, training run, task recommendation, PR description or roadmap edit**. The next deliverable is one complete **existing ReferenceNature / Chance Law -> neural PUCT -> real visit targets -> policy/value optimizer -> updated search** cycle on a supported A10 combat subset. Its integration smoke precedes a substantive held-out comparison; no further H10/PPO-only rounds may be described as search training.

**Do not make posterior recovery a general prerequisite.** Reuse the existing emulator stochastic providers, Nature-mode unresolved deck constraints and semantic continuations. Root-conditioned exact-state UCT/PUCT may be used only with explicit leakage/provenance labels. Calibrated native RNG, true full-history posterior and installed-game fidelity are separate investigations, not training gates. Any proposed new stochastic provider requires an explicit, concrete deficiency of all existing owners and the user's decision.

**Mandatory conflict response:** If a milestone log, old roadmap, source guard/comment, test, or experimental constraint disagrees with this plan, **tell the user before shifting work**. Quote the conflicting location/claim, distinguish actual mechanics blockers from optional fairness/fidelity projects, propose the smallest plan-preserving route, and obtain an explicit decision before materially changing implementation goals, adding acceptance gates or launching long alternative experiments. When a plan is revised, update this file, the canonical plan, and documentation entry points in the same change. Silence or implicit scope drift is a process failure.

**Document precedence for H15:** current H15 plan + this file > emulator canonical stochastic ownership (for mechanics/providers) > H15 experimental logs and historical Nature/PUCT/prototype roadmaps (evidence only). Do not promote historical "next steps" into active gates.

## Mandatory separation: training versus original-game fidelity

**Native-game fidelity checks must never block agent training.** This is a
standing, cross-repository project decision. A playable, supported emulator
subset with legal actions, fair observations, internally consistent declared
Chance Laws, deterministic replay and explicit rejection of unsupported
stochastic sites is sufficient to begin training and reference-law evaluation.
Keep a real-emulator build/terminal-combat gate; do not turn it into a native
parity gate.

- `UNVERIFIED` denotes that the emulator's conditional Chance Law has not been
  calibrated against the installed game. It **does not block training**.
  Preserve the label and describe findings as emulator-relative.
- The **reference/ordinary RNG holdout** means an independent held-out
  evaluation using an engine-owned reference policy (`ReferenceNature` or
  another declared ordinary emulator RNG policy). It does **not** require
  reproduction of the installed game's RNG distribution.
- Original-game RNG stream/seed/cursor/hook parity, installed-game Fidelity
  Inspector work (emulator #122), source-law calibration/N5 and exhaustive
  S3/S4 or all-Act cross-provider equivalence are **independent projects**.
  Require original-game evidence only for claims about original-game accuracy
  or original-game-unbiased likelihoods/importance weights.
- In training plans and status updates, list emulator-internal mechanical and
  statistical blockers separately from independent native-fidelity work.
  Never label fidelity or N5 as a training acceptance blocker.

Related: [emulator #143](https://github.com/nn971/sts2-emulator/issues/143),
[AI Nature PR #82](https://github.com/nn971/sts2-ai/pull/82),
[emulator #122](https://github.com/nn971/sts2-emulator/issues/122).

## Mandatory stochastic architecture reuse (H15 retrospective)

**Read the emulator's canonical**
[`UNIFIED_STOCHASTIC_ARCHITECTURE.md`](https://github.com/nn971/sts2-emulator/blob/research/h15-experimental-search-20261030/docs/UNIFIED_STOCHASTIC_ARCHITECTURE.md)
**and**
[`ADR 0005`](https://github.com/nn971/sts2-emulator/blob/research/h15-experimental-search-20261030/docs/adr/0005-h15-reference-nature-reuse-retrospective.md)
**before proposing or coding ANY new RNG, Nature, Chance Law, stochastic
continuation, or H15 fair-search feature.** The previous implementation
mistake was reasoning from recent H15 milestones alone, overlooking
`ReferenceNatureV111`, atomic reference sessions and the already-written
fixed-fixture `H8ReferenceNaturePosterior`; it produced a redundant
Python chance sampler and incorrect descriptions of what was missing.

- First inspect `StochasticSiteRegistryV111`, `ChanceLawV111`,
  `ReferenceNatureV111`, `H8LocalReferenceNatureV1`, existing
  `ReferenceNature*SessionV111` owners, and
  `sts2_ai.emulator.h15_h8_reference`.
- Reuse the emulator's reference sampler and mechanics continuation.
  Python may implement **learned proposal** sampling but must not
  reimplement reference-law roulette under an H15-only name.
- Always distinguish source legality, `SOURCE_VERIFIED` local
  probabilities, a posterior under a **declared fixture prior**, and a
  native full-run posterior. Do not refer to native-structure
  **prototype-combat-RNG** corpora as native-RNG calibrated.
- The existing M13 experimental native-entry candidate paths remain `UNVERIFIED`
  and have produced no certified fair PUCT targets. Keep their provenance
  intact. This statement is **not** a prohibition on new, explicitly
  root-conditioned/ReferenceNature-model search-training experiments.
  Preserve pinned PPO artifact identities; require focused provider/cursor/
  event tests when the stochastic boundary actually changes.
- Before adding a new system, explicitly identify why the existing
  provider or session API cannot own the operation and document that
  gap. Append new capabilities to the current-state crosswalk rather
  than copying historical progress text into new architecture.

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
replication, an untouched source-run-disjoint engine-owned reference-RNG holdout, and
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
player improves under *engine-owned reference* RNG for equal compute, and (c) whether
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
