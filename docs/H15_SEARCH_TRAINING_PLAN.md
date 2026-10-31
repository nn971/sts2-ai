# H15 — canonical combat search-training execution plan

**Decision date:** 2026-10-30. **Status:** agreed current H15 implementation priority.
**Authority:** this document governs H15 training priorities and acceptance gates. The emulator's [unified stochastic architecture](https://github.com/nn971/sts2-emulator/blob/research/h15-experimental-search-20261030/docs/UNIFIED_STOCHASTIC_ARCHITECTURE.md) governs stochastic ownership and mechanics. The repository [AGENTS.md](../AGENTS.md) governs change control. Earlier H15 milestone logs, Nature designs, PUCT variance roadmaps and PPO runbooks are **historical evidence, not competing instructions**.

## Objective and first deliverable

Implement a **real emulator-backed combat search → search-policy targets → policy/value optimizer update → updated search** loop, initially for a supported Silent Ascension 10 combat curriculum. A single completed combat providing valid search visits and a verified gradient step is the first *integration* milestone; long-run improvement must subsequently be measured with the [experiment evidence standard](EXPERIMENT_EVIDENCE_STANDARD.md). Do not substitute ordinary PPO rounds for this search-guided experiment.

The intended eventual hierarchy remains: shared policy/value representation; combat policy/value with neural-guided search; an outcome interface for macro decisions; and a macro policy/value head. The combat-search learning loop comes **first**. Do not revive the abandoned standalone combat outcome predictor as a training prerequisite.

## Existing, reusable machinery

- Emulator deterministic combat mechanics, legal player decisions, fork/replay, and the *existing* typed Chance Laws.
- **Existing** emulator-owned stochastic providers: Native RNG for native play; Nature/ReferenceNature for controlled unresolved stochastic choices. Reuse their current state/cursor ownership and semantics.
- H4's root-conditioned stochastic UCT as the *execution precedent*: \`src/sts2_ai/nature/h4_mcts.py\` with \`DeclaredLaw\`/\`SeededLawSampler\`; this was a real search baseline with limited horizon and heuristic cutoff, **not** an information-set-fair teacher.
- H8's in-emulator reference sampling and stateful deck mechanics (including \`SemanticDeckStateV111\`), the H9/H10/H15 policy/value checkpoint and encoding code, H15's \`search/combat_puct.py\`, visit-target serializer and distillation optimizer.

**No new H15-only RNG, Nature, Chance Law, or hidden-state posterior provider is authorized by this plan.** A missing adapter or supported mechanics continuation should be connected to its existing owner. First inspect and reuse that owner; make any actual gap explicit before proposing a new subsystem.

## State semantics and stochastic correctness

The *public policy observation* includes player-known state, e.g. the draw-pile count and known cards, and never the private order or RNG cursor. The *complete simulation state* must preserve hidden realizations/constraints for replay; that is compatible with public-only player inference.

The intended ownership is \`SimulationState { Mechanics, Stochastic }\`: mechanics owns card-zone **membership**; the provider owns private **order**, deferred/known deck slots, and stochastic cursors. The current \`CombatState.Mechanics.CardZones.DrawPile\` ordered physical array duplicates ordering for compatibility; \`CombatState.Stochastic.SemanticDeck\` is the provider-side state. Follow the existing \`SemanticDrawPileAdapterV111\` synchronization contract; do not treat the compatibility array as information available to the search player.

Reference search must sample unresolved outcomes under declared Chance Laws **with persistent provider state**, preserving depletion, known placements, correlated draws and other committed constraints across the simulation. At native-exact entries, a precommitted private card order cannot be magically made unresolved by merely changing the draw RNG. Use an existing supported Nature initialization/continuation path; for the first vertical slice it is acceptable to begin with a supported Nature-mode combat scenario instead of a replayed native-exact snapshot.

A **root-conditioned** search that forks an exact hidden simulator state is permitted as an explicitly labelled *diagnostic/initial approximate training teacher* with measured possible privileged-information bias. Do not claim that its targets are player-information-set-fair or native-game unbiased. For the primary reference-law training evaluation, compare under the declared emulator distribution with source-run-disjoint/seed-disjoint holdouts. Exact native-run public-history posterior recovery, installed-game RNG parity and probability calibration are **independent later refinements**, not H15 training gates.

## Ordered implementation

1. **One playable Nature-mode combat:** pick a source-supported A10 slice that already executes terminal combat using existing mechanics and ReferenceNature. Exercise legal player actions, mixed deterministic/chance steps, hidden deck consistency, terminal H9 non-minion-HP reward and a complete action/receipt journal. Report supported/unsupported action and encounter coverage; do not silently skip invalid continuations.
2. **Attach H15 neural PUCT:** use the existing network to provide legal-action priors and leaf values. Use ReferenceNature to advance chance boundaries, keeping its hidden state consistent throughout each simulation. Use a bounded simulations/depth budget; avoid mandatory full-combat rollout for each search leaf. H4's heuristic and exact-state oracle variants remain labelled baselines.
3. **Collect real targets:** root policy target = normalized PUCT visit counts at **actually executed** player decisions; critic supervision = real terminal H9 encounter reward (normalized as the existing value head expects). Record emulator/provider revision, chance-model ID/probability status, root mode (Nature vs root-conditioned exact), observation/action schema, teacher checkpoint hash, seed, simulation budget, completed/censored/unsupported counts and complete-search provenance.
4. **One optimizer update:** use the existing H15 policy/value loss or minimal compatible changes. Verify the policy fits visit distributions, value fits realized outcomes, and a new checkpoint can immediately guide a second search. This completed feedback cycle, not another isolated test or extra PPO run, is the next deliverable.
5. **Training and evaluation:** repeat search/data/optimization with refreshed checkpoints, using the preferred 15 workers for substantive runs. Compare (a) frozen PPO baseline, (b) raw learned policy, (c) search-guided policy, and (d) model after distillation, on held-out independent emulator ReferenceNature combats with matched source distribution and compute accounting. Report win rate, exit HP, decisions, completed combats/s, search overhead, training curves, confidence intervals and independent training-seed uncertainty. Small runs are integration smoke only; use substantive longer replicated runs for strength judgments.
6. **Extend only observed blockers:** increase combat/mechanics/action-codec coverage based on actual failures in the target curriculum. Once combat learning works, connect the outcome interface and the agreed macro-combat hierarchy. Do not substitute an unrelated predictor or stochastic-fidelity project.

## Explicit exclusions and provenance

- A native full-run hidden-state posterior is **not** a prerequisite to searching in the emulator's declared stochastic game. Formal information-set fairness is a separately labelled target; information leakage in root-conditioned approximations must be disclosed and measured.
- \`UNVERIFIED\` Chance Law provenance denotes uncalibrated source-game probabilities, **not** unusable emulator-relative stochastic training. Never claim original-game win-rate validity without source evidence.
- H15 M13 public-history rejection conditioning and experimental native-entry candidate sessions remain optional diagnostics. They do not gate the first search-training cycle.
- The previously completed 120-round run used ordinary H10 PPO initialized from H9. It generated **no PUCT targets**, and its checkpoint is a baseline, not evidence of search improvement.
- The current \`H15PublicCombatAdapter\`/\`FairContinuationSampler\` capability gate applies to that **specific fair-search path**. It must not be generalized into a project-wide ban on a versioned, clearly labelled ReferenceNature/root-conditioned experimental search loop.

## Mandatory conflict and scope-change procedure

**Before doing work**, read this plan, \`AGENTS.md\`, and the emulator ownership contract. If an older roadmap, a test's guard, a source comment, an implementation limitation, or a new theoretical concern appears to conflict with the plan:

1. **State the conflict explicitly to the user**, identifying the two conflicting claims and their exact source files/locations.
2. Classify it as (a) real gameplay/mechanics correctness failure, (b) missing integration/codec/throughput work, (c) provider fidelity/fairness research, or (d) a documentary/terminological contradiction.
3. Give the simplest plan-preserving resolution and its tradeoff. Ask for a decision **before materially changing goals, imposing a new acceptance gate, or launching a long alternative experiment**.
4. Amend this canonical plan and the relevant repository instructions **together** whenever the decision actually changes. Mark superseded guidance as historical or remove it. Do not quietly change the priority because a test or older document expects a different architecture.

The next named milestone must remain **H15 search-guided combat learning**, until the user explicitly changes it.


## First integrated feedback-cycle implementation (M14; awaiting workstation smoke)

The experimental H15 AI branch now contains
`tools/run_h15_h8_feedback_cycle.py`, a direct composition of the
existing H8 declared-law ReferenceNature collector, H15 policy/value
distillation and renewed H15 PUCT. It implements the **first ordered
training deliverable on a fixed A10 fixture** without changing the pinned
H10 PPO experiment or adding a stochastic provider.

The runner performs, in order:

1. Open the same A10 H8 fixture under **emulator-owned**
   `ReferenceNatureV111.SampleValidated` Chance Laws and collect value-leaf
   PUCT visits. The search uses bounded simulations/tree depth; the
   *actually executed* full fight supplies the terminal H9 HP reward.
2. Decode and verify all saved target rows and the exact declared chance-prior
   ID and source-model SHA. Distill visits via policy cross-entropy and
   realized encounter reward via value regression into a **new** H15 model.
   Compare initial and updated model tensors, rejecting a metadata-only
   checkpoint with zero parameter change.
3. Reconstruct the **identical opening public decision and Nature seed**,
   collect PUCT visits with the updated H15 teacher, check the root's
   observation and legal actions still match, and report both visit policies,
   teacher/checkpoint hashes and actual combat outcomes. The same seed is for
   controlled comparison, not a claim that changed actions share identical
   subsequent hidden outcomes.

This is deliberately scoped to the **fixed-fixture declared-prior model**
(`H8_DECLARED_LAW_MODEL`), with terminal rewards from the real reference
combat, not model-generated pseudo-labels. `UNVERIFIED` probability
calibration against the original game remains a separate project. The
updated value head can guide bounded leaves; the existing
`--leaf-mode terminal` path remains available for older terminal-only
comparisons. A short single-fixture cycle is an **integration smoke**,
not evidence that H15 outperforms H9 or works across arbitrary native entries.

Run on the existing local workstation **after the pinned PPO checkout is
free**, without making another worktree:

```fish
set ai ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
git -C $emu switch research/h15-experimental-search-20261030
git -C $emu pull --ff-only
git -C $ai switch research/h15-experimental-search-20261030
git -C $ai pull --ff-only

dotnet build $emu/src/Sts2Emulator.Cli -c Release
cd $ai
source .venv/bin/activate.fish
python -m pytest -q tests/test_h15_feedback_cycle.py \
    tests/test_h15_native_search_collector.py \
    tests/test_h15_combat_search.py \
    tests/test_h15_h8_reference.py

python -u tools/run_h15_h8_feedback_cycle.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --emulator-dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll \
    --initialize-from-model results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --output-root results/h15-reference-feedback/mawler-a10-seed113-v1 \
    --scenario mawler-a10 --deck silent-starter-v1 \
    --source-nature-seed 113 --simulations 16 \
    --tree-depth 8 --max-searches 1 \
    --epochs 1 --device cuda --execute
```

The runner refuses to overwrite an existing output directory and saves
`before/targets.jsonl`, `before/report.json`, `student.pt`,
`after/targets.jsonl`, `after/report.json`, and
`feedback-cycle.json`. Remove `--execute` to preview the exact commands.

**Status and next step:** remote code is committed; no .NET/CUDA-backed
feedback-cycle result has been generated in the repository environment.
Run the integration smoke first. Repair actual source mechanics/codec
failures if they occur; once this completes, scale to a source-supported
A10 Nature combat curriculum and repeated search-training updates under
disjoint reference seeds. For comparative strength claims, follow the
experiment evidence standard and multiple independent training seeds.


## M14 workstation result and codec-controlled follow-up (2026-10-30)

**Integration smoke succeeded on user-provided reports**, under emulator revision
`a411a19491a93b58ae4fdbc6d1f3345c9fca67cf` and declared fixed-fixture
prior `h15-h8-fixed-fixture-declared-chance-law-v1` (Silent A10 Mawler,
starter deck, source Nature seed 113). The initial and updated root observations
and legal-action menus match exactly.

| Metric | Before H15 optimizer | After H15 optimizer |
| --- | ---: | ---: |
| Completed combat | victory | victory |
| Exit HP | 28 | 32 |
| Realized H9 encounter reward | 1.4 | 1.457142857 |
| Executed combat decisions | 27 | 23 |
| Root search visits | 16 | 16 |
| Search neural-value cutoffs | 16 | 16 |
| Search terminal evaluations | 0 | 0 |
| In-sample policy loss, measured by H15 codec | 2.23785591 | 1.54206824 |
| In-sample value loss | 0.0000005354 | 0.0000374693 |

The optimizer made **one** step, changing 103 tensors (`L2=0.1030645`).
Opening visits changed from `[3,0,5,5,3,0]` to `[5,1,3,3,4,0]`
for (Strike #5, Survivor #12, Defend #9, Defend #6, Strike #2, end
turn). Visit-policy L1 change was 0.5. These are **integration diagnostics**
only: one training root and one realization per policy cannot establish
an improvement or reliable action-value ranking. The value loss grew,
though it remained small in absolute magnitude.

**Identified control defect in the v1 feedback-cycle comparison:** the initial
teacher checkpoint was `sts2-h7-shared-object-transformer-v1` using
`SHARED_OBJECT_SCHEMA`, whereas the post-update checkpoint was an H15
checkpoint using the move-augmented `H15_COMBAT_SCHEMA`. A single gradient
step therefore coincided with a **codec change**. Although both reported
training losses use the H15 codec, the before/after *search* comparison
cannot attribute its policy shift to optimization alone.

**Correction committed for subsequent cycles:** feedback-cycle schema v2
writes `bootstrap-h15.pt` by promoting the H9 checkpoint metadata to
`H15_COMBAT_SCHEMA` while asserting the weights remain byte-for-byte
tensor-identical. **Both** initial and updated PUCT calls now use the H15
codec, and the optimizer receipt is bound to the promoted checkpoint SHA.
The original source checkpoint SHA and bootstrap SHA are reported separately.
The existing v1 files remain valid historical evidence and are **not
retrospectively renamed** to v2. Use a **new output directory** and rerun the
same smoke once to establish the codec-controlled baseline.

A separate narrow H15 experimental ReferenceNature posterior CI check found
that the batch adapter only checked `providerId` in its receipts. The first
batch event now verifies its law ID, event ID, selected outcome index and
outcome ID against the advertised pending Chance Law; later batch events
remain emulator-owned. This change does not alter H8 fixture chance sampling
or the successful M14 execution.

**Next:** after the v2 codec-controlled replay, move directly to a *substantive*
search-guided training run covering multiple supported A10 scenarios/independent
Nature seeds, refreshing PUCT targets after each learner update. Preserve 15
workers, a held-out reference-seed split, compute accounting, and periodic
evaluation against frozen H9 and raw/distilled H15 policies. Choose search
budgets based on observed throughput; do not interpret the original 16-visit
smoke as a search-budget optimum.


## M15 — repeated ReferenceNature search-guided combat training (implementation committed)

The user-supplied **v2 codec-controlled integration** completed under emulator
`a411a19491a93b58ae4fdbc6d1f3345c9fca67cf`, with genuine A10 Mawler
combat rewards. One H15-codec model update changed 103 tensors; terminal exit
HP was 24 before and 35 after, both victories. The root visit distribution
shifted from `[3,8,1,1,3,0]` to `[7,2,0,0,7,0]` (16 visits per
search), while the H15 in-sample policy cross-entropy **rose**
`1.331495→1.673004` and value MSE fell
`0.00077505→0.00019169`. The search target's entropy was
`1.320888`, yielding initial policy KL≈`0.010607` and post-step
KL≈`0.352116`. This near-zero starting KL suggests that this small
search mostly reproduced its existing network prior; the optimizer moved
the policy away from the target. The reports support integration, **not**
improvement, or an attribution of the policy-loss change to any specific
gradient component.

**Next executable tool:** `tools/train_h15_reference_rounds.py` runs
multiple **full, real H8 ReferenceNature A10 combats per round**, collects
genuine PUCT visit distributions at executed root decisions, then invokes
the H15 policy/value distillation optimizer and repeats with the refreshed
checkpoint. Each cohort deterministically assigns explicitly supported
scenario/deck combinations using the existing H2 seed-assignment helpers;
distinct train and heldout source seed namespaces prevent overlap. **No
native-run posterior and no new random sampler were introduced.**

- Default **15 CPU actor workers** each reuse one emulator JSONL process
  and a frozen model; main process trains on CUDA. No need to run 15
  separate GPU models.
- Default **80 rounds**, **128 training combats/round**, **32 PUCT
  simulations/root**, 1 searched root/combat, 1 optimizer epoch with LR
  `3e-6`, 64 heldout combats every ten rounds, source-seed-disjoint.
  These are initial resource defaults, **not experimentally selected
  optimal hyperparameters**. Search targets are rejected if any
  controlled fight cannot complete; unsupported sites fail closed.
- Full per-case outcome, target, chance-prior, source scenario/deck/seed,
  model SHA, numeric optimizer receipt, training policy KL to search and
  entropy, stable per-worker chunk files, and per-round progress files.
- Evaluation compares **frozen H9 greedy**, **updated H15 greedy**, and
  **updated H15 PUCT** under the same heldout H8 ReferenceNature
  case seeds. Fixed seed is a pairing device only: changed player actions
  can change the downstream sequence of chance requests. H9 and H15
  policy codecs also differ; this remains an explicit baseline comparison.
- Completed rounds are immutable; the runner checks emulator DLL SHA,
  source model SHA, code hashes, model/target/optimizer identities and
  saved worker chunk case IDs before continuing a partially completed
  round. An incompatible code/emulator/config change must start a new
  experiment identity. Outputs stay inside `sts2-ai/results/`.

**One cohort integration smoke (separate output directory):**

```fish
set ai ~/projects/sts2-ai
set emu ~/projects/sts2-emulator
cd $ai
source .venv/bin/activate.fish
git pull --ff-only

python -m pytest -q tests/test_h15_feedback_cycle.py \
    tests/test_h15_native_search_collector.py

python -u tools/train_h15_reference_rounds.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --emulator-dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll \
    --initialize-from-model results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --output-root results/h15-reference-m15-cohort-smoke-v1 \
    --rounds 1 --train-combats 15 --eval-combats 15 --eval-every 1 \
    --workers 15 --simulations 16 --max-searches 1 \
    --scenarios mawler-a10,cubex-a10,vine-shambler-a10 \
    --decks silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1 \
    --learner-device cuda --execute
```

This tests collection with **several encounters, complete fights and
checkpoint refresh** once, then evaluates the three actual policies.
An unsupported scenario/deck combination is a concrete mechanics/codec
coverage failure. Report it; use the source-supported narrower pool until
the owner is extended, rather than silently skipping the source cohort.

**Substantive H15 search training (start fresh after the cohort works):**

```fish
python -u tools/train_h15_reference_rounds.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --emulator-dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll \
    --initialize-from-model results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --output-root results/h15-reference-m15-a10-long-v1 \
    --rounds 80 --train-combats 128 --eval-combats 64 --eval-every 10 \
    --workers 15 --simulations 32 --max-searches 1 \
    --scenarios mawler-a10,cubex-a10,vine-shambler-a10 \
    --decks silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1 \
    --epochs 1 --batch-size 32 --lr 0.000003 \
    --learner-device cuda --execute
```

Identical configuration and output root resumes completed rounds and
per-worker chunks; increasing `--rounds` extends a run. The main
`progress.json` reports train rewards, heldout comparisons and
per-round KL after each finished optimizer update. Re-run first with
fewer scenarios/decks only when the explicit coverage report shows
unsupported combinations, and give that alternate pool a **new**
experiment identity.

**Validation status:** implementation committed to the independent H15
research branch, focused Python CI added, full workstation multi-encounter
smoke and long search training have **not yet run**. This stays a draft
until genuine collection and holdout evidence exist. The original native
game law calibration is independent of this emulator-reference training.


## M15 one-round multi-encounter workstation evidence (2026-10-30)

The user-provided `progress(20261030-143225).json` confirms the M15
**15-combat collection and 15-combat heldout cohort succeeded** under the
existing fixed H8 declared ReferenceNature prior. A new model checkpoint was
produced, with `15` actual searched training roots. The heldout mixture
comprised `cubex-a10:7`, `mawler-a10:4`, and
`vine-shambler-a10:4` combats.

| Heldout diagnostic | frozen H9 greedy | H15 raw greedy | H15 PUCT |
| --- | ---: | ---: | ---: |
| Victories | 15/15 | 15/15 | 15/15 |
| Mean exit HP | 36.4 | 35.4667 | 35.4667 |
| Mean encounter reward | 1.52 | 1.50667 | 1.50667 |
| Mean actions | 36.7333 | 32.9333 | 32.9333 |
| Search roots | 0 | 0 | 15 |

The new learner's in-sample policy KL to its search visit targets was
`0.0306663`. These values show pipeline functionality only. All 15
heldout fights were won, so win rate is saturated; 15 combats are too few
to infer tactical superiority. The **identical raw-H15 / PUCT aggregates**
are particularly important: the existing `progress.json` cannot
distinguish PUCT making the **same opening choice** from different choices
with offsetting final outcomes. The current evaluation searches only
**one** decision per combat; search at that root may simply reinforce
its own network prior.

**Offline audit added, no new emulator rollouts:** run

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
python -u tools/audit_h15_reference_search.py \
    --output-root results/h15-reference-m15-cohort-smoke-v1 \
    --round 1 \
    --report-path results/h15-reference-m15-cohort-smoke-v1/search-audit-round-0001.json
```

The audit reads the original updated checkpoint and heldout `eval-raw`
and `eval-search` worker files. It recomputes the **raw policy's legal
logits on exactly each searched public root**, compares selected actions
with PUCT visit argmax (same tie-breaking), computes visit-vs-prior KL,
and pairs outcome/HP/reward changes by exact case ID. It fails if checkpoint
hash, worker case set, encounter/deck/seed identities or target checkpoint
drift. It writes a **separate** JSON sidecar and leaves the completed
round immutable. This is the preferred next diagnostic; it avoids
re-running 15 reference combats.

For **new** M15 runs, `NativeSearchCollection.searched_root_diagnostics`
and `aggregate().search_override_rate` now report interventions and
visit-vs-prior KL directly. This is a changed code identity; the
completed smoke remains a valid immutable historical experiment. Use a
new output directory for any further runtime training, and keep the
previous completed-round files untouched.

After the audit, decide how to increase useful search signal: deeper
roots beyond the opening, sufficiently substantial search budgets,
and explicit search-vs-raw policy intervention measurements. Deep
public-history conditioning with H8's existing rejection sampler can
exhaust; test supported coverage rather than assuming many later
search roots always work. A 15-case smoke cannot justify a statement
that PUCT helps or hurts. The 80-round plan is retained as an
**initial experimental configuration**, with decisions about actual
budget informed by throughput, search interventions and genuine
heldout reference-law outcomes.


## M15 heldout search-intervention audit and budget-probe gate (2026-10-30)

The user-supplied offline audit `search-audit-round-0001.json` reads
the actual completed 15-case heldout cohort. **PUCT overrode the
same checkpoint's greedy opening action in 0/15 cases.** The paired
raw-H15 and PUCT records have identical combat outcomes, remaining HP,
rewards and decision counts on **every case** (not merely equal means).
Each root had 16 simulations, each ending with a neural-value cutoff:
`240` neural cutoffs, `0` internal terminal outcomes. The mean
search-visit-to-prior KL was just `0.0301383` nats. Six root visit
histograms cover all 15 roots, with three distinct selected greedy
action IDs. The differences reported in v2's single fixture do not
contradict this result: the present audit evaluates the **new student
on 15 different heldout fixture instances**, and at this low budget
its search never changed the raw policy's selected action.

**Assessment:** the existing one-root/16-simulation configuration
provides essentially no evidence that PUCT teaches a better policy
than the network already produces. The report establishes observed
behavior on this cohort; possible causes include insufficient search
budget/depth and limited value-head discrimination. Merely scaling
the same training loop to 80 rounds risks self-distillation. Avoid
conclusions about performance from the current small saturated sample,
and keep the native-game probability-calibration project separate.

**New diagnostic (committed):** `tools/probe_h15_reference_search_budget.py`
reuses the *existing frozen round-1 student checkpoint* and the
*exact same saved heldout case IDs, encounter/deck specifications and
independent ReferenceNature seeds*. It runs 16, 64 and 128 simulated
PUCT visits at the first decision, with **no optimizer updates**.
It compares to the immutable per-case saved raw-H15 evaluation.
New PUCT root diagnostics include neural-value-cutoff count, actual
terminal count, expanded public histories, simulated state
transitions, and the per-action empirical backed-up Q values.
The resulting `mean_visited_action_q_spread` tests whether search
estimates meaningful **action-dependent** differences rather than
only replicating neural priors. All outcomes are measured in the H8
declared reference-law game.

Run this *before* the expensive 80-round experiment, in a new
output directory (fish):

```fish
set ai ~/projects/sts2-ai
cd $ai
git pull --ff-only
source .venv/bin/activate.fish

python -m pytest -q \
    tests/test_h15_native_search_collector.py \
    tests/test_h15_reference_search_audit.py \
    tests/test_h15_reference_search_budget.py

python -u tools/probe_h15_reference_search_budget.py \
    --source-run results/h15-reference-m15-cohort-smoke-v1 \
    --round 1 \
    --output-root results/h15-reference-m15-puct-budget-probe-v1 \
    --budgets 16,64,128 --cases 15 --workers 15 --execute
```

The probe saves one `budget.json` per simulation budget plus a
combined `progress.json`, and resumes completed per-worker chunks
only when their case set, model SHA, budget, code/emulator identity and
prior configuration agree. The completed original cohort is immutable.
After this controlled budget probe, select a substantive budget and
increase **search decisions per combat** on a source-supported
A10 subset. Later public-history conditioning can be more expensive
than the first root: profile acceptance rates and report explicit
rejection exhaustion rather than silently omitting roots.


## M15 matched search-budget result and post-draw turn-start pilot (2026-10-30)

The user-uploaded `progress(20261030-145422).json` provides **real**
A10 ReferenceNature budget-probe results on the same fifteen heldout cases,
frozen H15 model, opening public roots and declared emulator chance model:

| Simulations per root | 16 | 64 | 128 |
| --- | ---: | ---: | ---: |
| PUCT override of greedy | 0/15 | 0/15 | 0/15 |
| KL(search visit policy || neural prior) | .0301383 | .0133173 | .0036900 |
| Mean visited-action backed-up Q spread | .01720 | .02561 | .14610 |
| Mean expanded information sets | 17.0 | 63.93 | 98.67 |
| Mean simulated transitions | 40.67 | 311.93 | 741.33 |
| Internal terminal evaluations | 0 | 0 | 0 |

All 15 matched raw/PUCT **individual** completed fights had identical
outcomes, exit HP, decisions and rewards at every budget. All simulations
ended at neural-value cutoffs (240/960/1920). The Q-value spread increased
by 128 simulations, but the visit policy **converged toward the neural
prior**, and it never changed a selected opening action. This demonstrates
the current **opening-root-only, value-cutoff** search makes no measured
policy intervention on this limited cohort despite considerable added work.
It does not establish that PUCT generally fails, and it does not measure
later-decision searches or terminal-rollout search.

**Next controlled pilot:** search at the **first ordinary player decision
of successive turns**, after new draws, instead of the first several
actions of the opening turn. The collector, round trainer and matched
budget probe now accept `--search-schedule turn_start`; the existing
`first` behavior remains the default. The schedule saves one search
per eligible public combat turn, bounded by `--max-searches`, and uses
the **existing H8 public-history-conditioned independent ReferenceNature
sampler**. Each searched root still gets its real completed-fight HP
reward for the value target; all chance outcomes remain emulator-owned.
This is an independent new experiment identity; the previous heldout
cohort and budget-probe outputs stay immutable.

First test **three source cases**, at most **two turns** and 16
simulations per selected root. This is a conditioning/coverage smoke,
not a training run:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish

python -m pytest -q \
    tests/test_h15_native_search_collector.py \
    tests/test_h15_reference_search_budget.py

python -u tools/probe_h15_reference_search_budget.py \
    --source-run results/h15-reference-m15-cohort-smoke-v1 \
    --round 1 \
    --output-root results/h15-reference-m15-turnstart-probe-v1 \
    --budgets 16 --cases 3 --workers 15 \
    --max-searches 2 --search-schedule turn_start \
    --execute
```

Inspect `results/h15-reference-m15-turnstart-probe-v1/progress.json`
for searched root count, override rate, Q separation and terminal/cutoff
counts. The per-worker source case diagnostics identify any unsupported
mechanics or posterior rejection exhaustion. If later-turn conditioning
exhausts the existing budget, this is a **provider coverage/throughput
finding**, rather than a reason to switch to an unverified native-RNG
model or block first-root training. Revisit the bounded H8
conditioning algorithm or choose supported combat snapshots at later
observable decisions as separate independent search roots.

**Decision before long training:** choose the search-root curriculum
and budget from demonstrated variation in state/turn and meaningful
counterfactual signal; then run extended search-guided learning on
A10/reference-law source runs. Preserve the original user preference
for 15 workers and full-length training once the learning signal has
actually been characterized.


## M15 user-requested difficult-combat final-turn study (2026-10-30)

**Motivation (user request):** opening-root PUCT agreement may conceal
useful late-combat decisions; lethal choices, imminent defeat and other
near-terminal states offer clearer value supervision. Investigate a
difficult A10 combat with roughly 50% observed **frozen H15 raw-policy**
wins, and record **every actual player action on its final turn**,
including losing trajectories.

**One diagnostic runner:** `tools/probe_h15_final_turn.py`.
It holds the **existing M15 round-1 H15 checkpoint** frozen, uses the
existing emulator-owned declared-law `H8ReferenceCombatBackend`
with `ReferenceNatureV111`, and executes baseline fights without PUCT
or optimizer changes. A single 15-worker screening cohort covers
**both source-supported A10 boss fixtures** (Ceremonial Beast and Vantom)
paired with the **two eligible boss-conditioned 20-card decks**:
boss-cycle and boss-poison (4 combinations, 12 independent combats per
combination by default). Standard and poison-kit decks create a normal-room
fixture and are invalid for these boss entries. Source IDs and boss/deck allocation are fixed *before* the
outcomes. The candidate whose measured win rate is closest to 50%
is validated on **32 independent fresh combats**; the result explicitly
reports both rates, Wilson intervals and whether the chosen candidate
actually falls within `0.50 ± 0.20`. No win-rate claim is made before
emulator evidence exists, and no case is selected by the favorable
outcome.

The final-turn extractor stores ordered *actual* actions including
kind, action ID and public payload, turn/decision index, visible player
HP/block/energy, enemy HP, number of legal choices, the policy
probability assigned to the executed action, and critic values at each
decision. It records the last player turn for **victories and defeats**
and compares the first final-turn critic estimate with the true
completed-fight H9 encounter return. All critic inputs are the public
decision observation only; retrospective selection of the final turn
is strictly **analysis provenance**, never a search-policy input. The
initial result is an observational endgame/value diagnostic, not a
claim that PUCT has improved the final turn. Full public-history
conditioning for late-turn PUCT remains a distinct experiment, if
warranted by this study.

The executable command (fish) is:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish

python -m pytest -q tests/test_h15_final_turn_probe.py \
    tests/test_h15_native_search_collector.py

python -u tools/probe_h15_final_turn.py \
    --source-run results/h15-reference-m15-cohort-smoke-v1 \
    --round 1 \
    --output-root results/h15-a10-boss-final-turn-v1 \
    --screen-per-pair 12 --validate-combats 32 \
    --workers 15 --execute
```

The single run writes `progress.json` containing all screened
scenario/deck win rates and the independently validated selected
baseline, plus `final-turn-actions.json` with ordered last-turn
action and value-head traces for every validation case. Saved worker
chunks permit safe resume; a pinned SHA manifest rejects changed
code/DLL/model/data configurations. **No new training gate is added.**
The established search-guided trainer remains independently runnable,
and the H15 learning agenda is unchanged.

## M15 difficult-combat multiprocessing exception transfer (2026-10-30)

The first 96-case A10 boss/deck screen failed in the **parent process**
while deserializing a worker-side exception:

`ModuleNotFoundError: No module named 'sts2_emulator'`,
reported as `BrokenProcessPool`.

The `sts2_emulator` Python binding defines its own `EmulatorError` class,
and worker imports were local to `_collect_worker`. A failure of
`backend.begin_reference` occurred **before** the per-case exception
handler, so the process manager tried to transport the custom exception
instead of a string. The underlying boss/fixture/codec error is **not
yet known**.

**Correction:** `tools/train_h15_reference_rounds.py` now registers
the emulator Python binding path in the parent; wraps worker failures
in JSON/pickle-safe text tracebacks before they cross the process
boundary; and includes `begin_reference` within the case-context
handler. This changes diagnostics, not the H15 search/learning
algorithm. A regression verifies safe transport of worker exceptions
without requiring their custom class in the parent.

The original `results/h15-a10-boss-final-turn-v1` manifest contains
old source-code hashes. To preserve experiment provenance, retry in
a **new** output directory:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
python -m pytest -q tests/test_h15_final_turn_probe.py \
    tests/test_h15_native_search_collector.py
python -u tools/probe_h15_final_turn.py \
    --source-run results/h15-reference-m15-cohort-smoke-v1 \
    --round 1 \
    --output-root results/h15-a10-boss-final-turn-v2 \
    --screen-per-pair 12 --validate-combats 32 \
    --workers 15 --execute
```

If the resumed screen fails on a specific boss/deck or source action,
the full original per-case traceback should now appear. Only that
actual failure should motivate a subsequent mechanics change; the
existing reference-law training run remains independent.




## M15 boss-room fixture eligibility correction (2026-10-30)

The second final-turn screening run unmasked the real emulator exception
`InvalidDataException: Fixed playable encounter is ineligible for its run entry.`
at `backend.begin_reference()` for a **Ceremonial Beast + Silent starter
deck** source case. The pinned emulator `ControlledPlayableCombatV111.cs`
lines 790–807 assigns a Boss room only for
`SemanticConditionedDeckPresetV111.IsBossDeck(fixtureDeckPresetId)`
(or an explicitly specified boss research fixture). The method
`IsBossDeck` recognizes **only** `silent-boss-cycle-20-v1` and
`silent-boss-poison-20-v1`. Starter, poison-kit and other ordinary decks
enter a Combat room and correctly fail boss encounter eligibility.
This was an **invalid source scenario/deck pairing in the AI probe**, not
an RNG coupling or missing native-game parity concern.

Corrected `tools/probe_h15_final_turn.py` to screen the **four valid
boss/deck combinations**, and reject normal-room deck presets before
launching 15 emulator workers. This keeps the purpose of the study
unchanged: measure a source-supported difficult A10 combat's empirical
raw-policy win rate, select the one nearest 50%, independently validate,
and inspect the ordered **last actual combat turn** for wins and defeats.

The corrected independent experiment identity is
`results/h15-a10-boss-final-turn-v3`:

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
python -m pytest -q tests/test_h15_final_turn_probe.py \
    tests/test_h15_native_search_collector.py
python -u tools/probe_h15_final_turn.py \
    --source-run results/h15-reference-m15-cohort-smoke-v1 \
    --round 1 --output-root results/h15-a10-boss-final-turn-v3 \
    --screen-per-pair 12 --validate-combats 32 \
    --workers 15 --execute
```

Four screening pairs × 12 fights = **48 screening fights**, followed by
**32 separately seeded validation fights**. Existing versioned v1/v2
experiment directories remain untouched. Observations are under the
existing declared-law H8 ReferenceNature distribution, and this study
does **not** gate substantive search-guided training.


## H15 M16 — substantive hard-A10 late-combat search/value training (2026-10-30)

The completed difficult-combat probe on 32 *independent* declared-law
ReferenceNature cases selected **Ceremonial Beast A10 + Silent 20-card
boss-cycle** (14 wins / 32; 43.75%, Wilson 95% interval
28.17–60.67%). The first last-turn critic prediction was systematically
optimistic on defeats: approximately +0.185 realized H9 reward error
on 18 losses, compared with +0.051 on 14 wins. This is a *diagnostic*
on one frozen H15 student and a supported synthetic boss research
fixture, not a search-improvement claim.

**M16 implementation, now committed:**

1. Add `--search-schedule danger`: consider first ordinary legal player
   decision on a new combat turn meeting *only current visible*
   turn / player HP / enemy HP thresholds. The `max-searches` limit bounds
   PUCT roots per actual completed fight. No terminal outcome or future
   information participates in root selection. A fight that ends without
   an eligible danger root is fully counted; it supplies no fictitious
   PUCT sample. Entire all-empty PUCT cohorts still fail closed.
2. Add **value-only** source rows (new separate
   `sts2-h15-combat-realized-value-target-v1` schema) for at most
   `--late-value-max` public decisions, sampled at later distinct turns
   and once more on visible near-lethal enemy HP. These observations are
   labelled by the *realized* completed-fight H9 encounter reward and
   are never given synthetic PUCT visits. Repeated same-decision labels
   at searched roots are omitted. Mixed minibatches compute policy
   cross-entropy **only on genuine PUCT targets**, and critic MSE on
   *all* actual completed-combat targets, including critic-only batches.
3. For a 20-card, ~60-decision boss, the existing **fair posterior**
   rejects candidate Nature seeds unless they reproduce the full public
   history, including opening draws. That is still a supported, default,
   explicitly scoped *fair* mode, but it is computationally unsuitable
   for repeated deep boss-turn PUCT. The opt-in
   `--search-provider root_conditioned` reuses the **existing emulator
   H8 `h8_fork`**, which atomically clones the complete mechanics and
   declared ReferenceNature cursor. Every player policy/value prediction
   still receives only public observations, but the *search teacher*
   conditions on the source's hidden realization. The search visit
   targets are tagged with
   `h15-h8-root-conditioned-combat-v1`, and distillation refuses a
   mislabeled fair-vs-root-conditioned search corpus. This mode is a
   **possibly privileged, approximate search teacher**; it does not
   support a certified fair-posterior or native-game-unbiased claim.
4. Optionally use `--search-leaf-mode terminal`, continuing each H8
   fork to a true combat-terminal reward under its engine-owned
   ReferenceNature cursor. Censored/unsupported mechanics fail explicitly.
   Per-round metrics now report actual terminal evaluations, neural
   cutoffs, root action overrides, visit-policy KL, visited Q spreads,
   late value target counts and completed fights without eligible roots.
5. Preserve repeated 15-worker training, CUDA optimization and
   matched source-run-disjoint reference-law evaluations. The frozen
   baseline in a new M16 run is the **H15 checkpoint supplied through
   `--initialize-from-model`**, not automatically an old H9 checkpoint.
   Both raw and root-conditioned-PUCT heldout performance are reported
   under fresh ReferenceNature source seeds.

**Aggressive substantive run, selected after user's request to proceed:**

```fish
cd ~/projects/sts2-ai
git pull --ff-only
source .venv/bin/activate.fish
set emu ~/projects/sts2-emulator
set rev a411a19491a93b58ae4fdbc6d1f3345c9fca67cf
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll

python -m pytest -q tests/test_h15_late_value_training.py \
    tests/test_h15_root_conditioned_search.py \
    tests/test_h15_native_search_collector.py \
    tests/test_h15_combat_search.py
and python -u tools/train_h15_reference_rounds.py \
    --emulator-root $emu --emulator-revision $rev \
    --emulator-dll $dll \
    --initialize-from-model results/h15-reference-m15-cohort-smoke-v1/round-0001/student.pt \
    --output-root results/h15-m16-ceremonial-boss-search-r80-v1 \
    --rounds 80 --train-combats 128 --eval-combats 64 \
    --eval-every 10 --workers 15 \
    --scenarios ceremonial-beast-boss-a10 \
    --decks silent-boss-cycle-20-v1 \
    --fixture-seed h15-reference-m15-fixture-v1 \
    --seed-prefix h15-m16-ceremonial-boss-v1 \
    --search-provider root_conditioned \
    --search-schedule danger --search-leaf-mode terminal \
    --simulations 32 --max-searches 2 \
    --tree-depth 8 --rollout-depth 256 \
    --late-value-max 6 --late-value-min-turn 7 \
    --late-value-player-hp 25 --late-value-enemy-hp 40 \
    --epochs 1 --batch-size 32 --lr 0.000003 \
    --value-weight 1.0 --learner-device cuda --execute
```

This is 80 × 128 = **10,240 training combats**, up to 20,480
selected PUCT roots × 32 = **655,360 possible terminal-rollout
simulations**, plus independent heldout source fights at every tenth
round. The actual number of searched roots is observed, not assumed.
Full-combat terminal rollouts may be expensive; persist every round
and worker chunk and record the actual throughput and stop/failure
provenance. Do not interpret the single run as decision-grade model
improvement; later replicate independent training seeds with a matched
compute comparator and untouched holdout following
`docs/EXPERIMENT_EVIDENCE_STANDARD.md`.

**Conservative alternative** for later controlled comparison:
run the same 80-round cohort with
`--search-provider fair_posterior --search-schedule first
--search-leaf-mode value --max-searches 1`, while keeping
`--late-value-max 6`. It uses the validated bounded opening
fair-search teacher plus the same dense late critic supervision,
avoids privileged state in PUCT policy labels, and leaves the
late-stage PUCT question to the explicitly biased experimental mode.

**Execution state:** Code and focused tests committed to existing
H15 draft PR #146. GitHub CI and user's workstation emulator/GPU
execution are not yet verified. The documented fish command begins
the full requested run immediately after its short correctness tests
pass; no further exploratory pilot is a prerequisite.


## H15 M17 — victory-margin reward and deeper tactical PUCT (2026-10-31)

**Latest user decision (supersedes M16 for *new* training):**
M16's round-20 held-out comparison was again reported as having no
meaningful raw-network improvement. The user explicitly requested both
the defeat-side `beta` reward and **more frequent, deeper tree
expansion**. M16's already-running 80-round artifacts and branch remain
unaltered. Implementation lives on a **new**
`research/h15-m17-victory-margin-puct-20261031` branch based on M16.

### Reward definition and provenance

The historical H9 terminal objective remains unchanged in
`encounter_total_hp_score`. M17 adds a *new*, separately versioned
`encounter_nonminion_total_hp_defeat_margin_beta050_v1` objective:

- Victory: `R = 1 + player_exit_hp / player_max_hp`
- Defeat: `R = (1 - 0.5) * (1 - remaining_nonminion_hp / total_nonminion_hp)`

It preserves a full 0–1 HP-preservation increment on wins and makes
the strongest loss worth at most 0.5; minimum victory reward is 1.0.
The common H15 normalization `V = R / 2` remains valid. All M17
**actual episode targets**, value-only labels, **PUCT terminal
backups**, frozen-baseline and trained-model evaluation rewards use this
same formula. The loader rejects any mixed reward-version rows.
Comparisons to M16 should use **win rate / exit HP**, or explicitly
recompute both reward summaries under a shared objective; raw means
from two different reward versions are not directly comparable.

### Search algorithm and source-visible trigger

- M16 `--search-schedule danger` searches at most one decision
  per qualifying turn and exhausts `--max-searches 2` quickly.
  M17 adds **`--search-schedule tactical`**. Once turn >= 7 and
  public player HP <= 25 **or** any visible living enemy HP <= 40,
  any ordinary legal player decision with at least 2 legal actions
  may receive its own PUCT search, *including multiple decisions
  within a single turn*. Nested player-choice fallback decisions
  remain excluded from training roots.
- `--max-searches 6 --reserved-lethal-searches 3
  --lethal-enemy-hp 20` reserves three of the six root budgets for
  states with a visible enemy at or below 20 HP. A fight may use at
  most three early danger searches, preserving searches for late
  combinations. The trigger inspects only **current public data**.
- `--tree-expansion-mode full_depth` extends each simulation beyond
  its first unvisited node: the newly expanded node's policy is used
  immediately for another PUCT selection, continuing until the
  configured tree depth or a true terminal state. All visited edges
  receive the corresponding backed-up value. In M16 `one_leaf`,
  search stopped expanding after *one new node* each simulation
  and used a policy-sampled terminal rollout.
- `--tree-depth 16` (was 8), `--simulations 64` (was 32),
  `--search-leaf-mode terminal`. At the chosen depth limit, complete
  a real ReferenceNature-emulator continuation rather than faking
  terminal labels or cutting at the imperfect critic.
  `--rollout-depth 256` is unchanged. New v3 deep-tree target
  provenance distinguishes it from the prior v2 PUCT visit teacher.
- As in M16, `root_conditioned` forks the **existing emulator
  `h8_fork`**, including its ReferenceNature cursor. This is a
  **possibly privileged, approximate teacher**: only public
  observations enter the neural network, while PUCT rollout outcomes
  retain source-root hidden-state conditioning. The default
  `fair_posterior` remains selectable and is separately labeled.
  No new chance sampler or emulator mechanics was introduced.
- Round reports retain original M16 metrics and add
  `near_lethal_search_roots` and
  `mean_maximum_expanded_depth`. Recorded target rows include
  reward version, PUCT v2/v3 identity and tree-expansion mode.

### Recommended new full run (fish)

```fish
# First stop the running M16 process at a completed-round boundary;
# do not switch a checkout while its training job still uses these files.
cd ~/projects/sts2-ai
git fetch origin
git switch research/h15-m17-victory-margin-puct-20261031
git pull --ff-only
source .venv/bin/activate.fish

set emu ~/projects/sts2-emulator
set rev a411a19491a93b58ae4fdbc6d1f3345c9fca67cf
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll

python -m pytest -q \
    tests/test_h15_m17_reward_search.py \
    tests/test_h15_native_search_collector.py \
    tests/test_h15_root_conditioned_search.py \
    tests/test_h15_late_value_training.py \
    tests/test_h15_combat_search.py
and python -u tools/train_h15_reference_rounds.py \
    --emulator-root $emu --emulator-revision $rev --emulator-dll $dll \
    --initialize-from-model results/h15-reference-m15-cohort-smoke-v1/round-0001/student.pt \
    --output-root results/h15-m17-ceremonial-boss-beta05-deep-r80-v1 \
    --rounds 80 --train-combats 128 --eval-combats 64 \
    --eval-every 10 --workers 15 \
    --scenarios ceremonial-beast-boss-a10 \
    --decks silent-boss-cycle-20-v1 \
    --fixture-seed h15-reference-m15-fixture-v1 \
    --seed-prefix h15-m16-ceremonial-boss-v1 \
    --search-provider root_conditioned \
    --search-schedule tactical \
    --defeat-margin-beta 0.5 \
    --tree-expansion-mode full_depth \
    --search-leaf-mode terminal \
    --simulations 64 --max-searches 6 \
    --reserved-lethal-searches 3 --lethal-enemy-hp 20 \
    --tree-depth 16 --rollout-depth 256 \
    --late-value-max 6 --late-value-min-turn 7 \
    --late-value-player-hp 25 --late-value-enemy-hp 40 \
    --epochs 1 --batch-size 32 --lr 0.000003 \
    --value-weight 1.0 --learner-device cuda --execute
```

The separate output path and model/revision/code hashes preserve M16
artifacts; the same seed prefix intentionally supplies matched
independent heldout source cases for the M16/M17 comparison. This is
**80 × 128 = 10,240 training combats** and **up to
80 × 128 × 6 × 64 = 3,932,160** root simulation trajectories,
versus M16's upper bound of 655,360. Search is thus potentially
~6 times as intensive before accounting for extra depth and longer
terminal rollouts; actual compute must be measured rather than assumed.
Only completed fights produce labels. Source-run-disjoint engine
reference holdouts remain at every tenth round. Interpretation follows
`docs/EXPERIMENT_EVIDENCE_STANDARD.md`: this is a substantive
single-training-seed comparison, **not** a decision-grade five-seed
promotion claim.

**Validation status:** focused source-level regression tests and CI
integration added. Workstation emulator and GPU training require a
separate execution; no completed M17 results are yet claimed.

## M17 R10/R30 fixed-public-state diagnosis (2026-10-31)

Motivation: On the same 64-case Ceremonial Beast A10 holdout, the raw R10
checkpoint wins 29 fights and the raw R30 checkpoint wins 16. The uploaded
progress report contains aggregate results but neither checkpoint weights
nor per-decision records. The goal is to identify where their action
distributions first change and whether PUCT prefers different actions
at the exact same public state.

New diagnostic tool: tools/diagnose_h15_m17_r10_r30.py.

Method:
1. Load the existing R10 and R30 student.pt checkpoints and versioned
   M17 config. Replay all 64 original heldout Nature seeds under each
   raw greedy policy, evaluating BOTH checkpoints at EVERY encountered
   public state and legal menu. Verify 29 and 16 wins against the original
   evaluated round.json receipts, failing on a mismatch.
2. Rank cases where R10 won / R30 lost first. Select first raw-policy
   disagreement, largest policy divergence and late near-lethal roots,
   including controls. Replay the action prefix to reconstruct the exact
   original emulator state. Check every public observation hash AND
   legal menu before proceeding.
3. Run both root-conditioned PUCT evaluators at that same state, with
   the same 64 simulations, depth 16 and RNG seed. Record per-action
   neural priors, values, PUCT visits, Q estimates and search actions.
   These are *paired on the same hidden realization*, not posterior
   unbiased action values.
4. Optionally compare forcing the two different raw actions followed by
   the same frozen R10 continuation policy, sampling paired RNG seeds.
   Report conditional rollout win rates and rewards; these are noisy,
   root-conditioned diagnostics rather than proofs of action optimality.

Fish command (stop active M17 training at a completed-round boundary
before switching the checkout):

~~~fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h15-m17-r10-r30-paired-diagnostic-20261031
git pull --ff-only
source .venv/bin/activate.fish

python -m pytest -q tests/test_h15_m17_paired_diagnostic.py

and python -u tools/diagnose_h15_m17_r10_r30.py \
    --run-root results/h15-m17-ceremonial-boss-beta05-deep-r80-v1 \
    --output-root results/h15-m17-r10-vs-r30-paired-v1 \
    --cases 64 --search-cases 16 --roots-per-case 3 \
    --simulations 64 --tree-depth 16 --rollout-depth 256 \
    --counterfactual-rollouts 8 --max-decisions 512 \
    --workers 15 --execute
~~~

Output: paired-report.json contains exact regressed/improved case IDs,
all-public-state prediction divergence, matched search-action differences
and optional common-continuation differences. public-decisions.jsonl
contains per-decision predictions; paired-roots.jsonl contains
per-action PUCT/Q comparisons at verified identical states. Worker
chunks are resumable and hash-provenanced.

All 64 cases are greedily replayed; up to 16 × 3 = 48 selected
decision roots receive two 64-simulation PUCT searches each (up to
6,144 search simulations), plus optional forced-action continuations.

The checkpoints and live H8 simulator are on the user's workstation,
not attached to this conversation or checked into GitHub. Numerical
same-state probabilities, visits and Q values require running this
diagnostic on those files. Do not infer them from the progress summary.
This is a *diagnostic*, not decision-grade model validation.

### M17 paired diagnostic performance and cached-trace recovery

The original diagnostic first completes all 64 greedy paired traces, then
runs up to 16 cases × 3 roots/case × 2 checkpoints × 64 simulations
= **6,144 terminal PUCT simulations**. Each search may consume many
H8 emulator transitions and CPU model inferences, particularly when
the chosen first-disagreement root occurs early in a fight.
The original diagnostic printed a line only **after each complete
search worker's cases**, so after all 15 trace workers the log could
remain silent for a long time despite active CPU computations.

The current branch adds per-case/per-root/per-checkpoint PUCT start
and finish log lines with elapsed seconds. These require restarting
the diagnostic process because an already-running Python process
cannot load the new source. The complete greedy traces remain safely
saved in the old output directory's traces/worker-*.json.

After interrupting the search and pulling the latest diagnostic branch,
extract the useful **already-completed raw policy comparison**
without re-running any H8 session:

~~~fish
python -u tools/summarize_h15_paired_traces.py \
    --run-root results/h15-m17-ceremonial-boss-beta05-deep-r80-v1 \
    --paired-root results/h15-m17-r10-vs-r30-paired-v1
~~~

This creates trace-only-report.json (first raw disagreement per case,
R10→R30 outcome reversals and all-state policy divergence; no PUCT
claim). The cached 64-case traces are hash-verified.

For a **lightweight separate paired PUCT check**, reuse complete
trace worker chunks and change the heavy budget with a *new*
output-root to preserve the original run's immutable config and
analysis-code hash:

~~~fish
set old results/h15-m17-r10-vs-r30-paired-v1
set quick results/h15-m17-r10-vs-r30-quick-v1
mkdir -p $quick
cp -a $old/traces $quick/

python -u tools/diagnose_h15_m17_r10_r30.py \
    --run-root results/h15-m17-ceremonial-boss-beta05-deep-r80-v1 \
    --output-root $quick \
    --cases 64 --search-cases 3 --roots-per-case 1 \
    --simulations 8 --tree-depth 8 --rollout-depth 256 \
    --counterfactual-rollouts 0 --workers 15 --execute
~~~

This reuses all 64 matching trace cache chunks and runs at most
3 × 1 × 2 × 8 = 48 PUCT simulations. The result is a **diagnostic
smoke** for a few highly changed cases, not a stable estimate of
optimal actions. Logs should show each PUCT root and model start/done.
Only increase the budget after measuring a representative root's
wall time. Running both the original heavy job and the quick job
concurrently risks memory/CPU saturation.
