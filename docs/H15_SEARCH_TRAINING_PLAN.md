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
