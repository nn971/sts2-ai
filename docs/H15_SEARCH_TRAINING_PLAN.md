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
