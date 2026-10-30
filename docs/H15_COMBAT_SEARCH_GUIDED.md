# H15 M1 — fair stochastic combat PUCT and policy/value distillation

**Implementation status:** search kernel, public move-aware codec, strict sample
schema, and offline trainer. **No native H8 fair-combat adapter or native
training result yet.** This branch is based on H10's native-state combat stack
so the native research fixtures and corrected H9 reward remain available.
H14 macro/GAE and its checkpoints are unchanged. This is the first milestone
of the agreed hierarchical MuZero-inspired design.

## Reward (unchanged H9)

```text
defeat: 1 - remaining_non_minion_hp / total_non_minion_hp
win:    1 + player_exit_hp / player_max_hp
critic: reward / 2
```

Use authoritative `encounter_nonminion_total_hp_v1` accounting (summons,
multistage encounters, exclusion of minions) from H9. It is **not** a newly
designed reward. For each combat, the terminal outcome is the observed
realization under the eventual search-selected behavior; never relabel a
search cutoff with a fictitious victory/defeat. The value head predicts that
realized terminal reward / 2. No derived continuation value or H12 predictor.

## Components

- `models/h15_combat_codec.py`: additive enemy-owned, public `move_id` tokens.
  Original action joins and Transformer shapes are unchanged. H15 has its own
  input schema, so H9 checkpoints still load unchanged, while H15 checkpoints
  MUST be served by this encoder.
- `search/combat_puct.py`: stochastic PUCT using the actor's logits for
  priors and the critic for expanded-leaf evaluation. Chance probabilities
  are sampled rather than maximized; all simulation paths use fresh
  publicly conditioned hidden continuations. Cutoff estimates affect search
  values but are **not** passed off as measured returns. An adapter lacking
  `h15-public-history-combat-chance-v1` is rejected.
- `training/h15_search_supervision.py`: certified, versioned target records.
  Root visits train the actor; the realized terminal H9 reward trains the
  critic. Both heads share the 14.66M network backbone.
- `tools/train_h15_search_distill.py`: load an existing H9 or H15 state dict,
  fit on certified JSONL targets, save an H15-versioned checkpoint with model
  and input hashes. Training requires explicit `--execute`.

## Required next integration before native training

1. Implement **and certify** a native H8 combat sampler conditioned on the
   entire public root history. The existing exact forked `CombatSnapshotRecipe`
   alone does not provide fair chance branches. Avoid routing its hidden RNG
   directly into search. Reuse interchangeable native Chance Law providers.
2. Run the frozen H9 R250 as the first policy/value search teacher. Its value
   output may be poorly calibrated on native combat; clip invalid negative
   predictions when constructing the evaluator (record clip count) and use
   full terminal rollouts/controlled cutoff checks before trusting deep search.
   No model checkpoint is changed by adding the move tokens.
3. Collect real combat-entry states including the weak pool and normal/elite
   encounters encountered by H14. Search on selective difficult decisions
   and retain the entire realized fight to label value targets. Search labels
   must record source-run seed, pinned emulator revision and teacher SHA.
4. Confirm supervised policy/value gradients and improvement on **heldout
   native reachable combats**, measuring survived encounters, damage taken
   at equal entry conditions, potion consumption, and full Act 1 clearance
   using the same frozen macro policy. Do not select a model from training
   loss alone.

## Offline fitting interface (after native fair target collection)

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

python -u tools/train_h15_search_distill.py \
    --initialize-from-model results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --search-data results/h15-fair-search/targets.jsonl \
    --output results/h15-fair-search/player-m1.pt \
    --epochs 8 --batch-size 32 --lr 0.00003 \
    --device cuda --execute
```

The example `targets.jsonl` is **not produced by this milestone**.
The collector must be implemented and verified first. The CLI rejects
unmatched teacher SHA, uncertified chance identity and malformed target
menus. Every search example's visit counts must sum to its simulation budget.

## Deliberate limitations

- The search adapter is a strict protocol boundary, not a certificate of
  correctness by itself; native source-conditioned chance tests must
  establish that an implementation actually obeys its declared capability.
- An H15 checkpoint cannot be deployed by the old H9 actor, because that
  actor omits move tokens. Serving needs an H15 inference adapter.
- Search improvement is **not claimed** without independent native replay.

## Public-state fair replay integration (M1 continuation)

`search/h15_public_combat_adapter.py` now adapts **an explicitly supplied,
certified** `FairContinuationSampler` to H15 search. A full public
observation/action/menu transcript determines the information-set root.
Each simulation draws a distinct conditioned continuation; the adapter
retains the same hidden continuation through all subsequent decisions and
releases it after the simulation. The resulting combat ends at the first
postcombat public observation, before reward picking.

The H9 score uses public `defeat_total_non_minion_enemy_hp` and
`defeat_remaining_non_minion_enemy_hp` at defeat; missing HP fields fail
loudly. Victory uses public exit HP/max HP. This bridge cannot itself
certify the *native* posterior: supplying an oracle-fork sampler is invalid.
Its toy tests validate the observer, score and ownership lifecycle; the
source-conditioned native sampler and distribution test remain next work.

`models/h15_policy_value.py` supplies frozen H9/H15 checkpoint inference
for search, with move-aware encoding and a counted [0,1] critic clamp.


## M2 implementation: native history collector and robust search teacher

The following code is now present on this branch, but **native fair
rollouts are not yet executable** until the H8 sampler contract below exists.

- `H15CombatEvaluator` loads the old H9 checkpoint with the **original H7
  public combat codec** and unchanged logits. H15-trained checkpoints use
  the move-aware codec. Adding enemy move tokens to H9 before distillation
  would silently change the policy through attention and is prohibited.
- `CombatPuct(..., leaf_mode="terminal")` now completes sampled combat
  futures by following the frozen policy's legal-action priors. Chance
  events come from the independent fair sampler. Censored futures raise
  `IncompleteCombatRollout`: the teacher must never treat a cutoff
  critic estimate as a real H9 combat outcome. The existing `value`
  leaf mode remains available for later calibrated critics. Search target
  rows explicitly record the leaf mode and complete-terminal/cutoff counts.
- `training/h15_native_search_collector.py` reconstructs the entire
  player-visible observation/action/LEGAL-MENU history from an H10
  `CombatSnapshotRecipe` using the pinned native A10 run. The exact
  hash is used only by the replay verifier. For each searched combat
  decision, it conditions independent simulations on the public history,
  chooses the most visited legal action, executes that action in the
  actual run, and at combat exit labels every saved search root with
  the **realized** `encounter_nonminion_total_hp_v1` score. An unfinished
  combat yields no targets. The collector is an API requiring an
  independently certified `FairContinuationSampler`, NOT a CLI that
  silently falls back to an exact fork.
- `search/h15_public_combat_adapter.py` now hashes newly revealed public
  actions/states incrementally inside each simulation; it stores the
  complete public prefix only for roots at which a fresh hidden
  continuation must be drawn. This removes repeated quadratic copies
  of long full-run transcripts during tree search.

### Blocking emulator contract: fair native combat continuations

The current H8 runtime has provider-controlled local reference card draws,
enemy intents, exact decision replay, version-owned combat sessions, and
source-valid combat mechanics. It does **not** yet provide an independently
certified *public-history-conditional combat-root sampler* that supports the
arbitrary native H10/H14 deck, HP, relic, enemy state, and observed reveals.

An implementer must connect the H8 stochastic provider to native combat
state construction **without retaining unseen future outcomes from a live
run's RNG cursor**. In particular, a plain H10 `backend.fork(handle)`,
an H8 fork with an unchanged hidden provider, or changing only the local
reference seed while retaining other hidden future random state cannot be
asserted to be fair. The sampler must either reconstruct independent
publicly conditioned combat states or prove that the retained checkpoint
has no relevant hidden future state for each supported chance site. It must
return owned `StateHandle` objects through the existing
`FairContinuationSampler` contract, with an explicit versioned scope of
supported sites. Unsupported conditioning must fail loudly.

Native acceptance should start with first-combat public states, verify
chance distributions (card draws, enemy intents, random targets and
source-dependent effects), then progressively enable later public roots
and multi-stage boss mechanics. Do not use source-run future draws,
hidden seed, RNG cursor, or exact-state MCTS visit statistics as
search-policy labels.

Only after the fair sampler is working is it appropriate to write a
15-worker native collection CLI and launch long H15 search/distillation
training. The existing offline `train_h15_search_distill.py` already
consumes the target-record format; neither the macro training system nor
H9 reward needs a redesign.


## M3 — runnable fixed-fixture H8 Reference Nature search (research model)

The first executable search collector uses a **declared, limited chance
prior**, `h15-h8-fixed-fixture-independent-reference-nature-v1`.
It holds an H8 source-supported A10 Silent combat fixture and all
non-Reference-Nature seed-derived mechanics fixed; it varies only the
63-bit H8 Reference Nature seed. Search samples new independent seeds,
replays the observed combat action/observation/legal-menu prefix, and
accepts a candidate only when every public frame and menu matches.
This is exact rejection conditioning **under that explicitly fixed
fixture/Reference Nature prior** (up to the deterministic PRNG model).
It does not claim native-run posterior fairness or H14 deck distributions.
All unsupported Nature boundaries and rejection exhaustion fail closed.

The H8 JSONL runtime mutates its versioned session in place. H15 now
supports that exact capability while continuing to release normal
immutable-state handles as before. Full independent combat rollouts
grow the PUCT information-set tree across repeated simulations.
Results include both search visits and the actual completed fight's H9
terminal reward. The data file records `chance_prior_id` and
`experiment_regime`, and distillation requires an exact matching
`--accepted-chance-prior`.

Run **one small experiment** before any long searches. This uses the
existing `sts2-emulator` directory and a compiled .NET 9 DLL:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h15-combat-search-guided-20261029
git pull --ff-only
source .venv/bin/activate.fish

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
dotnet build $emu/Sts2Emulator.sln -c Release -m:1

python -m pytest -q \
    tests/test_h15_h8_reference.py \
    tests/test_h15_combat_search.py \
    tests/test_h15_public_combat_adapter.py \
    tests/test_h15_native_search_collector.py

python -u tools/collect_h15_h8_reference.py \
    --emulator-root $emu \
    --emulator-revision $rev \
    --emulator-dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll \
    --h9-checkpoint results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --output-root results/h15-h8-reference-smoke-001 \
    --scenario mawler-a10 --deck silent-starter-v1 \
    --simulations 4 --max-searches 1 --max-candidates 16384 \
    --device cuda --execute
```

If a source-specific public opening is so rare that the independent
Reference Nature seed rejection runs out of candidates, the experiment
will explicitly fail rather than resample the live hidden seed. This
rejection procedure is a correctness-first research baseline; its cost
will likely preclude large training on arbitrary late-game histories.

To distill targets from that experiment, explicitly name the same
model prior:

```fish
python -u tools/train_h15_search_distill.py \
    --initialize-from-model results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --search-data results/h15-h8-reference-smoke-001/targets.jsonl \
    --accepted-chance-prior h15-h8-fixed-fixture-independent-reference-nature-v1 \
    --output results/h15-h8-reference-smoke-001/player-m1.pt \
    --device cuda --execute
```

**Do not infer policy improvement from the one-fight smoke.** Follow-up
requires disjoint fixture/seed cohorts, many combat searches, measured
per-fight compute, and comparison of H9, H9+search and standalone H15
on identical held-out source distributions.

### Native full-run H14 integration remains separate

`collect_native_combat_search` retains its separate pinned H10
`CombatSnapshotRecipe` reconstruction, but cannot use the fixed-fixture
sampler for full native H14 histories. Such a substitution would fix
hidden non-Nature state from one realized run and misrepresent its
posterior. Implement an H8 native combat-root constructor with public
HP/deck/relic/enemy state and a public-history-conditioned joint
stochastic provider before asserting a genuinely native full-run
continuation distribution. Search under the limited Reference Nature
fixture model is preparatory learning, not the final deployed agent.


### M3 nested H8 player-choice continuation

The initial Mawler search reached a real H8 `player_choice` boundary.
H8 publishes `player_card_selection` and `draw_hook_action_choice`
as player decisions; they are **not** random Nature choices. The H15
fixed-fixture adapter now uses the existing
`fair_player_choice_actions` validator to intersect the source pending
action IDs with the fair public action menu and forwards **only** the
public observation/menu. A public kind tag distinguishes these nodes
without forwarding the source's pending-choice internals.

H9 R250 did not learn the full non-hand card-selection action encoding.
At such nodes, the H15 teacher therefore uses explicitly counted,
uniform legal priors during its complete-combat rollouts. PUCT still
branches and evaluates these choices through realized final combat
rewards. The training collector omits nested-choice positions as
direct policy/value targets; ordinary combat positions remain trainable.
The output `report.json` includes
`nested_choice_uniform_fallbacks` so this approximation can be
quantified. A later versioned action codec must represent the full
selection menu and train those decisions directly.

Nested choices must be processed by player action, never by a Nature
draw/intent resolver, and must never be silently skipped. Unsupported
choice kinds continue to fail closed.


## M4 — primary dataset from actual native runs

**Important change in priority:** `tools/collect_h15_long.py` remains a
*research-only synthetic fixture* experiment. It is NOT the primary long
training source. Instead `tools/collect_h15_native_entries.py` records
`CombatSnapshotRecipe` entries while a real policy plays entire native
Overgrowth A10 runs. The emulator generates each legal deck, relic, potion,
HP, floor, map, opponent and card acquisition history jointly; no manual
deck sampling is used for the primary corpus.

The collector implements:

- 15 CPU workers, each with a persistent emulator and pinned frozen H9
  combat model and an H10 phase-split macro checkpoint. Multiple distinct
  H10 macro checkpoints can be supplied to broaden policy coverage.
- Separate full-run train/heldout seeds, never a random split of combat
  snapshots from the same seed. Every native entry has a replayable public
  action prefix, fair observation identity and auditor-only exact state hash.
- Per-run atomic/resumable artifacts. Consolidated `train.jsonl` and
  `heldout.jsonl` use the existing H10 `load_recipes` format. A public-only
  build descriptor hashes normalized actual deck contents and records
  encounter, floor, HP, relic and potion inventory for coverage reporting.
- Explicit run/error accounting, structured live progress and
  `coverage.json`. Repeat the same invocation to resume. Use
  `--retry-failed` to retry errored source runs after diagnosing the cause.
- Initial pinned distribution: H10 frozen H9 + *H10 phase-split macro*.
  This is a genuine native macro-to-combat configuration distribution.
  The stronger H14 macro policy currently lives on a separate experimental
  branch and is **not yet an accepted loader** for this collector. Integrate
  its versioned macro codec/checkpoint loader or collect from the H14 branch
  before declaring the final joint actor distribution representative.

First check CLI and a very small *native* collection:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h15-combat-search-guided-20261029
git pull --ff-only
source .venv/bin/activate.fish

set emu ~/projects/sts2-emulator
set rev (git -C $emu rev-parse HEAD)
# Set $macro to an existing H10-compatible phase-split macro .pt checkpoint.
set macro /path/to/existing/h10-phase-split-macro.pt

python -m pytest -q tests/test_h15_native_entry_corpus.py
python -u tools/collect_h15_native_entries.py \
    --emulator-root $emu \
    --emulator-revision $rev \
    --emulator-dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll \
    --combat-checkpoint results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --macro-model $macro \
    --output-root results/h15-native-a10-source-v1 \
    --train-runs 8 --heldout-runs 4 \
    --workers 4 --execute
```

For a *new* long-run corpus, choose a new output root and use
`--train-runs 2048 --heldout-runs 256 --workers 15 --execute` while
retaining the checkpoint/revision/seed-prefix identity. Each run produces
zero or more native entries. Check `coverage.json` for rare elites/bosses,
weak early-game concentration, and failures before choosing training
weights. The default old H10 policy may reach the boss rarely.

Once the native source corpus has nonempty train AND heldout splits, the
existing `tools/train_h10_native_combat.py` can train from them using
`--synthetic-per-round 0` if a fully native PPO control experiment is
desired. That is legitimate on-policy combat learning from actual native
run entries. It does **not** yet produce H15 PUCT search visit targets.

### Intended joint-policy collection cycle

After the H14 actor is integrated, regenerate native source cohorts from
the *current* macro policy and current frozen/updated combat policy in
alternating training generations. Keep each cohort's policy/checkpoint
SHA and source-run seeds, and reserve independent evaluation runs. A
small, explicitly labeled supplementary stream of generated elite/boss
fixtures can improve coverage when the policy rarely reaches later acts;
never report that oversampled stream as naturally encountered frequency.

A native replay recipe faithfully reproduces the original hidden RNG
state for evaluation. It must **not** be handed to PUCT as a fair posterior
sample: fair search still requires a conditional chance provider for the
actual public history. The fixed-fixture Reference Nature sampler remains
limited to synthetic fixtures. Do not mix its targets into native-search
training without an explicit chance-model distinction.


## M5 — declared Chance Law bridge and native adaptive cohort planning

Research emulator branch:
`nn971/sts2-emulator:research/h15-native-chance-law-20261030`
(based on pinned `2c46c703a30875d6ef84fcbe0a0511fec9c74dcd`).
It exposes `H8CombatRuntimeV1.ChooseChanceLawOutcome` and JSONL
`h8_resolve_chance_law_outcome`. The caller obtains the current
`h8_reference_law`, selects an advertised `outcomeId`, and sends the
entire advertised law back. The mechanics owner validates the boundary
version, event/law identity, probabilities and selected support before
applying it. The local fast Reference Nature path is unchanged.

The existing H15 fixed-fixture adapter now accepts
`use_declared_chance_laws=True`. This changes the fixed-fixture prior
identity to `h15-h8-fixed-fixture-declared-chance-law-v1` and sends
individual decisions through the emulator-owned law bridge. The caller
uses independent seeded random numbers to draw exact integer-weighted
law outcomes; every resulting player-visible prefix is still checked by
the independent reference conditioning sampler.

**Scope:** This is an executable *fixed-fixture* interface test, not a
certification that full native H10/H14 entry states can be reconstructed
with a fair hidden-state posterior. The emulator still needs a native
combat-root constructor that retains authentic public deck/HP/relic/
potions/enemies without retaining unexplained hidden RNG futures. This
is the remaining blocker to native H15 PUCT target generation.

The executable native adaptive round planner
`tools/plan_h15_native_adaptive.py` is now available. It consumes the
actual `results/h15-native-source-long/train.jsonl` recipes, uses the
existing learning-progress weights and deterministic fixed probes, and
writes a stable `round-NNNN/train.jsonl` cohort, `probes.jsonl`,
and `manifest.json`. After each completed round's probe fights, pass
one JSONL result per probe with `source_seed`, `entry_hash`,
`outcome`, and realized H9 `reward`. `advance` validates every probe,
updates the curriculum and atomically saves `state.json`. This planner
does not launch PUCT or claim those combat recipes are fair search roots.

Example fish commands:

```fish
set emu ~/projects/sts2-emulator
set ai ~/projects/sts2-ai
git -C $emu fetch origin
git -C $emu switch research/h15-native-chance-law-20261030
dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release
dotnet test $emu/tests/Sts2Emulator.Core.Tests/Sts2Emulator.Core.Tests.csproj -c Release --filter FullyQualifiedName~H15ChanceLawBoundaryTests
git -C $ai switch research/h15-combat-search-guided-20261029
cd $ai
source .venv/bin/activate.fish
python -m pytest -q tests/test_h15_h8_reference.py tests/test_h15_native_adaptive_round_plan.py
python -u tools/collect_h15_h8_reference.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --emulator-dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll \
    --h9-checkpoint results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --output-root results/h15-declared-law-smoke \
    --scenario mawler-a10 --deck silent-starter-v1 \
    --use-declared-chance-laws --simulations 4 --max-searches 1 \
    --max-candidates 16384 --device cuda --execute
python -u tools/plan_h15_native_adaptive.py plan \
    --train-corpus results/h15-native-source-long/train.jsonl \
    --study-root results/h15-native-adaptive \
    --round 0 --combats 256 --probes-per-family 3 \
    --teacher-checkpoint results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt
```

Next engineering gates: native combat-root reconstruction; stochastic-site
coverage and public-history conditioning; batched GPU evaluation; and
a resumable multi-worker PUCT collector/optimizer that consumes the
planned cohort without dropping most supervision samples.


## M6 — native-structure combat entry candidates (diagnostic only)

**Important correction following the 12-entry corpus probe.** The
`reset_native_overgrowth` environment has authentic 16-floor map and
Neow/room structure, but it is built by
`PrototypeNativeOvergrowthRunFactory.Create` and **does not attach
native combat RNG**. Therefore the original H15 native-entry candidate
constructor rejected all 12 exact-replayed corpus entries on the
native-RNG-ownership precondition. The corpus remains valid for its
declared native-structure **prototype-RNG simulator**; it is not a
native-RNG combat corpus.

The candidate constructor now recognizes two modes and reports them
separately: `prototype-combat-rng` (the existing native-structure corpus)
and `native-combat-rng` (source-audited, explicitly enabled v111 combat
RNG). It independently initializes available provider state and
unresolved draw order, preserves combat mechanics, and checks identical
public observation and legal action menus. It does not invent absent
native RNG owners or label prototype source entries as native RNG.

**Strict limitation:** Both modes remain *unverified experimental
candidates*, not a posterior conditioned on the full public history.
The CLI operation `h15_native_combat_candidate_probe` discards the
candidate and returns no state handle. `FairSearchCapabilityId` is
always `unavailable`; no native or prototype fair PUCT supervision can
be collected from this endpoint.

The Python binding verifies that the advertised candidate model ID
agrees with `providerClass`. The corpus tool
`tools/probe_h15_native_candidates.py` requires exact hash and public
observation replay on the descendant emulator before each probe, and
reports candidate counts by RNG ownership. The next milestone should
attach controlled Chance Law continuation to the **existing prototype
combat training simulator**, audit required chance sites, and establish
public-history conditional sampling. Native-RNG combat fidelity can
remain an independent line of work.

Fish validation commands:

```fish
set emu ~/projects/sts2-emulator
set ai ~/projects/sts2-ai
git -C $emu fetch origin
git -C $emu switch research/h15-native-chance-law-20261030
dotnet build $emu/src/Sts2Emulator.Cli/Sts2Emulator.Cli.csproj -c Release
dotnet test $emu/tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H15NativeCombatEntryCandidateV1Tests|FullyQualifiedName~H15ChanceLawBoundaryTests'
cd $ai
git fetch origin
git switch research/h15-combat-search-guided-20261029
source .venv/bin/activate.fish
python -m pytest -q \
    tests/test_h15_h8_reference.py \
    tests/test_h15_native_candidate_probe.py \
    tests/test_h15_native_adaptive_round_plan.py
python -u tools/probe_h15_native_candidates.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --source-revision 2c46c703a30875d6ef84fcbe0a0511fec9c74dcd \
    --train-corpus results/h15-native-source-long/train.jsonl \
    --output results/h15-native-source-long/root-candidate-probe-v2.jsonl \
    --limit 12
```

Interpretation: a successful probe verifies that an exact native-structure
simulator entry can be structurally reconstructed under an explicitly independent
experimental chance prior without changing the current public choice.
It does NOT certify subsequent stochastic transitions or their joint
conditional probabilities. The next milestone is to attach controlled
Chance Law continuation to this entry-state representation, audit
every remaining stochastic site, and establish full public-history
conditional sampling before generating long-run H15 PUCT targets.
