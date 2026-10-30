# H15 M1 — fair stochastic combat PUCT and policy/value distillation

**Current integration note (2026-10-30):** The statements below
are chronological H15 milestone records. We already have
`ReferenceNatureV111`, `H8LocalReferenceNatureV1`, the atomic
`ReferenceNatureSessionV111` family, and the AI's
`H8ReferenceNaturePosterior` conditioning independent Nature seeds
on complete *fixed-fixture combat-local* public history. The missing
component is a certified posterior for authentic native-entry histories,
**not** a generic sampler from scratch. H15's M7–M9
`H15ControlledCombatSessionV1` is an experimental narrow
continuation branch. Its duplicate Python `_sample_law` has now been
removed on the independent H15 research branch: reference choices
use the emulator's `ReferenceNatureV111.SampleValidated` with an
atomic reference cursor and receipt, while explicit external choices
remain available for scripted/learned Nature. Audited card and
potion effects also use one `SemanticPlayerActionEngineV111`.
Its independent candidate still is **not** conditioned on the
authentic source public history, its law provenance remains
`UNVERIFIED`, and its supported continuation sites remain narrower
than the full semantic combat provider. The H10-based 120-round A10 PPO launch belongs to the
existing native-*structure*, **prototype-combat-RNG** training
distribution; its eventual outcomes do not prove native-game parity.
See the emulator
[canonical stochastic architecture](https://github.com/nn971/sts2-emulator/blob/research/h15-experimental-search-20261030/docs/UNIFIED_STOCHASTIC_ARCHITECTURE.md).

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


## M7 — experimental real-run controlled end-turn continuations

The emulator research branch provides `H15ControlledCombatSessionV1`,
constructed from an **exact-replayed real Act 1 native-structure/prototype-RNG
combat entry** via `H15NativeCombatEntryCandidateV1.Build`. It preserves
the original combat mechanics and independently initializes private
prototype-RNG and unresolved draw-order state. The pinned Act 1
weak/normal/elite/boss encounter mappings are reused for the semantic
intent selector; unidentified encounters fail closed.

This is the first *executable* real-run continuation slice, using the
existing `SemanticEndTurnEngineV111.BeginFromPlayerActionOrTerminal`
and `Resolve` mechanics under the controlled chance guard. It accepts
a real legal `end_turn` action and exposes typed `ChanceLawV111`
outcomes for: automatic enemy-action index, next-turn enemy intent,
draw-card identity, and random target. The selected outcome is checked
against the exact current law and converted to the source mechanics
ticket/index; mismatched laws or unsupported sites are rejected.
Source-verified primitive Native grids are deliberately **downgraded**
to `UNVERIFIED` and, where necessary, to raw/uniform experimental
weights because the origin uses prototype combat RNG.

The separate JSONL namespace has:
`h15_experimental_begin` (original `state_handle` and independent
seed), `h15_experimental_status`,
`h15_experimental_end_turn` (legal `action_id` and exact version),
`h15_experimental_choose` (the full current declared `chance_law`,
selected `outcome_id`, and version), and
`h15_experimental_release_many`. The session has no implicit RNG
fallback and updates its handle only after a complete successful
transition. It is independent of generic state, H8, and certified
search handles.

**Limitations:** This slice supports end-turn player actions only.
Card/potion plays, other automatic nested chance types, exact full
public-history conditional distributions, and broader encounter
source coverage still need migration. It is **not** a fair
`FairContinuationSampler`; the session and bridge advertise
`fairContinuationCapability: unavailable`. No native fair PUCT
targets are generated.

Run the focused validation on the current branches:

```fish
set emu ~/projects/sts2-emulator
set ai ~/projects/sts2-ai
git -C $emu pull --ff-only
dotnet test $emu/tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H15ControlledCombatSessionV1Tests|FullyQualifiedName~H15NativeCombatEntryCandidateV1Tests|FullyQualifiedName~H15ChanceLawBoundaryTests'
cd $ai
git pull --ff-only
python -m pytest -q \
    tests/test_h15_controlled_turn_protocol.py \
    tests/test_h15_native_candidate_probe.py
python -u tools/probe_h15_controlled_turns.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --source-revision 2c46c703a30875d6ef84fcbe0a0511fec9c74dcd \
    --train-corpus results/h15-native-source-long/train.jsonl \
    --output results/h15-native-source-long/controlled-turn-coverage-v1.jsonl \
    --limit 12
```

The coverage report distinguishes successful next-player/exit
transitions, explicitly unsupported boundaries, and rejected
continuations; any `UNSUPPORTED_CONTROLLED_CHANCE` errors identify
unowned source RNG sites requiring migration. Only proceed to fair
PUCT training when an independent public-history-conditioned Nature
model has been implemented and validated.


### M7 first corpus coverage: The Kin source-provenance correction

The first 12-entry controlled-turn report
(`controlled-turn-coverage-v1.jsonl`) reached a new player decision for
**11/12** source-exact-replayed entries, resolving **55 draw-card** and
**5 intent** Nature selections across those 11. No enemy-action or
random-target boundary was exercised by this deterministic first-outcome
probe. Zero fair PUCT search roots were generated.

One boss entry rejected at the first end-turn with
`InvalidOperationException: Unknown native KinFollower starting slot.`
The root was a provider-semantics mismatch: the native `TheKinBoss`
intent preprocessing expects `slot1`/`slot2`, while the
`native-overgrowth` corpus uses the prototype-RNG combat whose
followers already carry `quick`/`dance` slot names.

The experimental controlled session now uses
`h15-prototype:<encounter-id>` as the intent selector's non-native
context, preserving the ordinary prototype source behavior. Native
encounter identities remain a separate coverage gate. A focused
`PrototypeTheKinTests.H15PrototypeContinuationKeepsTheKinFollowerSlotSemantics`
regression verifies both followers' intent results against the
ordinary prototype selector. This fix does not change game mechanics,
the recorded source corpus, or the fair-search capability.

Rerun the **same** 12-entry probe after fetching both branches, using
`--output results/h15-native-source-long/controlled-turn-coverage-v2.jsonl`
(the prior report is intentionally immutable). The goal is to verify
that this one boss continuation now advances; further unsupported
chance sites should be recorded rather than bypassed.


## M8 — audited real player actions (experimental)

The unchanged 12-entry `controlled-turn-coverage-v2.jsonl` report
advanced all 12 genuine native-structure combat entries through
one end turn: **60 draw_card** and **5 enemy_intent** Nature selections,
with zero rejections. This validates the narrow source end-turn route,
not a fair conditional continuation posterior.

The existing `H15ControlledCombatSessionV1` now exposes
`PlayerAction(actionId)` to execute legal `play_card` or `use_potion`
actions at real combat player boundaries. Card plays reuse
`SemanticPlayerActionEngineV111.Step` for its audited
supported-card allowlist, physical/semantic hidden-deck invariant,
and controlled RNG guard. The first potion gate covers source-implemented,
deck-neutral block/dexterity/energy/strength/explosive/fire/poison/weak
effects; unknown potions, arbitrary card-draw hooks, nested player
selections and non-migrated stochastic sites fail closed.
The existing `EndTurn` and `Choose` flows remain intact, and
each successfully committed action increments the session version.
JSONL `h15_experimental_player_action` and Python
`h15_experimental_player_action` preserve the same isolated, unverified
provenance contract. Client-side requests reject invalid versions.

The new `tools/probe_h15_player_actions.py` replays exact native-structure
corpus roots, attempts an audited legal card/potion action, checks
that rejected actions leave the session unchanged, then takes an
ordinary end turn and resolves declared Nature events. It writes
specific accepted/rejected actions and unsupported chance-site
diagnostics. The first action exercised in each entry depends on
its legal action order; this tool is a coverage gate rather than
a policy benchmark.

Fish verification:

```fish
set emu ~/projects/sts2-emulator
set ai ~/projects/sts2-ai
git -C $emu pull --ff-only
dotnet test $emu/tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H15ControlledCombatSessionV1Tests|FullyQualifiedName~PrototypeTheKinTests'
cd $ai
git pull --ff-only
python -m pytest -q tests/test_h15_controlled_turn_protocol.py
python -u tools/probe_h15_player_actions.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --source-revision 2c46c703a30875d6ef84fcbe0a0511fec9c74dcd \
    --train-corpus results/h15-native-source-long/train.jsonl \
    --output results/h15-native-source-long/player-action-coverage-v1.jsonl \
    --limit 12
```

Every candidate remains experimental and the session explicitly
advertises `fairContinuationCapability: unavailable`. No fair
PUCT labels or long-run search supervision are produced by M8.


## M9 — frozen-policy multi-action, multi-turn experimental combat

The M8 `player-action-coverage-v1.jsonl` report accepted one
real card play and an end turn in **12/12** exact-replayed
A10 native-structure combat entries. It resolved **60 draws**
and **5 enemy intents**, and returned to the next player
boundary every time. One entry rejected its initial card-action
candidate and then succeeded on another. This probe did **not**
exercise potions, repeated card plays, or multiple turns.

M9 extends the experimental H15 status response with
`observationPolicyId`, `observationPayloadJson` (the **public**
fair-policy projection), `combatTurn`, and `exitHp`. Private
provider RNG and hidden state stay inside the emulator.

The new `sts2_ai.training.h15_experimental_rollout` module runs
the actual frozen H9/H15 combat policy on **only** that public
observation and legal action menu. It ranks legal actions by
neural logits and invokes version-checked H15 real player-action
or end-turn endpoints. Unsupported actions are recorded, and
the next ranked candidate is tried at the **same unchanged**
state, checked by session version and public hash. Such a
fallback is an explicit coverage diagnostic, **not** an
on-policy RL sample.

At each declared Nature boundary the diagnostic sampler uses
the integer `ChanceLawV111.Outcomes` weights, rejecting
non-`UNVERIFIED` laws and inconsistent support totals.
Every experiment saves counts of real card plays, potions,
ended turns, sampled Nature sites, policy decision ranks,
rejected actions, and final combat outcome/exit HP when
available. Limits on player actions, turns and Nature
decisions yield **censored** reports, never invented wins
or losses. Unmodeled continuation boundaries end with
explicit unsupported status. There are **no fair search
roots and no training targets** from this workflow.

Fish commands on the H15 research branches:

```fish
set emu ~/projects/sts2-emulator
set ai ~/projects/sts2-ai
git -C $emu pull --ff-only
dotnet build $emu/src/Sts2Emulator.Cli -c Release
cd $ai
git pull --ff-only
python -m pytest -q \
    tests/test_h15_controlled_turn_protocol.py \
    tests/test_h15_experimental_rollout.py

python -u tools/run_h15_experimental_policy_rollouts.py \
    --emulator-root $emu \
    --emulator-revision (git -C $emu rev-parse HEAD) \
    --source-revision 2c46c703a30875d6ef84fcbe0a0511fec9c74dcd \
    --train-corpus results/h15-native-source-long/train.jsonl \
    --teacher results/h9-adaptive-totalhp-r40-r250/models/player-r0250.pt \
    --device cuda \
    --output results/h15-native-source-long/multiturn-policy-64-v1.jsonl \
    --limit 64 \
    --max-player-actions 256 \
    --max-turns 32 \
    --max-nature 1024
```

This is a substantive transition toward H15 combat-search support,
but its stochastic model remains independent/unverified. The
next step after this rollout is to repair the *highest-coverage
blocking combat effect or Nature site* reported by real policy
decisions, and ultimately establish a history-conditioned fair
continuation sampler before launching PUCT supervision.


## M10 — training begins on the existing on-policy native PPO track

**Track A: usable neural training now.** H15's fair search sampler is still
unavailable, but the existing H10 native-combat PPO trainer can learn from
actual A10 emulator fights with a public-only actor. It collects fresh on-policy
combat trajectories and uses source terminal H9 total-nonminion-HP reward.
No counterfactual hidden-state posterior is required. Do **not** relabel the
current H15 `independent-candidate` diagnostic rollouts as fair or PPO
training data.

`tools/run_h15_native_ppo_long.fish` is the pinned execution entry point:
- Builds the research emulator, runs the existing H10 PPO/corpus tests.
- Collects new source-run-distinct **256 training / 64 heldout** A10
  macro-generated source-run populations under the *current exact emulator
  commit*. The prior 11,000/1,418 H15 recipes refer to an older emulator SHA;
  do not silently reinterpret or retag them.
- Starts the existing H10 optimizer, warm-started from frozen H9 R250, for
  **120 rounds × (64 native + 32 synthetic)** on-policy combats,
  **15 native workers** and CUDA learner.
- Writes incremental collector logs, `train.log`, atomic
  `optimizer-checkpoint.pt`, `progress.json` per round, and model exports
  every ten rounds. Rerunning on the same commit/input fingerprint resumes
  only when a checkpoint exists. An emulator update generates a separate
  experiment directory, and the trainer rejects mismatched corpus/dll/model
  fingerprints.
- Keeps all generated artifacts inside `~/projects/sts2-ai/results/`;
  creates no additional project checkout or worktree.

```fish
cd ~/projects/sts2-emulator
git switch research/h15-native-chance-law-20261030
git pull --ff-only
dotnet test tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H15ControlledCombatSessionV1Tests|FullyQualifiedName~PrototypeTheKinTests'

cd ~/projects/sts2-ai
git switch research/h15-combat-search-guided-20261029
git pull --ff-only
source .venv/bin/activate.fish
fish tools/run_h15_native_ppo_long.fish
```

The launcher runs **on the user's CUDA workstation**. Remote repository
changes alone do not launch GPU training. Existing H10 checkpoint verification,
exact source replay, and corpus disjointness remain authoritative. A failed
source-mechanics transition should be fixed rather than masked as a training
loss or substituted synthetic episode. Compare final and baseline on the
new heldout corpus using the existing H10 evaluation tool once a genuine
training checkpoint is available.

**Track B: H15 stochastic PUCT.** M9's 64-combat frozen-policy diagnostic
can run separately on the old exact-replay H15 corpus. The source-compatible
semantic card adapter now accepts a genuine player-card kill that enters
the postcombat reward/defeat boundary, with only previously audited reward
RNG sites delegated to the source engine. The regression test constructs a
test-only low-HP combat end from a real native-structure entry. Additional
unsupported card/draw/potion sites and full public-history-conditioned
Nature distributions remain future work. `fairContinuationCapability`
continues to advertise `unavailable`, so the H15 PUCT collector and
search-policy distillation remain disabled.


## M11 — canonical provider reuse and recorded architecture mistake

The independent research branches `research/h15-experimental-search-20261030`
(leave the pinned PPO branches alone) remove two H15-specific implementations:

1. The Python diagnostic no longer implements `_sample_law` or holds a
   separate per-event `random.Random` sampler. The C# H15 session now
   owns a forkable `ReferenceNatureStateV111` cursor, initialized through
   `ReferenceNatureRunCursorV111.Begin`. One
   `h15_experimental_reference_choose` transaction uses
   `ReferenceNatureV111.SampleValidated`, resolves the existing typed
   `SemanticEndTurnEngineV111` continuation, and publishes a law/event/
   outcome/provider receipt. Failed continuations do not advance the
   published mechanics state or Nature cursor. The original
   `h15_experimental_choose` remains for externally scripted/learned
   outcomes and does not consume the reference cursor.
2. The duplicated H15 potion allowlist and draw-pile checks were deleted;
   the restricted card/potion player path is now owned by
   `SemanticPlayerActionEngineV111` with one source replay/guard/
   semantic-deck validation contract.

The root comparison tool continues to report `UNVERIFIED`,
`fair_search_roots=0` and `training_targets=0`. Matching seeds
across root-action candidates now drive the *same emulator-owned
ReferenceNature provider* and candidate-state initialization;
there is no second Python chance stream.

The underlying `H15NativeCombatEntryCandidateV1` remains an
independently reseeded, **native-structure/prototype-combat-RNG**
candidate, not a history-conditioned true native-run posterior.
Similarly the narrower `SemanticEndTurnV111` H15 adapter has not
been magically converted into the larger
`SemanticPilotCombatV111` reference-session family. Those are
distinct source-mechanics admission/conditioning tasks, and we should
reuse existing continuation owners when expanding coverage.

The root-cause retrospective and mandatory provider-inventory checklist
live in [emulator ADR 0005](https://github.com/nn971/sts2-emulator/blob/research/h15-experimental-search-20261030/docs/adr/0005-h15-reference-nature-reuse-retrospective.md).
The AI repository's `AGENTS.md` requires reading them before future
stochastic implementation work.

**Local validation after the existing long PPO experiment completes:**

```fish
set emu ~/projects/sts2-emulator
set ai ~/projects/sts2-ai

git -C $emu switch research/h15-experimental-search-20261030
git -C $emu pull --ff-only
dotnet test $emu/tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H15ControlledCombatSessionV1Tests|FullyQualifiedName~RandomIndexProvidersV111Tests'

git -C $ai switch research/h15-experimental-search-20261030
git -C $ai pull --ff-only
cd $ai
python -m pytest -q \
    tests/test_h15_controlled_turn_protocol.py \
    tests/test_h15_experimental_rollout.py
```

## M12 — remove the remaining H8 declared-law Python sampler

The fixed-fixture `H8ReferenceCombatBackend(use_declared_chance_laws=True)`
previously used a separate Python `random.Random` cursor and weighted roulette
in `_resolve_declared_laws`. This duplicate implementation has been removed.
Both H8 fixed-fixture reference modes now resolve Nature entirely inside the
emulator:

- The default fast mode uses existing H8 event-keyed
  `H8LocalReferenceNatureV1`, with `local_reference_seed`.
- The explicit declared-law mode starts `ReferenceNatureStateV111` with
  `declared_reference_seed`, queries the current typed `ChanceLawV111`, uses
  `ReferenceNatureV111.SampleValidated`, and resolves the selected outcome
  through `H8CombatRuntimeV1.ChooseChanceLawOutcome`.
  The H8 JSONL session owns the persistent immutable cursor across player
  decisions and forks; release disposes its cursor. External selection is
  rejected on an attached declared-reference session. Only the emulator
  sees stochastic outcome selection.
- The Python posterior still samples **independent candidate seeds** for
  the fixed-fixture prior and replays the entire combat-local public history
  for acceptance. Seed proposal in Python is distinct from sampling a
  Chance Law outcome, which now belongs exclusively to the engine.

These two H8 reference providers have distinct random-stream conventions.
The declared-law mode now uses emulator SplitMix64, replacing the previous
Python MT19937 stream. The declared probabilities themselves retain their
existing source-law provenance; the new stream does **not** claim replay
equivalence with old same-seed Python trajectories.

**Scope:** The same fixed-fixture prior and public-history conditioning
restrictions remain. This refactor introduces no native-structure posterior,
fair native PUCT labels, or evidence of policy improvement. The existing
H15 experimental native-entry interface continues to advertise
`fairContinuationCapability=unavailable`. The pinned 120-round PPO
experiment is unchanged.

Validation on the experimental branches (run separately from the pinned
PPO checkout):

```fish
dotnet test ~/projects/sts2-emulator/tests/Sts2Emulator.Core.Tests -c Release \
    --filter 'FullyQualifiedName~H8CombatRuntimeV1Tests'
cd ~/projects/sts2-ai
python -m pytest -q tests/test_h15_h8_reference.py
```

The C# regression checks typed-law provenance, provider journal,
persistent cursor progression, replay and stale-boundary rejection.
The Python regression checks that declared mode passes the seed
directly to the emulator and does not call a Python chance sampler.
End-to-end .NET/Python runtime validation is pending local execution.
