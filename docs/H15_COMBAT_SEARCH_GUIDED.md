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
