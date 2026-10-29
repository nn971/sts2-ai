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
