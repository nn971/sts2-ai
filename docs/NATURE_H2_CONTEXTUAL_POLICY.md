# H2 contextual Nature (draw-only)

Implementation: `src/sts2_ai/nature/contextual_draw.py`. This experimental
policy **controls only unresolved physical card draws**; enemy intents remain
sampled from the emulator-declared Chance Law. The player is the same frozen
historical v8 model, served through the unchanged engine player-action menu.

## Data and interface

The emulator exposes `SemanticCombat.observe_for_nature()` at pending
`nature` boundaries (isolated emulator PR #180). This returns the same
serialized `PrototypeAiObservation` as the player's observation projection,
using `PrototypeAiEnvironment.ObservePublic` so no player legal action
generation is attempted at a Nature boundary. The serialized observation
contains public player HP, max HP, combat turn/energy/block, hand, discard,
exhaust and play piles, the count of cards in the hidden draw pile, enemy
species/HP/block/statuses/powers/visible next intents, player powers, relics,
potions, persistent deck cards and public run context. The *semantic Chance
Law* separately gives Nature the legal physical draw-instance IDs and card
model labels, whose identities are needed for it to select an outcome.

The source implementation **does not export the draw pile's exact hidden
order, internal RNG state, future results, snapshot internals, or player's
private inference state**. The binding verifies the Nature handle and
fingerprint match the pending request. Existing `semantic_observe` still
requires a player decision boundary; the new Nature-specific read is
read-only, and player policy uses exactly its old fair interface.

## Model

`encode_public_observation` recursively feature-hashes the public
observation into 256 bounded scalar buckets. Numeric values contribute both
scaled magnitudes and bins; visible collections of cards, enemies, effects,
statuses and relics are pooled, with stable instance IDs excluded. Entity
permutations do not affect encoded context.

A 24-dimensional tanh state network `h = tanh(W s)` supplies state
context. For each physical candidate `i` with card model label `c_i`,
a learned candidate vector `v[c_i]` and bias `b[c_i]` produce
`logit_i = v[c_i] · h + b[c_i]`. Multiple eligible physical instances of
the same model receive equal logits but remain **separate legal outcomes**.
The policy is a true contextual interaction between public state and
card candidate, not a static per-card lookup table.

A legal-support softmax is blended with the emulator's reference draw law:
`q = (1 - mix) p + mix softmax(logits)`. Default `mix=0.75`, so all
reference-supported physical cards remain available with positive mass.
Optimizer: full-combat REINFORCE with leave-one-episode-out return baseline,
on-policy checkpointed exact q, analytic gradients through the mixture,
norm clip 5.0 and learning rate 0.03 for initial H2. The *opponent*
minimizes the frozen player's terminal HP-preservation score.
No native RNG calibration or original-game fidelity is required to run.

This compact network is a first contextual baseline, not a large transformer.
The 256-bucket hashing and pooled entity representation deliberately
compress some ordering/detail, so further representation work may be
needed if the H2 results justify it. The frozen v8 player is unchanged.

## Provenance and assessment

The normal A10 difficulty screen (`report(4).json`) used 768 source worlds
× two reference-draw replicas, with all 1,536 v8 combats won, but mean HP
score 0.5833 and paired score differences in 88.67% of source worlds.
All 12 encounter/deck cells remain in H2; do not condition on victory
or compare H2 against the prior weak-only score as if they were matched
worlds.

Source-level and C# integration tests check: response on an actual pending
Nature boundary, read-only public observation, no hidden order/RNG fields,
state-varying policy outputs at unchanged draw support, permutation
invariance, legal support, deterministic replays, state-conditioned
REINFORCE updates and restored checkpoints.

The original tabular model remains as `--nature-model tabular`.
It should be compared on matched held-out worlds against
`--nature-model contextual`, controlling seeds, deck and enemy pools.
`nature_grad` is the global gradient norm *before* optimizer clipping;
it does not by itself measure training success.

## Run from existing H2 worktree

After the draft integration check passes, use the **existing**
`~/projects/sts2-ai-h2` worktree; no extra worktree is needed.

```fish
cd ~/projects/sts2-ai-h2
git status --short
git fetch origin
git switch --detach origin/research/nature-contextual-draw-h2-20261021
git submodule update --init --recursive emulator
source .venv/bin/activate.fish
dotnet build emulator/Sts2Emulator.sln -c Release

set model ~/projects/sts2-ai/results/ppo-v8-batched-40/models/stage-0120.json
set dll (pwd)/emulator/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set bindings (pwd)/emulator/bindings/python
set rev c254bd99816ac2b89b350ec132fe1cfbb27dfac6

# First smoke the C# Nature-read path and replay.
python -u tools/probe_nature_h2_combat_profiles.py \
    --nature-model contextual \
    --scenarios mawler-a10,cubex-a10,vine-shambler-a10 \
    --emulator-dll $dll --emulator-bindings $bindings \
    --emulator-revision $rev

# Then test independently initialized Natures, with balanced contexts.
for seed in 29 101 173
    python -u tools/train_draw_only_nature_h1.py \
        --nature-model contextual --nature-lr 0.03 \
        --player-model $model --emulator-dll $dll \
        --emulator-bindings $bindings --emulator-revision $rev \
        --player-action-scope all-legal-v1 \
        --scenario-pool mawler-a10,cubex-a10,vine-shambler-a10 \
        --deck-pool silent-starter-v1,silent-strike-heavy-v1,silent-defend-heavy-v1,silent-poison-kit-v1 \
        --training-seed $seed --actor-seed 31337 --reference-seed 23387 \
        --workers 15 --rounds 40 --episodes-per-round 256 \
        --eval-episodes 1000 \
        --output-root results/nature/h2-a10-contextual-v8-seed$seed \
        --execute 2>&1 | tee h2-a10-contextual-v8-seed$seed.log
    if test $pipestatus[1] -ne 0
        break
    end
end
```

An interrupted seed may use `--resume` with the exact same arguments.
The evaluator shares the same enemy/deck contexts and source seeds across
learned and reference draw arms, reporting paired intervals and joint
enemy/deck strata. This is an **experiment to be run locally**, not a
claimed training outcome. C# CI smoke players are synthetic and do not
validate the frozen-v8 adversarial learning result.
