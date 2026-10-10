# Next training: paired public-intent v5 versus v6 ablation

## Purpose and isolation

This is a controlled **representation test**, not the proposed future
full-scale PPO/GAE run. The experimental sts2-ai branch
`agent/committed-intents-base-damage-v6` pins the separately tested emulator
commit `7356eb3c89a55ec30c7dc567bb7100f426face1f` (all 803 emulator
tests passed) without modifying either repository's main branch or any
ongoing emulator feature branch.

In that emulator revision, enemy moves are committed before the player
acts, the enemy executes the preselected move, and the public observation
contains up to three optional enemy fields:

- `intent_base_damage`: per-hit base damage after act/ascension scaling,
  before enemy/player powers or statuses.
- `intent_damage`: current per-hit damage after visible Strength, Weak,
  Vulnerable, and applicable damage caps; recalculated after every action.
- `intent_hits`: count of attack hits. All damage values are before Block.

The v5 `relational` encoder already consumes `intent_damage` and
`intent_hits`; it is preserved **byte-for-byte in feature semantics** by
the v6 patch. The new separately versioned
`--tactical-state-encoding relational_damage` additionally exposes
base damage, its difference from modified damage, and per-enemy bound
attack threat tokens. The exported model format is
`sts2-neural-policy-value-v6-relational-damage-tactical`.
Both current and base threat stay associated with the announcing enemy.
No private RNG or unknown concealed intents enter features.

The v6 categorical hash salt and the number of reserved numerical
coordinates differ from v5, so warm-starting from an older model resets
the tactical **state projection and value head**, while preserving
compatible tactical action/policy parameters and the strategy head.
Do NOT resume an old v5 checkpoint as a v6 checkpoint.

## Setup (fish, Ubuntu WSL)

```fish
cd ~/projects/sts2-ai
git fetch origin agent/committed-intents-base-damage-v6
git switch --track origin/agent/committed-intents-base-damage-v6
git submodule update --init --recursive
git -C emulator rev-parse HEAD
source .venv/bin/activate.fish
mkdir -p results
```

The submodule SHA should be
`7356eb3c89a55ec30c7dc567bb7100f426face1f`.
The backend verifies that the gitlink and submodule checkout are equal.
Do **not** manually check out emulator/main or a different emulator
branch inside this AI experiment.

Build/test the pinned emulator before training:

```fish
dotnet test emulator/Sts2Emulator.sln -c Release
```

For a short end-to-end smoke run, run with `--rounds 2 --episodes 8
--workers 4 --eval-seeds 8` first, using distinct output filenames.

## Matched experiment

Use the *same* v3/legacy warm-start model for both modes, not an
already-v5-trained model. That ensures **both** runs reset the
incompatible tactical state projection, avoiding an unfair v5 head
start. The prior stronger legacy model was used in the earlier comparisons.
If your file is named differently, substitute the correct path.

```fish
cd ~/projects/sts2-ai
source .venv/bin/activate.fish

for encoding in relational relational_damage
    python -u tools/train_phase_split.py \
        --warm-start results/legacy_episode_sum-model.json \
        --tactical-state-encoding $encoding \
        --combat-objective hp_preservation \
        --combat-advantage-baseline leave_one_run_out \
        --loss-normalization legacy_episode_sum \
        --rounds 12 --episodes 32 --workers 15 \
        --eval-seeds 64 \
        --eval-seed-prefix "intent-damage-ablation-20261010" \
        --learning-rate 0.001 \
        --checkpoint "results/intent-$encoding-checkpoint.pt" \
        --combat-samples-dir "results/intent-$encoding-samples" \
        --output "results/intent-$encoding-model.json" \
        --report "results/intent-$encoding-report.json" \
        --build
end
```

Both models will be trained with the **same emulator revision**, the
same run-seed schedule and learning hyperparameters, the same reward
and baseline, and identical fresh heldout evaluation seed strings.
Watch the per-round stderr stream for win count, combats won/lost,
net HP lost per victorious combat, tactical value MSE, and heldout
frontier/boss performance. Potion use is still diagnostic only;
optimizing a future HP/potion tradeoff is explicitly deferred.

**Interpretation:** With 12 rounds of 32 episodes, each policy head
still receives only **12 optimizer steps**. No Act 1 wins or a weak
difference between v5 and v6 does not establish that the representation
is useless; these runs test integration and early learning signal, not
convergence. Boss comparisons conditional on reaching a boss may be
selection-biased; use paired seed comparisons.

Our next large training milestone should replace combat REINFORCE
with action-level actor-critic PPO/GAE and a genuine entity-aware
(enemy/card/power) encoder. That should precede attempting hundreds
of rounds of the current tiny network.

## Restore AI main after this isolated experiment

```fish
git switch main
git submodule update --init --recursive
```

This restores the AI main branch's own emulator pin. Do not merge
the draft emulator review PR into ongoing development as part of
this experiment.
