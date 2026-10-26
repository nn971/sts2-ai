# H6-E — richer macro policy (Silent Ascension 10)

Goal: test whether missing action-to-object attention and limited network
capacity explain H6-D's flat learning curve. Do not change the frozen H4
combat policy. The emulator stays at submodule pin
`0469424fc20679fd81c0d5e4ee8d9a484ec19108`.

## Architecture

This is a **new opt-in V19 strategy model**, format
`sts2-neural-policy-value-v19-deep-macro-attention`, selected with
`--strategy-encoding macro_deep`. The original V17/V18 serializers and action
features are unchanged. Only V19 uses the additional object pointers.

- Grouped deck, relic, potion, map-candidate, reward, and **public shop offer /
  event choice** objects are encoded as typed tokens.
- Action logits attend to selected objects, so two shop offers and two event
  choices can have both different features and different contextual targets.
- Retain first four-head relation block (H6-C/H6-D warm-start weights).
  Follow with an independently trained **eight-head** relation block.
- Per-object residual **4x feed-forward** expansion, 32 → 128 → 32.
  Residual gates for the new block and expansion start at zero.
- Keep global/target hidden width 32 and hash feature dimension 128 to match
  the frozen H4 combat checkpoint. This expands **strategy capacity only**;
  use the architecture flag, *not* an incompatible `--hidden 64`.
- Torch per-decision, batched PPO and dependency-free portable JSON inference
  must agree. Hidden RNG and unpublished event/bundle outcomes are not encoded.

## Reward schedule

The original failure-shaping objective is
`alpha * (0.7 * normalized_floor_progress + 0.3 * remaining_hp_ratio)`;
victory target is 1. Set `--win-anneal-threshold 1000000` to suppress effective
annealing over the experiment **without removing the annealing implementation**.
At 500 cumulative wins, alpha remains `0.4 * (1 - 500 / 1000000) = 0.3998`.
Losses still receive progress-shaped returns, not only terminal win/loss.

## Experiment

Use the completed H6-D-80 portable checkpoint as the warm start. A new PPO
optimizer checkpoint is created; **do not resume the old optimizer**. Use fish:

```fish
cd ~/projects/sts2-ai
git fetch origin
git switch research/h6e-deep-macro-attention-20261027
git pull --ff-only
git submodule update --init --recursive
source .venv/bin/activate.fish
mkdir -p results/nature

python -u tools/train_phase_split.py \
  --warm-start results/nature/h6d-semantic-a10.json \
  --freeze-combat --strategy-encoding macro_deep \
  --optimizer-method ppo --ppo-backend batched \
  --tactical-state-encoding enemy_instances \
  --combat-objective hp_preservation \
  --combat-advantage-baseline critic --hp-monotonic-weight 0 \
  --dimension 128 --hidden 32 --ascension 10 \
  --temperature-start 0.85 --temperature-end 0.65 \
  --temperature-decay-rounds 20 \
  --rounds 80 --episodes 128 --workers 15 \
  --win-anneal-threshold 1000000 --max-decisions 4096 \
  --eval-seeds 32 \
  --train-seed-prefix h6e-deep-a10-train-v1 \
  --eval-seed-prefix h6e-deep-a10-select-v1 \
  --checkpoint results/nature/h6e-deep-a10.pt \
  --output results/nature/h6e-deep-a10.json \
  --report results/nature/h6e-deep-a10-report.json \
  --build 2>&1 | tee results/nature/h6e-deep-a10-train.log
```

For an interrupted run, add `--resume` with the original flags and paths.
`--rounds` is the total target, not extra rounds.

## Evaluation

- Require local / CI parity, byte-identical frozen combat weights, and a
  native rollout with nontrivial shop/event actions before full training.
  A short rollout is a correctness check only, not a performance experiment.
- Compare H6-C-80, H6-D-80, H6-E-80 under **paired new held-out seeds**;
  use the established 2,560-run confirmation protocol if feasible.
- Track Act 1 clears, mean frontier progress, censored runs, boss HP for
  defeats, PPO KL/entropy, and zero/near-zero shop/event logit separation.
- If 80-round H6-E remains flat, inspect sampled **full run decision histories**,
  including map room types and connectivity, boss identity, actual offers,
  all action logits, purchases and skips before trying more capacity.
- Fidelity-inspector parity does not block training; separate emulator
  integration and native game fidelity concerns.
