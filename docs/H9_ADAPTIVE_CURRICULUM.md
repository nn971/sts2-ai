# H9 adaptive combat curriculum — minimal learning-progress controller

Branch: `research/h9-adaptive-progress-20261029`. This is opt-in to
`tools/train_h7_reference_combat.py --curriculum h9-rich --adaptive-rich`;
existing H9 v2 source generation, H7/H8 training defaults, and the model
architecture stay unchanged.

## Algorithm

Each of the 21 supported encounter IDs is one *family*. We use the existing
versioned H9 total non-minion encounter HP reward; no hand-coded
difficulty/HP/deck taxonomy is involved. A fixed set of six training-pool fixtures per family is evaluated
with greedy player actions at initialization and every ten PPO rounds.
Reference-Nature seeds and fixture IDs are identical between evaluations.

For family `i`, let `m_i(t)` be the fixed-probe mean reward. Define

```text
delta_i(t) = abs(m_i(t) - m_i(t-1))
progress_i(t) = 0.5 * delta_i(t) + 0.5 * progress_i(t-1)
p_i(t) = 0.3/21 + 0.7 * (0.01 + progress_i(t)) /
                         sum_j(0.01 + progress_j(t))
```

At startup, all 21 family probabilities are equal. A probability floor avoids
permanently eliminating difficult or previously mastered encounters.
Regression and improvement both increase temporary sampling priority;
stably unwinnable encounters gradually fall back toward the uniform floor.
The smoothed differences are *measurement feedback*, not PPO sample
importance weights, and never modify Nature's event laws.

Each round, choose 256 families from this distribution using a deterministic
round-indexed sampler, and select fixtures from their existing 1,024-member
training population. Within a family, fixture indices are sampled without
replacement until exhausted. Multiple episodes may use the same fixture
with distinct source identities; no zero-shot indices are sampled.
Learning-progress probes use another fixed and distinct source identity.

The optimizer checkpoint stores the feedback state (last means, smoothed
progress, weights, evaluation counter). All sampling is deterministically
derived from study seed, round index, fixture index, and persisted controller
state. Resuming with a changed adaptive configuration fails fingerprint
validation. The report logs per-round family allocations and per-probe reward
changes and weights.

## Run 80-round R35 continuation (fish)

This is a *new* run with a new optimizer, initialized from frozen rich R35.
Keep the previous `results/h9-overgrowth-rich-v2-r40` untouched.
It uses the same 15-worker H8 backend, 256 episodes/round, and sampled
terminal replay auditing (1 per 16 episodes within each family).

```fish
cd ~/projects/sts2-emulator
git fetch origin
git switch fix/total-nonminion-encounter-hp-20261029
git pull --ff-only
dotnet build src/Sts2Emulator.Cli -c Release

cd ~/projects/sts2-ai
git fetch origin
git switch research/h9-total-hp-reward-20261029
git pull --ff-only
source .venv/bin/activate.fish

pytest -q tests/test_nature_h4_reward.py tests/test_h9_adaptive_curriculum.py tests/test_h9_rich_curriculum.py

set emu ~/projects/sts2-emulator
set dll $emu/src/Sts2Emulator.Cli/bin/Release/net9.0/Sts2Emulator.Cli.dll
set rev (git -C $emu rev-parse HEAD)

python -u tools/train_h7_reference_combat.py \
  --emulator-dll $dll --emulator-bindings $emu/bindings/python \
  --emulator-revision $rev \
  --output-root results/h9-adaptive-rich-r35-r80 \
  --initialize-from-model results/h9-overgrowth-rich-v2-r40/models/player-r0035.pt \
  --curriculum h9-rich --structured-curriculum-seed 20261028 \
  --adaptive-rich --adaptive-interval 10 --adaptive-probes-per-encounter 6 \
  --rollout-backend h8-native --actor-backend shared \
  --actor-device cuda --actor-collation packed --local-reference-nature \
  --device cuda --lr 0.00003 --rounds 80 --episodes-per-round 256 \
  --workers 15 --eval-every 10 --eval-episodes 256 \
  --train-terminal-replay-every 16 --max-boundaries 1200 --execute
```

To resume a stopped run, rerun the identical command with `--resume`.
Do not change the output directory or any fingerprinted parameter.

## Evaluation discipline

The fixed probes belong to the **training** structural pool (indices
0–1023). They are part of the adaptive training procedure, and are not
unseen generalization evidence. The original 256 zero-shot fixtures
(indices 1024–1279) have already been consulted to select rich R35 and
must not be fed into this controller. For the next final comparison,
generate a fresh structural test partition or, preferably, naturally
reachable native A10 combat entries.

The added probe overhead is 21 × 6 = 126 fights at initialization and
every ten rounds. For 80 rounds this is 1,134 probe fights in addition to
20,480 training combats and scheduled selection evaluations.

## Future combat–macro integration

The sampler accepts public source IDs and terminal scalar rewards only.
It is independent of the Transformer and of legal-action encodings.
The existing `SharedObjectTransformer` already has shared object
representations and distinct combat/macro heads; future macro training
can reuse the feedback mechanism with appropriately scaled macro
reward, once the map/action representation issues are resolved.


## V3 terminal reward (2026-10-29)

The follow-up HP fix is documented in
`docs/H9_TOTAL_ENCOUNTER_HP_REWARD.md`. Non-minion death-summoned
stages and all non-minion opponents contribute to the same encounter
HP budget, including future stages before spawn. Minions are
excluded; the remaining HP and total are snapshotted at defeat by
the emulator. This is a *new reward version*: historical H9 rich
R35/R40 models can initialize fresh training, but earlier reward
scores are not numerically comparable.
