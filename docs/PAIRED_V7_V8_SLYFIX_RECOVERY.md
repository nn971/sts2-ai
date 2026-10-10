# Paired v7-v8 experiment: recovery after Sly auto-target failure

The previous paired run used `agent/instance-aware-combat-v8` and pinned
emulator `f9fcf270...`. Results uploaded in
`ppo-v7-v8-enemy_instances.zip` show:

- v7 `public_resources` trained 9/20 rounds, 29 victories in 288
  episodes. During training round 10, seed
  `ppo-v7-v8-controlled-train-9-0`, decision 199, `select_cards`
  hit `NotSupportedException` for Sly Snakebite.
- v8 `enemy_instances` completed 20 rounds, 49 victories in 640
  episodes and 2,556 PPO updates. Its 128-seed evaluation finished:
  25 Act-1 victories (19.53%), 74 runs ending on floor 16. The evaluation
  of the *old warm-start baseline* then failed on Sly Snakebite after
  15 baseline seeds. No final stage report was committed to the results.
- The training win percentages are not a matched held-out comparison.
  The archive has one v8 stage-0020 model but no v7 stage-0020 model.

The emulator branch `agent/sly-random-target-v4` resolves Sly
explicit enemy-target cards by RNG-selecting a live enemy and changes
generic autoplay to randomize its target independently of the source
card. This AI branch pins that new emulator revision and retains the
v8 species-aware encoder, the v7 control option, and the certified Act-1
win condition.

## Compatibility

This **changes game dynamics**. The old optimizer checkpoints are intentionally
not resumable under the new emulator pin; do not override that check. Existing
v6/v8 exported weights may be used as *warm-start model parameters*, but a fair
v7-v8 architecture comparison should start both from the same original
round-160 v6 model and train afresh.

## Verification (Fish)

```fish
git fetch origin
git switch agent/sly-random-target-retest
git submodule update --init --recursive
dotnet test emulator/tests/Sts2Emulator.Core.Tests \
    --filter 'FullyQualifiedName~PrototypeSlyTests|FullyQualifiedName~PrototypeRandomTargetTests'
python -m pytest -q tests/test_enemy_instances_v8.py \
    tests/test_phase_split_ppo.py tests/test_act1_episode_goal.py
```

The paired training experiment (new output directories):

```fish
for encoding in public_resources enemy_instances
    python -u tools/train_longrun.py \
        --warm-start results/ppo-pr51-large/models/stage-0160.json \
        --tactical-state-encoding $encoding \
        --rounds 40 --stage-size 20 --episodes 32 --workers 15 \
        --eval-seeds 128 --win-anneal-threshold 128 \
        --train-seed-prefix ppo-v7-v8-slyfix-train \
        --eval-seed-prefix ppo-v7-v8-slyfix-eval \
        --output-dir results/ppo-v7-v8-slyfix-$encoding
end
```

Compare Act-1 clears on identical 128 held-out seeds and boss-HP
distributions rather than comparing unpaired training victory counts.
Still audit further unsupported mechanics before claiming end-to-end
fidelity to the retail game.
