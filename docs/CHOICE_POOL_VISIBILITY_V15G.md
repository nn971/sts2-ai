# v15g ChoicePool compatibility

## Reproduced native failure

The sts2-emulator revision pinned by sts2-ai (9117b4af09f0164a19bb1c41b70f688c8948e0e3) contains CombatState.ChoicePool. Generated-card selection actions reference these card instances, but PrototypeAiEnvironment.CreateCombatObservation exposes hand/draw/discard/exhaust/play piles **without exposing ChoicePool**. A valid select_cards action may therefore reference an absent observed card, e.g. ObjectRef(kind='card', key=19).

This is an emulator **public-observation completeness gap**, not an optimizer failure. The permanent public schema work is in [sts2-emulator#99](https://github.com/nn971/sts2-emulator/issues/99). Players see the generated card options, but the old AI observation currently cannot identify them.

## Explicit, non-fabricating fallback

The new helper extract_actionable_combat_objects(state, legal_actions) handles **only** legal select_cards menus where combat.pending_choice_id is select_cards. Missing card pointers become explicit unobserved-choice tokens with identity <unobserved-choice> and marker attributes_unavailable.

The missing card instance integer is **only an action-routing key** and never a learned scalar feature. Its identity, cost and effects are not invented. Other missing card/enemy pointers still raise errors, as do invalid/duplicate select_cards pointers. This restores PPO preparation and portable/torch inference but **does not let the model compare unknown generated card identities**.

The compatibility extractor is wired into v15c/v15d Torch and portable inference, PPO sample/vocabulary construction, v15e audit, and v15g native corpus capture and replay.

When a future emulator explicitly provides a public combat.choice_pool array, the original extractor already handles it and uses real instance card metadata. No fallback is needed for those cards. The corpus and profile report count unobserved_choice_tokens to expose incomplete data rather than hiding it.

## Regression and rerun (Fish)

The focused CI suite includes tests/test_choice_pool_visibility_v15.py, covering old and future observations, action strictness, PPO resolved combats, and Torch/portable inference parity.

~~~
git fetch origin
git switch agent/v15g-choice-pool-visibility
git submodule update --init --recursive

python -m pytest -q tests/test_choice_pool_visibility_v15.py tests/test_real_minibatch_profile_v15.py

python -u tools/profile_v15_real_minibatches.py \
    --base-model results/ppo-v8-batched-40/models/stage-0120.json \
    --corpus results/v15g-public-combat-ppo-corpus.json \
    --report results/v15g-real-minibatch-profile.json \
    --runs 4 --max-decisions 512 \
    --devices cpu cuda --batch-sizes 32 64 128 \
    --threads 1 --width 64 --warmups 2 --trials 5 --build
~~~

Do not update the pinned emulator just to run this fallback; the underlying public schema fix remains a separate explicit change. This does not modify any model weights or legacy v14 fidelity data.
