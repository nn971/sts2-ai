# H6-C v18: Public macro choice identity

## Diagnosis

In H6-C v17, the shop-buy and event-choice action encoder falls back to action kind. There are no shop/event object pointers. Two different shop card purchases or event choices therefore have identical action inputs, identical target context, and necessarily identical logits, regardless of training. This is a representation defect rather than a visualization problem.

The paired replay reports now record input-layer feature-alias groups separately from tied logits.

## Compatibility

- v17 macro attention format and previous H6-C-80/H6-C-160 exported checkpoints are unchanged.
- New v18 format: sts2-neural-policy-value-v18-macro-public-choice-identity; opt in with --strategy-encoding macro_attention_v2.
- Shop offer IDs join to public item IDs, prices and other visible properties. Event choice IDs join to public choice identities and known costs/consequences. Offer IDs are NOT hashed as identities.
- The batched PPO path, single-step differentiable forward and portable inference all use the same v18 representation.
- Warm starting v18 from v17 retains global/object attention but resets action_weight, whose feature hash basis has changed. The combat weights remain frozen.
- Start a NEW v18 optimizer checkpoint; do not resume an existing v17 .pt checkpoint as v18.

## Replays, maps, events, and missing skip relic

The pinned emulator serializes room types numerically. The viewer decodes 0=Combat, 1=Elite, 2=Event, 3=Shop, 4=Rest, 5=Boss, 6=Unknown, 7=Treasure. The map draws every public node/edge and shows both routes. Boss identity stays in the sticky header. Visited event IDs and encounter enemy IDs are shown in floor histories and hovered map nodes once revealed. Future event/encounter identities are not guessed.

The replay also expands shop item names, visible event-choice values and full legal menus. If skip_reward_relic is unavailable despite a relic offer, the viewer explicitly notes that the emulator did not offer it. The pinned emulator omits skip_reward_relic in ordered reward mode; see https://github.com/nn971/sts2-emulator/issues/192 for a separate mechanics fix. Emulator fidelity verification remains independent of agent training.

## Follow-up training

After verification, initialize a fresh v18 training from the preserved H6-C-80 model using --strategy-encoding macro_attention_v2 with new checkpoint/output/report paths and fresh evaluation seeds. This is a representation correction requiring training, not an improvement already proven by the old checkpoint.