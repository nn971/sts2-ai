# H4 negative-result record — 2026-10-25

Status: **research experiment concluded; preserve all artifacts**.
This is an engineering prioritization decision, not a proof that Nature or
Monte Carlo tree search cannot help in other settings.

## Pinned experiment

- Repo: nn971/sts2-ai, H4 PR #97; H4 schema
  \`sts2-ai-h4-educational-nature-ppo-v1\`.
- Emulator pinned at \`e56779ccd86725582af580925ca8242826ba2229\`,
  DLL SHA-256 \`945d1c595cc84cdacb2d8fd22ae6c398b6e72014c672a5926d940e93e2502ac9\`.
- Both start at v8 player SHA-256
  \`df0808503416fc85384e430a1cd609d5fb8cb5c9f55bfa6a9c2e8bfc568237d5\`.
- Each: 40 rounds × 256 combats = **10,240 combats**, 15 workers,
  Ascension 10, Ceremonial Beast/Vantom bosses, two pinned 20-card decks,
  reward \`boss_progress_v2\`.
- Educational: 50% reference trajectories, 50% learned-Nature trajectories
  with auxiliary player weight 0.25, Nature teacher alignment target.
- Control: 100% reference Chance Law PPO. Neither training mode depends on
  native RNG parity or inspector fidelity.

## Training and selection results (120 source worlds per checkpoint)

| Round | Reference-only reference-law score | Educational reference-law score |
|---:|---:|---:|
| 5 | 1.0722 | 1.0409 |
| 10 | 1.0745 | 1.0491 |
| 15 | 1.0787 | 1.0811 |
| 20 | 1.1180 | 1.0866 |
| 25 | 1.1224 | 1.0849 |
| 30 | 1.1329 | 1.1149 |
| 35 | 1.1357 | 1.0962 |
| 40 | **1.1325** | 1.1038 |

Round-40 reference-law greedy wins: 86/120 (71.7%) for the reference
control; 78/120 (65.0%) for educational Nature. Across the educational
experiment, mean teacher gradient alignment ≈ −0.0023 and positive-alignment
frequency ≈ 50.1%; no convincing teacher improvement was observed. Those
alignment statistics may also be affected by the chosen surrogate definition
and should not be interpreted as direct measured learning gains.

**Confound:** Replacing half the reference samples with educational
trajectories means this A/B test does not isolate the quality of the
Nature gradient objective from the effect of reducing reference exposure.
A 50/50 untrained-Nature control could isolate that effect, but is
deferred until there is reason to revisit Nature.

## Independent confirmation — 120 source worlds

Confirmation split from user-supplied H4 greedy-versus-UCT log,
\`source_digest=a38c33978d0f8ed475df3e84f0c525b42e377867aed354a8df673a75a33c05e4\`.

| Frozen player | Greedy reference-law score | Greedy reference-law wins | UCT-16 reference-law score | UCT-16 reference-law wins |
|---|---:|---:|---:|---:|
| v8 | 1.0146 | 61/120 (50.8%) | 0.9777 | 54/120 (45.0%) |
| H4 reference-only r40 | **1.1283** | **92/120 (76.7%)** | 1.0658 | 77/120 (64.2%) |
| H4 educational r40 | 1.0865 | 81/120 (67.5%) | 1.0354 | 68/120 (56.7%) |

These are *descriptive* values, without per-source paired confidence
intervals. The reference-only control wins 11 more out of 120 than
educational PPO on this confirmation suite. Do not assert statistical
significance from these rounded aggregates alone.

**UCT scope:** 16 rollouts per decision, 24 semantic-boundary rollout
horizon, 4000 simulated boundaries per combat, with greedy fallback after
exhaustion. Root-conditioned hidden state is not fair belief-state search.
The poor result applies to this costly bounded UCT baseline, not all MCTS.
Observed UCT evaluation arms require thousands of seconds versus hundreds
for greedy; do not make this search a training gate.

**Nature evaluation caveat:** Confirmation log
\`nature_checkpoint_sha256=null\`, so its \`learned\` arms used an initialized
PpoDrawNature, **not the trained H4 Nature checkpoint**. Those arms cannot
be cited as the behavior of a trained educational Nature. The shared log
ends after 11 of 12 arms; the educational/UCT/initialized-Nature arm was
incomplete at the recorded endpoint.

## Decision and artifact preservation

1. Mark H4 educational Nature and bounded UCT **negative for our current
   training plan**, not impossible in principle.
2. Train the player using **reference Chance Law** exclusively. Retain
   the semantic Chance Law, provider abstraction, and H1–H4 evidence.
3. Keep H4 checkpoints in their current outputs:
   \`results/nature/h4-educational-a10/\` and
   \`results/nature/h4-reference-only-a10/\`. Do **not** overwrite them.
4. Preserve PR #97 and the H4 source/log record; do not make fidelity
   inspector or exact native RNG parity a prerequisite.
5. Start H5 in a separate branch, with independent confirmation and
   cross-combination holdouts; extend supported combat configuration
   coverage before proposing further planning algorithms.

No claim of native encounter/deck probabilities or complete StS2 coverage.
