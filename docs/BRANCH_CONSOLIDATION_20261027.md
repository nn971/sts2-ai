# Branch consolidation — 2026-10-27

## Canonical integration state

This integration unites **H6-F V20 card-selection representation** and the
**H7 opt-in, scratch-trained shared combat Transformer**, preserving both
research lineages and original mainline development/evidence policies.
The merge commit includes the H6-F, H7 and main parents. No emulator branch
or H4/H5/H6 checkpoints were changed. The AI submodule pin is retained from
H6-F. This consolidation is **contract/unit and integration smoke evidence**,
not proof of improved player strength.

| Workstream | Treatment |
| --- | --- |
| H3/H4/H5 experiments and reference law | Retained in H6-F ancestry with their versioned formats |
| H6-A through H6-E | Retained in H6-F ancestry; keep V17/V18/V19 compatible |
| H6-F / PR #111 | Canonical V20 macro operation-aware deck card pointers (Remove/Upgrade/Transform/Enchant) |
| H7 / PR #110 | Independent opt-in 14.7M shared game-object Transformer, public combat actor, reference-law PPO |
| main unique policy | Preserve AGENTS.md, experiment evidence standard, README reference |
| Older divergent macro prototypes | Retain as reproducible history, never blindly overwrite successors |
| Emulator | Unmodified; native fidelity remains independent of the training gates |

A historical branch being open does not mean its implementation should be
layered on top of a newer successor: older shop/event alias fixes and
alternative macro versions are superseded by the H6-D/E/F series. Do not
delete historical experiments/checkpoints or rewrite their format IDs.

## Supported entrypoints and owners

- H6-F macro: `tools/train_phase_split.py --strategy-encoding macro_card_ops`
- Readable macro run decisions: `tools/replay_h6_macro.py`
- H7 scratch combat: `tools/train_h7_reference_combat.py`
- H7 held-out evaluation: `tools/evaluate_h7_reference_combat.py`
- H7 numerical probe: `tools/probe_h7_transformer.py`

The normal H4/H5/older v15 tools remain available. H7 requires an
H5-compatible semantic P0 emulator DLL, and the current training cells are
**16 configurations**, so scaling the model alone is not a broad experiment.
The H7 macro head is a reserved interface: it does **not yet** encode all
H6-F pending card operation/choice semantic pointers. Do not enable it
as a replacement for the V20 macro actor.

## Gates before a mainline merge / next experiment

1. Validate **H6-F V20**, H6-E/D/C action semantics, and all relevant
   portable/batched PPO invariants; preserve frozen-combat bytes.
2. Validate H7 Torch tests and real C# reference-law PPO smoke. Record
   both as integration checks; a short run is not performance evidence.
3. Preserve the mainline AGENTS/evidence policy. Keep native fidelity
   independent and do not require native RNG/inspector parity.
4. Benchmark H7 actual actor inference and 15-worker throughput before
   committing GPU resources to a 14.7M model training campaign.
5. Expand coherent Silent A10 deck, relic, potion, HP, and encounter
   configurations (emulator issue #185) before meaningful scaling studies.
6. Use separate source-run-disjoint holdouts and multiple independent
   training seeds for any claimed player-quality improvement.

The canonical branch will be proposed to `main` for review, retaining
the original research branches and unmerged draft PRs as traceable history.
