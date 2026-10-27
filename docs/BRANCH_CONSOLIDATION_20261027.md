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


## Validation and compatibility audit

- Integration branch created by a three-parent merge (H6-F, H7, main).
  Direct ancestry comparisons show the integrated branch is ahead of each
  parent without missing commits.
- H6-F V20 card operations, H6-D/E action semantics, macro attention, and
  H7 public Torch Transformer forward/backward checks have passed focused
  CI on the integration branch. H7 reference-law real P0 PPO smoke also
  passed on integration
  [run 40452658661](https://github.com/nn971/sts2-ai/actions/runs/40452658661).
- Repository-wide Ruff initially found 18 historical style errors and
  now passes; an H6-C escaped HTML action tooltip test exposed an actual
  replay regression and was repaired (the public payload is escaped
  before being included in the tooltip).
- `parent-ci` remains red due to strict typing debt: initially **139 mypy
  errors across 42 files**, with missing optional Torch in the parent CI
  environment and numerous legacy typing issues. Track this in
  [issue #113](https://github.com/nn971/sts2-ai/issues/113). This must
  stay visible; do not claim full green CI or quietly disable mypy.
- The historical H2 emulator pin supports only four conditioned deck
  presets; the later H5 boss presets had accidentally widened
  `DECK_PRESETS` used by its old tests. H2 tests/CI now use the pinned
  four-deck curriculum explicitly, preserving its original 8/12-cell
  comparisons rather than pretending the old emulator supports H5 decks.
- The legacy native-run `enemy_instances` PPO smoke exposed missing
  public `draw_pile`/`relic_counters` arrays. The global encoder now
  uses V6 public features and explicit missingness markers for absent
  arrays, retaining the previous V8 calculation when both lists exist.
  Explicitly malformed present arrays remain errors. Re-run the real
  native batched PPO CI before relying on this path.

Other historical research CI may require additional compatibility
repairs. The original-game fidelity and native RNG inspector are not
gates. Do not combine these integration smoke outcomes into a claim that
H7 is a strong combat agent.
