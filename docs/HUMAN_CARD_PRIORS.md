# Optional human-statistics priors for card rewards

**Status:** implemented as an opt-in, standalone policy-prior component.
**Integration:** `search/card_priors.py`; no live data ingestion, automated scraping,
or production rankings included. The stochastic PUCT kernel is implemented
separately in `search/puct.py`, but a real emulator fair-history adapter is
still required.

## Why this is a useful complement to teacher-free self-play

Early PUCT has few successful outcomes from which to learn which card rewards
improve eventual survival. Aggregated human **card choices conditional on
being offered** can provide a cheap exploration prior. This is a starting
bias only: independent on-policy stochastic outcomes determine value.

Potential sources worth evaluating include sts2log.com (Skada Analytics),
Untapped.gg, and STS Tracker. A source's availability, export permissions,
methodology and game version must be verified before ingesting any statistics.
This repository does not claim that any named source exports offer-level
counts or that it grants bulk automated reuse. Use authorized snapshots.
No source-specific scraping or APIs are implemented.

A choice when a card is offered is conceptually different from the probability
law for **which card is offered**. The latter comes from game rules and stays
inside the emulator chance model.

## Snapshot contract (version sts2-offer-pick-prior-v1)

Provide a locally obtained **aggregate** JSON snapshot with genuine stable
in-game card identifiers and an auditable collection method:

```json
{
  "format": "sts2-offer-pick-prior-v1",
  "provenance": {
    "source": "YOUR_AUTHORIZED_SOURCE",
    "source_url": "https://example.org/documentation",
    "collected_at": "YYYY-MM-DD",
    "game_build": "PINNED_GAME_BUILD",
    "character": "Silent",
    "act": 1,
    "ascension_min": 0,
    "ascension_max": 0,
    "methodology": "Explain how offered and picked were counted, the sample population and exclusions"
  },
  "cards": [
    {"card_id": "EXACT_CARD_ID_A", "offered": 120, "picked": 50},
    {"card_id": "EXACT_CARD_ID_B", "offered": 160, "picked": 30}
  ]
}
```

**The numbers above are illustrative, not collected statistics.** The importer
refuses malformed counts, duplicate IDs, inverted ascension ranges, and missing
provenance. It hashes the canonical snapshot to make experimental attribution
possible. Never reinterpret wins among decks containing a card as
`picked/offered`; that is a different statistic.

A snapshot activates only on an exact character/game-build/Act/ascension match,
when the public observation agrees with Act and the legal actions are exclusively
card-reward picks and skips. Missing cards and Skip receive neutral relative
scores. Other actions use exactly the unmodified model prior.

## Mathematics

Let `offered_c` and `picked_c` be counts from the same specified
population and event semantics. Set

```text
p_base = sum_c picked_c / sum_c offered_c
p_c = (picked_c + alpha * p_base) / (offered_c + alpha)
r_c = logit(p_c) - logit(p_base)
```

The default `alpha=20` is a tunable shrinkage scale (not empirical truth).
Given the currently legal card/Skip actions, take a softmax of the scaled
`r_c`; unmatched cards and Skip use score zero. Mix with the masked learned
prior:

\[
P_k(a\mid h) = (1-\lambda_k)P_{\mathrm{learned}}(a\mid h)
             + \lambda_k P_{\mathrm{human}}(a\mid h).
\]

Only the learned prior gets updated by gameplay. Start with fixed
`lambda` under an experimental flag, then ablate scheduled decay to zero.
Conditioning on deck, map, current HP, offered alternatives, ascension and
sample selection can matter much more than aggregate pick rates; a scalar
card preference is intentionally only a weak initializer.

## Integration example

```python
from pathlib import Path
from sts2_ai.search.card_priors import (
    CardPriorContext, CardPriorDataset, blended_card_reward_prior
)

dataset = CardPriorDataset.load(Path("local-authorized-aggregate.json"))
context = CardPriorContext(game_build="PINNED_GAME_BUILD", character="Silent",
                           act=1, ascension=0)

def prior_provider(observation, legal_actions):
    learned_logits = model.evaluate(observation, legal_actions).action_logits
    return blended_card_reward_prior(
        observation, legal_actions, learned_logits,
        context=context, dataset=dataset, human_weight=0.25,
    ).probabilities

# After integrating a genuinely validated fair-history chance adapter:
# StochasticPuct(fair_model, prior_provider=prior_provider, seed=...)
```

The model used above is any existing observation-only policy/value model.
The fair stochastic model remains a separate, currently blocked integration.

## Evaluation gates

Use exactly the same training, validation and held-out **run seed groups**,
game revision, compute budget and legal action filters for:

1. Uniform initial priors (no external statistics).
2. Learned-only priors.
3. Human/learned mixture with fixed or annealed weight.

Report Act-1 clear rate, censored episode count, mean progress, time, simulator
transitions, card-prior activation count, matched-card coverage, source ID,
source patch/version and data sample sizes. Paired uncertainty intervals matter
more than small raw score differences. Run a mismatched-version negative
control; verify that changing human data has no effect on combat, map, RNG,
other character, and incomplete context. **Do not tune on held-out seeds.**

## Limits of the current implementation

- Does not download, scrape, ingest private runs, or bundle card rankings.
- Does not estimate causal card value from observational pick frequencies.
- Does not operate without matching source provenance and visible card IDs.
- Does not override game randomness or provide fair hidden-state continuations.
- Does not establish superior AI strength; this requires controlled experiments.
