# Purpose

## Research objective

The long-term goal of `sts2-ai` is to develop an AI that is strong at **whole-run strategy**, rather than merely controlling a live game or choosing locally plausible actions.

A successful agent should reason jointly about:

- immediate combat survival and resource expenditure;
- expected value of path alternatives;
- deck construction and card skipping;
- current and future potion value;
- gold reservation and shop timing;
- relic synergies and boss relic tradeoffs;
- event risk/reward;
- rest/upgrade/other rest-site choices;
- act-level and run-level win probability.

The central quantity is conceptually a long-horizon value such as

```text
V(observation) = expected probability / utility of eventual run success
```

under an explicit information policy.

## Foundation

Strong strategic learning requires a reliable environment. This repository therefore depends on `sts2-emulator`, whose first milestone is a behaviorally trustworthy and practically fast emulator of complete runs.

The simulator is not merely a convenience for training. It defines the transition semantics against which strategic claims are made.

## Philosophy

The project favors a combination of:

1. exact or highly faithful simulation;
2. targeted search at strategically important states;
3. persistent reuse of expensive strategic evidence;
4. learned policy/value models that generalize search results;
5. scenario archives that concentrate computation on hard states;
6. rigorous separation between hidden engine truth and player-legitimate information.

This is closer to an expert-iteration research program than to a generic LLM controller or pure model-free RL baseline.

## Non-goals

This repository does not aim to:

- automate mouse/keyboard input as the primary research contribution;
- ask a general-purpose LLM for each move;
- hide emulator inaccuracies behind a learned world model;
- optimize only combat while treating run strategy as hand-authored heuristics;
- use hidden RNG in experiments labeled as fair play.

Those approaches may be useful diagnostics or baselines, but they are not the target system.
