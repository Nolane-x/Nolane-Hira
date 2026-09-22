# R16 handoff — structural counterexample repair

Status: **protocol frozen; empirical run pending**.

## Starting point

R15 is the accepted starting checkpoint:
- HIRA head SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`
- matched MultiNLI: 0.5613
- hard high-overlap non-entailment: 0.6380
- adapted HANS non-entailment: 0.2263
- adapted Breaking contradiction recall: 0.2915

A13 remains frozen:
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- safetensors SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`

HIRA remains exactly 422,159 parameters.

## Why R16

R15 solved the catastrophic-retention problem but still showed a residual structural anti-entailment weakness. R16 targets that weakness without using HANS for model selection.

## Structural predicates

Lowercase alphanumeric tokenization only.

For premise token sequence P and hypothesis H:

1. **contiguous** — H appears as one exact contiguous token span in P;
2. **ordered subsequence** — every H token appears in P in order, gaps allowed;
3. **multiset subset** — P contains every H token with at least the same multiplicity.

A neutral/contradiction example is a structural hard negative if any predicate is true.

Ranking for deterministic balanced selection:
1. contiguous;
2. ordered subsequence;
3. multiset subset;
4. lexical recall;
5. source index.

## Frozen data roles

Only:
`nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221`

Structural train:
- train split;
- neutral + contradiction only;
- max 6,000 per label;
- use the same balanced count available for both labels.

Structural validation:
- validation_mismatched;
- neutral + contradiction only;
- max 500 per label.

Replay:
- train.shuffle(seed=13), first 15,000.

Matched validation:
- validation_matched.shuffle(seed=14), first 1,500.

Forbidden for selection:
- HANS
- Breaking NLI
- XNLI
- MASSIVE
- Banking77

## Retention authority

The exact frozen R15 selected head is both:
- the initialization;
- the replay teacher.

Teacher KL weight is fixed at **0.5**, the R15 winner.

## Tournament

Four independent candidates, all starting from exact R15:

- hard/replay ratio 1:1, LR 1e-4
- hard/replay ratio 1:1, LR 2e-4
- hard/replay ratio 2:1, LR 1e-4
- hard/replay ratio 2:1, LR 2e-4

Each:
- 4 epochs
- batch size 96
- A13 frozen
- HIRA unchanged at 422,159 params
- CE + 0.1 Brier + 0.5 anti-entailment margin on structural hard batches
- replay CE + Brier
- KL(R15 teacher || student) weight 0.5

An epoch is eligible only when matched accuracy >= 0.5606666613.

Across eligible candidates:
1. maximize structural validation non-entailment accuracy;
2. tie-break by matched accuracy.

## Preregistered primary gates

All must pass:

- matched accuracy >= 0.5606666613
- structural validation non-entailment accuracy >= 0.60
- structural improvement >= +0.10 absolute over frozen R15 baseline on the same structural slice
- HIRA parameter count exactly 422,159
- selected checkpoint eligible

No threshold changes after the empirical run.

## Post-selection diagnostics

Only if primary R16 selection passes may HANS/Breaking be re-evaluated.

Because their previous failures are already known, those remain **adapted diagnostics**, never restored to held-out status.

XNLI remains unopened as a model-selection source.
