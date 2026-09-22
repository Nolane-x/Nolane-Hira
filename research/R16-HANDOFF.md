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


## Empirical result — PRIMARY GATE FAIL

Workflow run `35690994565` completed successfully as software and preserved the exact tournament artifact.

Frozen R15 baseline:
- matched accuracy: **0.5613**
- strict structural non-entailment validation accuracy: **0.5000**

The strict structural miner discovered:
- structural train: **606** balanced examples (303 neutral + 303 contradiction)
- validation_mismatched satisfying strict predicates:
  - neutral available: **1**
  - contradiction available: **16**
- balanced structural validation: **2 examples total**

This is an important protocol failure: a 2-example validation set is not a credible model-selection authority. The observed structural 0.50 -> 1.00 jump is therefore **not accepted as meaningful evidence**.

Candidate selected matched accuracies:
- ratio 1:1, LR 1e-4: 0.4787
- ratio 1:1, LR 2e-4: **0.5073**
- ratio 2:1, LR 1e-4: 0.4613
- ratio 2:1, LR 2e-4: 0.4773

All candidates failed the matched floor 0.5606666613 and all were ineligible.

Failure-analysis head SHA-256:
`dbe0ddd8bf3811c98f5062bbf482f5d5b4f1f5991fa6f734f7ca88c24e88cc75`

### Conclusion

R16 fails for two reasons:
1. the strict structural slice is too rare in MultiNLI validation_mismatched to support selection;
2. over-concentrating on only 606 strict non-entailment examples causes severe retention loss despite the R15 teacher anchor.

No HANS/Breaking diagnostics are authorized. Their frozen adapted gates remain unconsumed for R16.

## Next research direction

Do not merely relax the R16 thresholds.

A successor should:
- create a **near-structural** MultiNLI slice using a frozen continuous structural score rather than exact subset predicates;
- require a minimum validation support before any model-selection claim;
- avoid global fine-tuning toward the structural specialist;
- test weight interpolation or conditional micro-expert routing so the R15 base competence is preserved by construction.
