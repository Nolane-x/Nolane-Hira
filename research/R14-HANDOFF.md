# R14 handoff — anti-entailment hard-negative repair

Status: **protocol frozen; empirical run pending**.

## Starting point

R13 showed two independent untouched failures of the exact R12-selected head:

- HANS: 51.09% overall, **8.37% non-entailment accuracy**.
- Breaking NLI: 22.34% overall, **12.62% contradiction recall**.

The R12 head itself is:
`0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86`.

A13 remains fully frozen:
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- safetensors SHA-256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`.

## R14 hypothesis

The first repair attempt tests whether the main R13 failure is **training geometry / entailment bias**, not insufficient parameter capacity.

No HIRA parameters are added. A13 is not unfrozen.

## Allowed training/model-selection data

Only pinned MultiNLI:
`nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221`.

Forbidden for R14 model selection:
- HANS;
- Breaking NLI;
- XNLI;
- MASSIVE;
- Banking77.

HANS/Breaking may be evaluated only after the repaired checkpoint has already been selected using MultiNLI.

## Mining rule

Lexical-overlap score is frozen as the multiset recall of hypothesis alphanumeric tokens found in the premise.

Training curriculum:
- top high-overlap entailment: 6,000
- top high-overlap neutral: 6,000
- top high-overlap contradiction: 6,000
- total: 18,000

Hard validation:
- `validation_mismatched`
- top 500 high-overlap neutral
- top 500 high-overlap contradiction
- total: 1,000

Matched preservation validation reproduces R12:
- `validation_matched.shuffle(seed=14).select(first 1500)`.

Selected-index SHA-256 values are written into the empirical receipt.

## Objective

Start from the exact R12 head, then train HIRA only:

```
loss =
  cross_entropy
  + 0.1 * multiclass_brier
  + 0.5 * anti_entailment_margin
```

For neutral/contradiction golds:

```
margin_loss = relu(0.5 - gold_logit + entailment_logit)
```

Default:
- 6 epochs
- batch 96
- LR 5e-4
- HIRA params: 422,159
- A13: frozen

## Selection rule

An epoch is eligible only when matched validation accuracy is at least:

`0.5806666613 - 0.02 = 0.5606666613`.

Among eligible epochs, select the highest:
1. hard-validation non-entailment accuracy;
2. hard-validation accuracy;
3. matched validation accuracy.

If no epoch is eligible, retain a fallback checkpoint only for failure analysis and mark `selected_eligible=false`.

## Primary preregistered gates

- matched validation >= 0.5606666613
- hard non-entailment improvement >= +0.10 absolute vs frozen R12 baseline on the same hard MultiNLI subset
- HIRA parameter count remains exactly 422,159
- selected checkpoint is eligible

All must pass.

## Adapted diagnostics after selection

Only after the checkpoint is selected:

- HANS non-entailment must improve >= +15 absolute points from R13's 8.37%.
- Breaking contradiction recall must improve >= +15 absolute points from R13's 12.62%.

These are **adapted diagnostics**, not new held-out claims, because R14 was designed after seeing R13 failures.

## Interpretation

If MultiNLI hard negatives + HANS improve but Breaking does not, the remaining deficit is evidence for missing lexical/world knowledge. The next experiment should externalize knowledge through a sidecar/retrieval/prototype mechanism rather than increase HIRA blindly.

Untouched R13 failures remain permanent evidence.
