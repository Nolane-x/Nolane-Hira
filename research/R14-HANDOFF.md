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


## Empirical result — PRIMARY GATE FAIL

Workflow run `35667400836` completed successfully as software and produced an immutable repair artifact.

Frozen R12 baseline on the R14 evaluations:
- matched MultiNLI accuracy: **0.5807**
- hard high-overlap non-entailment accuracy: **0.3390**

Failure-analysis checkpoint after the hard-negative curriculum:
- matched MultiNLI accuracy: **0.5067**
- hard high-overlap non-entailment accuracy: **0.6810**
- hard non-entailment gain: **+0.3420**

Primary gates:
- hard non-entailment gain >= +0.10: **PASS**
- HIRA params remain 422,159: **PASS**
- matched accuracy >= 0.5606666613: **FAIL**
- selected checkpoint eligible: **FAIL**
- overall R14 primary gate: **FAIL**

Failure-analysis head SHA-256:
`eb25737d913776a1c2c9cccc8905c90a614614455466429721e9377144852f56`

### Diagnosis

The experiment demonstrates that the R13 anti-entailment failure can be repaired with the existing HIRA capacity, but the naïve hard-negative curriculum causes a large competence trade-off. The problem is therefore not simply "too few parameters"; it is an optimization/retention problem.

Because the primary gate failed, the post-selection HANS/Breaking diagnostic script correctly refuses to run. R13 untouched failures remain the authoritative held-out evidence.

## Next state

R15 should test retention-preserving repair:
- replay ordinary MultiNLI alongside hard negatives;
- anchor the repaired head to the frozen R12 teacher distribution on ordinary examples;
- preserve matched accuracy while improving the hard subset;
- keep A13 frozen and HIRA parameter count unchanged.

Do not use HANS, Breaking NLI, XNLI, MASSIVE or Banking77 for R15 model selection.
