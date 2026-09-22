# R15 handoff — retention-preserving anti-entailment repair

Status: **protocol frozen; empirical run pending**.

## Why R15

R14 proved that the existing 422,159-parameter HIRA head can learn the hard anti-entailment regime, but naive hard-negative fine-tuning catastrophically trades away ordinary MultiNLI competence.

R14 failure:
- matched: 0.5807 -> 0.5067
- hard non-entailment: 0.3390 -> 0.6810
- primary gate: FAIL

R15 asks whether replay + teacher anchoring can keep both capabilities in the same unchanged HIRA head.

## Frozen start

- R12 HIRA head SHA-256:
  `0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86`
- A13 revision:
  `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- A13 safetensors SHA-256:
  `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`
- HIRA params: 422,159
- A13 frozen.

## Allowed source

Only:
`nyu-mll/multi_nli@da70db2af9d09693783c3320c4249840212ee221`.

Forbidden for model selection:
- HANS
- Breaking NLI
- XNLI
- MASSIVE
- Banking77

## Data roles

Hard train:
- same frozen lexical-overlap miner as R14;
- top 6,000 examples per NLI label;
- 18,000 total.

Replay:
- exact R12-style deterministic `train.shuffle(seed=13)` prefix;
- 15,000 examples;
- frozen R12 head supplies teacher probabilities.

Validation:
- same R12 matched validation 1,500;
- same R14 hard validation: top 500 neutral + 500 contradiction high-overlap examples from validation_mismatched.

## Objective

Each update receives one hard batch and one replay batch.

Hard loss:
```
CE + 0.1 * Brier + 0.5 * anti_entailment_margin
```

Replay loss:
```
CE + 0.1 * Brier
```

Total:
```
0.5 * hard_loss
+ 0.5 * replay_loss
+ lambda_KL * KL(R12_teacher || student)
```

This makes retention an explicit optimization term instead of hoping replay alone prevents forgetting.

## Preregistered tournament

Independent candidates start from the exact R12 head:

- lambda_KL = 0.25
- lambda_KL = 0.50
- lambda_KL = 1.00

Each:
- 5 epochs
- LR 2e-4
- batch 96
- hard/replay ratio 1:1
- A13 frozen
- HIRA parameter count unchanged.

An epoch is eligible only if matched accuracy >= 0.5606666613.

Across candidates:
1. reject all ineligible checkpoints;
2. maximize hard non-entailment accuracy;
3. tie-break by matched accuracy.

## Primary gates

All must pass:

- matched accuracy >= 0.5606666613
- hard non-entailment accuracy >= 0.55
- hard gain >= +0.20 absolute over R12 hard baseline 0.339
- HIRA params exactly 422,159
- selected checkpoint eligible.

No threshold may be changed after the run.

## Diagnostics authority

Only if the primary gate passes may adapted HANS/Breaking diagnostics be run.

Those diagnostics are no longer held-out claims because R13 exposed their failure modes. The untouched R13 receipts remain the permanent generalization evidence.

## Interpretation

If R15 passes, the single small HIRA core has enough capacity for both ordinary MultiNLI and the anti-entailment repair; the R14 problem was retention geometry.

If R15 still cannot satisfy both gates, the next architecture test should be conditional routing / micro-experts or a residual repair adapter, not blind global fine-tuning and not immediate parameter inflation.


## Empirical result — PRIMARY GATE PASS

The R15 tournament completed at workflow run `35688830869`.

Selected candidate:
- teacher KL weight: **0.50**
- matched MultiNLI accuracy: **0.5613**
- hard non-entailment accuracy: **0.6380**
- hard gain vs R12 baseline: **+0.2990**
- matched Brier: **0.5369**
- matched ECE: **0.0235**
- selected head SHA-256: `007e24fff0e0e48a096de59276f7ab8d0e25bdb826a3f81a7838c5bd6151723f`

All R15 primary gates passed. This demonstrates that the R14 trade-off was not a hard capacity limit: the same 422,159-parameter HIRA core can retain ordinary MultiNLI competence while learning much stronger anti-entailment behavior when replay and teacher anchoring are used.

## Adapted diagnostics — combined gate FAIL

After primary selection was frozen, workflow run `35689894137` evaluated the selected R15 head on the already-known R13 failure suites without updating weights or thresholds.

HANS:
- overall accuracy: **0.5291**
- entailment accuracy: **0.8319**
- non-entailment accuracy: **0.2263**
- non-entailment gain vs untouched R13 baseline: **+0.1425**
- required adapted gain: +0.15
- HANS adapted repair gate: **FAIL**, missing by about **0.00747 absolute**

Breaking NLI:
- accuracy: **0.3045**
- macro recall: **0.4514**
- contradiction recall: **0.2915**
- contradiction gain vs untouched R13 baseline: **+0.1653**
- Breaking adapted repair gate: **PASS**

Combined diagnostic gate: **FAIL** because the HANS gain narrowly missed its frozen threshold.

### Interpretation

R15 is a successful retention repair and a strong partial generalization repair:
- R14 catastrophic trade-off is solved;
- Breaking contradiction deficit improves enough to pass its adapted gate;
- HANS non-entailment rises from 8.37% to 22.63%, but structural/lexical-overlap robustness remains insufficient.

R16 should target structural anti-entailment using only non-HANS training/model-selection data. Do not reinterpret R13/R15 HANS as held-out after designing R16 from these results.
