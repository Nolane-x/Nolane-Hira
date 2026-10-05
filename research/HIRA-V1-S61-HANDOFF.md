# HIRA V1 S61 → S62 handoff

Parent verdict: **S61 Case B**

## What S61 established

1. S59: pairwise-only decisions carry strong stability signal but destroy correctness.
2. S60: a global bounded residual safely protects correctness but transfers almost none of the stability signal.
3. S61: a 5-parameter per-query confidence gate learns nonconstant alpha and keeps correctness safe, but gold CE alone does not teach the gate when pairwise evidence improves cross-view reliability.

Fresh S61:
- canonical: **+0.26 pp**
- paraphrase: **+2.60 pp**
- paired both-correct: **-0.52 pp**
- question-swap: **+1.56 pp**
- agreement: **0.00 pp**
- JS: **+0.003272 worse**
- alpha range: **0.0938–0.1137**.

## Required S62 question

> Can the same bounded residual family become useful when the gate is supervised on TRAIN-only pairwise-intervention reliability rather than only end-task gold CE?

## Handoff constraints

Retain:
- S51 native authority;
- immutable cache / one encoder state-once;
- S59 pairwise head;
- bounded residual with alpha <= 0.35;
- no pairwise-only final path;
- no teacher / DEV-derived target / pseudo-target;
- no gradient path into native/correction from gate objective;
- fresh S62 TRAIN/DEV;
- one DEV only.

S62 must preregister an explicit TRAIN-only reliability label/score before A0:
- based only on gold labels and paired canonical/paraphrase TRAIN views;
- measures whether a bounded pairwise intervention helps both correctness and cross-view stability;
- cannot use S61 DEV outcomes to label individual examples;
- must include an anti-collapse path where harmful pairwise interventions train alpha downward.

Do not reopen S61.
