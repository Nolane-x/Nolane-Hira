# HIRA V1 S48 A0 receipt — Query-Quotient Option Evidence

Status: **QUALIFIED**

Run: `37171579714`  
Artifact: `11291393129`  
Artifact digest: `sha256:de5ae3d7c936539c19d4b1a9a695fbd27a44fbeac7d1f5d46f26d5ab959090e7`  
Authorization head: `eeb9e328d28125723c3e92941ef48eb1d532ea12`

Outcome:
`HIRA_V1_S48_A0_QUERY_QUOTIENT_OPTION_EVIDENCE_READY`

## Frozen surface

- native trainable: **49,152**
- reference correction: **114,688**
- treatment correction: **114,688**
- total treatment: **163,840**
- quotient trainable parameters: **0**
- second encoder pass: **false**
- raw-query bypass: **false**

## Quotient mechanics

- K=3: PASS
- K=7: PASS
- K=255: PASS
- option-permutation quotient error: **0**
- option-permutation corrected-logit error: **0**
- orthogonal nuisance quotient error: **5.96e-8**
- orthogonal nuisance corrected-logit error: **7.45e-9**
- relation-relevant quotient norm: **0.99999994**
- distinct relation-direction absolute cosine: **0.496139**
- zero-subspace quotient max abs: **0**
- zero-subspace corrected-logit error: **0**
- probability-mass max error: **1.192e-7**

## Runtime / ownership

- one encoder batch: **true**
- state-view encodes: **32**
- matched native gradient max abs: **0**
- matched native one-step parameter max abs: **0**
- matched native one-step output max abs: **0**
- correction -> native runtime gradient: **0**
- native objective -> correction gradient: **0**
- JS-only native gradient: **0**
- JS-only A/B/W gradients: **0.000198148 / 0.004662225 / 0.388773382**
- W -> B -> A warm-start remains live

## A0-only diagnostic geometry

- raw-query cross-view cosine: **0.978216**
- quotient cross-view cosine: **0.444679**
- raw-vs-quotient canonical cosine: **0.114795**
- raw-vs-quotient paraphrase cosine: **0.091185**
- quotient zero fraction: **0**
- quotient norm mean: **1.0**

These values are diagnostic only and were not used for model selection.

The low A0 quotient cross-view cosine is a warning that the state-conditioned option-signature span may itself be view-sensitive. The preregistered fresh S48 court remains necessary; this diagnostic does not authorize redesign or tuning.

## Interpretation

S48 mechanically qualifies.

The query-quotient path:
- removes synthetic query nuisance orthogonal to the current option-difference span;
- has no raw-query bypass;
- adds zero parameters;
- preserves exact native/correction ownership;
- preserves the existing 163,840 total treatment surface and one-pass runtime.

Fresh S48 TRAIN/DEV remains separately gated.
