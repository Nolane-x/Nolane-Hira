# HIRA V1 S17 contract — Norm-Balanced Shared Gradient Optimization

Status: **OPEN / PREREGISTERED BEFORE S17-A0 EXPOSURE**

Issue: #214

Base main:
`82a2702b069f05b571a0cd7383ea7686d0ecdf42`

S16 is frozen as `HIRA_V1_S16_SHARED_GRADIENT_SURGERY_DEV_FAIL`.

## 1. Motivation

S16 found frequent shared-surface conflict (mean TRAIN conflict rate **0.3767361111**) and partially recovered both fused and relation quality over S15.

But late TRAIN gradient magnitude was severely imbalanced:
- epoch-24 mean primary norm: **4.2758044302**
- epoch-24 mean relation norm: **0.0354909398**

The primary direction therefore dominates the shared update even when relation-preservation gradients are not anti-aligned.

## 2. Frozen physical / inference surface

Trainable:
- A13 final-attention LoRA: **16,384**
- shared 256->128 projection: **32,768**
- total: **49,152**

Frozen:
- original A13
- HIRACore
- relation refinement
- reliability/calibration
- adaptive budget

No learned optimizer/router/head is added.

Inference remains exactly:
`fused = 0.5 * standardized(triadic) + 0.5 * standardized(relation)`
with fusion epsilon **1e-6**.

## 3. Frozen loss partition

Identical to S16:
- primary block = fused decision + 0.05 option alignment + 0.25 fused cross-view JS
- relation block = 0.10 relation CE + 0.15 signature canonicalization
- swap coefficient 0.25, margin 0.20
- option temperature 0.10
- role/pair/relation temperatures 0.10
- signature separation margin 0.20

S15/S16 relation-logit detach remains in the fused primary path.

## 4. Frozen norm-balanced gradient rule

Fixed epsilon: **1e-12**.

Let raw shared-surface gradients be `g_p`, `g_r` with norms `n_p`, `n_r`.

Special cases:
- if `n_p == 0` and `n_r == 0`: zero update
- if `n_r == 0`: return raw `g_p` unchanged
- if `n_p == 0`: return raw `g_r` unchanged

Otherwise:
`u_p = g_p / (n_p + eps)`
`u_r = g_r / (n_r + eps)`

Let `d = dot(u_p,u_r)`.

If `d < 0`:
`u_p' = u_p - d/(||u_r||^2 + eps) * u_r`
Else:
`u_p' = u_p`

Direction:
`v = u_p' + u_r`

Reference scale:
`s = 0.5 * (n_p + n_r)`

Final shared update:
`g = s * v / (||v|| + eps)`

No fitted coefficient.
No moving average.
No gradient-history state.
No per-module weighting.

Then existing global clip **1.0** and AdamW step.

## 5. Required operator invariants

- zero learned/module state
- finite input validation
- equal raw objective magnitude cannot dominate solely by norm
- conflicting normalized primary component has non-negative post-projection dot with relation direction
- zero-relation returns exact primary
- zero-primary returns exact relation
- common positive joint rescaling is equivariant
- same objective gradients return a collinear update
- inference unchanged

## 6. A0

16 wholly fresh English semantic cases.

Must prove:
- exact A13 token/pooled identity
- raw triadic logit/choice identity
- S17 inference equals S16/S14
- exact 49,152 physical surface
- A0 runtime trainable = 0
- balancer learned params/state = 0
- full-K/state-once/option permutation
- mass error <=1e-6
- a real 49,152-parameter gradient probe recording raw norms, normalized cosine, conflict, reference scale and final norm
- A0 never used for selection

Only qualified A0 may open TRAIN/DEV.

## 7. Fresh TRAIN / DEV

- TRAIN 768 wholly fresh semantic cases
- DEV 192 wholly fresh semantic cases
- 12 fresh domains
- K=4
- two state views
- two question wording views per semantic query
- two semantic views per option

Forbidden:
- S0-S16 A0/TRAIN/DEV rows
- M5 final/confirmatory
- W29-W34 sealed rows

## 8. Optimizer

- seed **27001**
- AdamW
- 24 epochs
- batch 16 semantic cases
- lr 2e-4
- weight decay 0.01
- post-balance global grad clip 1.0

## 9. Frozen DEV selection order

1. fused paired both-correct
2. fused canonical accuracy
3. relation canonical accuracy
4. relation signed margin
5. fused signed margin
6. question-swap
7. fused cross-view agreement
8. signature cosine
9. signature margin
10. lower canonical decision loss
11. earlier epoch

## 10. Frozen DEV_READY gate

All required:
- fused canonical >= 0.85
- paired >= 0.75
- question-swap >= 0.80
- fused cross-view agreement >= 0.95
- JS <= 0.05
- fused margin >= 0.15
- relation canonical >= 0.80
- relation margin >= 0.15
- signature cosine >= 0.90
- signature margin >= 0.15
- option-order flip <=0.02
- mass error <=1e-6
- full-K/state-once/relation-delta-zero
- exact 49,152
- original A13/HIRACore frozen
- no learned balancing/fusion/canonicalizer/downstream params

Scientific FAIL is valid.

No post-DEV:
- balancing formula changes
- reference-scale changes
- epsilon/coefficient/temperature tuning
- seed/LR/template retry
- gate weakening

Only DEV_READY may open sealed confirmation.
