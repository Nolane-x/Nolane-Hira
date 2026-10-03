# HIRA V1 S45 A0 receipt — Cross-View Consistent Private Correction

Status: **QUALIFIED**

Run: `37128007831`  
Artifact: `11275950524`  
Artifact digest: `sha256:90ece8fe112a14c44b29e03d217efa372f857b47d1bfa52cd58a208e75ed924e`  
Authorization head: `d9245996126cfe668131393e4ebab2f38cb55d8a`

Outcome:
`HIRA_V1_S45_A0_CROSS_VIEW_CONSISTENT_PRIVATE_CORRECTION_READY`

## Frozen architecture

- native trainable: **49,152**
- private A: **32,768**
- private B: **16,384**
- private adapter total: **49,152**
- full W: **65,536**
- correction-only: **114,688**
- treatment total: **163,840**
- hidden width: **64**
- one encoder pass / state-once

## Frozen objective

`L_corr = 0.10 * CE_corr + 0.25 * symmetric_js_divergence(corrected_canonical, corrected_paraphrase)`

The JS implementation is the pre-existing Hira `symmetric_js_divergence` primitive.

## Cross-view JS mechanics

- identical-logit JS: **0**
- deterministic non-identical JS: **0.23387247323989868**
- JS-only A gradient L1: **0.0004649879701901227**
- JS-only B gradient L1: **0.011057455092668533**
- JS-only W gradient L1: **0.5996381044387817**
- JS-only native-runtime gradient L1: **0**

Therefore the S45 consistency term is live on the full private A/B/W surface and remains exactly detached from the native runtime.

## Native ownership and identity

Exact zeros:
- matched native gradient max abs
- matched one-step native parameter max abs
- matched one-step native output max abs
- correction -> native gradient L1
- correction -> reference gradient L1
- native objective -> correction gradient L1
- warm correction -> native / LoRA / projection gradient L1

Zero-init:
- corrected relation delta: **0**
- private residual: **0**
- fused logit delta: **0**
- selected choice identity: **1.0**
- private-signature max abs delta: **2.9802322387695312e-8**

## Warm-start liveness

True zero B/W remains deterministic:

- step 1: **W live**, A/B zero
- step 2: **W+B live**
- step 3: **W+B+A live**

Step-3 A gradient L1: **0.0001729724754113704**  
Step-3 B gradient L1: **0.006173266097903252**  
Step-3 W gradient L1: **0.4221423864364624**

Private nonlinear family remains distinct from W-only:
- private-vs-W-only residual max abs: **0.00020124763250350952**

## Structural courts

PASS:
- arbitrary K=3
- arbitrary K=7
- logical option permutation
- question permutation
- masked padding
- checkpoint roundtrip
- probability mass
- full-K
- projection independence

A0 semantic accuracy is diagnostic only and **not used for model selection**.

## Authorization consequence

S45-A0 is qualified.

This receipt authorizes staging of the preregistered S45 matched TRAIN/DEV court, but does **not** itself authorize DEV exposure. A separate exact-head CI pass and one-shot TRAIN/DEV marker are still required.
