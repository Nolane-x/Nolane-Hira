# HIRA V1 S49 A0 receipt — Private Query-Free State–Option Identity

Status: **QUALIFIED**

Run: `37178164136`  
Artifact: `11293039754`  
Artifact digest: `sha256:ce07ed474c21185d5a4316550e116606d7c09b90c808aea028354f2f046678e0`  
Authorization head: `b793e54e8c969a110859a9966d99ed3f69ec7ffd`

Outcome:
`HIRA_V1_S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_READY`

## Frozen surface

- native trainable: **49,152**
- private correction: **114,688**
- treatment total: **163,840**
- identity trainable parameters: **0**
- identity API question inputs: **absent**
- second encoder pass: **false**

## Identity mechanics

- K=3 / K=7 / K=255: PASS
- arbitrary question substitution identity error: **0**
- option permutation error: **0**
- state-token permutation error: **2.98e-8**
- option-token permutation error: **2.24e-8**
- option-view permutation error: **0**
- masked-padding error: **2.98e-8**
- checkpoint roundtrip: **exact**
- probability-mass error: **1.19e-7**

Raw query remains live after fixed identity:
- corrected-logit max difference: **0.0232448**
- controlled probe changes ranking: **true**

## Actual runtime diagnostics

- one encoder batch: **true**
- state-view encodes: **32**
- identity same-option cross-state-view cosine: **0.799725**
- identity same-vs-strongest-wrong margin: **-0.005606**

These semantic diagnostics are A0-only and are not model-selection evidence.

## Ownership

- matched native gradient max abs: **0**
- matched native one-step parameter max abs: **0**
- matched native one-step output max abs: **0**
- correction → native runtime gradient: **0**
- native objective → correction gradient: **0**
- JS-only native gradient: **0**
- JS-only A/B/W gradients: **0.000296318 / 0.006569384 / 0.356110632**
- W → B → A warm-start remains live.

## Interpretation

S49 mechanically qualifies.

The treatment successfully enforces the intended causal split:
- option identity itself has no query input;
- raw query still changes the final private correctness readout;
- no capacity was added;
- native trajectory ownership remains exact.

The A0 identity discrimination margin is weak on the diagnostic suite, so the fresh matched S49 court is necessary; no positive scientific claim is made from A0 alone.
