# HIRA V1 S21 A0 receipt — Role-Gated Content Triadic Scoring

Status: **QUALIFIED / DIAGNOSTIC ONLY**

Issue: #225  
PR: #226

## Authority

- workflow run: `36701343136`
- artifact: `11089967708`
- artifact digest: `sha256:27219a7f99f8ef8f6f2552ada65ed97e6a99a6bc2234147177fdffdbb365c31b`
- exact head: `8eafefb9cc49dc1c8e8662b760b0526742645117`
- outcome: `HIRA_V1_S21_A0_ROLE_CONTENT_READY`

The earlier run `36692441039` is not a scientific exposure: it failed in unit `py_compile` before the A0 job ran and produced no authority artifact.

## Frozen surface

- A13 LoRA: **16,384**
- shared projection: **32,768**
- physical trainable surface: **49,152**
- runtime trainable during A0: **0**
- original A13 trainable: **0**
- role/content factorization added params: **0**
- relation canonicalizer added params: **0**
- fusion added params: **0**
- role temperature: **0.10**
- role weight: **0.50**
- content weight: **0.50**

## Identity / mechanics

- A13 token output identity: PASS
- A13 pooled output identity: PASS
- baseline choice identity rate: **0.46875**
- baseline logit identity rate: **0.0**

The low baseline identity is expected because S21 intentionally changes the primary inference operator.

- full-K: PASS
- relation refinement: off
- state encode calls: **32** for 16 × two state views
- option-order flip rate: **0.0**
- fused option-order flip rate: **0.0**
- max probability-mass error: **1.1920928955e-07**
- fused max probability-mass error: **1.1920928955e-07**
- fusion expert-swap max abs: **0.0**
- fusion vs S14 forward max abs: **0.0**

## Synthetic role/content court

PASS.

Question-role A:
- selected option: **0**

Question-role B:
- selected option: **2**

Margins under question A:
- correct same-role/same-content vs same-role/wrong-content: **+0.2051138282**
- correct same-role/same-content vs wrong-role/same-content: **+0.5093060732**

Additional:
- option permutation max abs: **0.0**
- degenerate residual geometry finite: PASS
- question A state-role max weight: **0.9959511757**
- question B state-role max weight: **0.9959511757**

This proves the preregistered structural mechanism exists on a controlled tensor court.

## Real shared-gradient court

- primary gradient norm: **3.1727573872**
- relation gradient norm: **0.5178444386**
- raw norm ratio primary/relation: **6.1268542268**
- normalized pre-dot: **+0.4964441359**
- normalized post-dot: **+0.4964441359**
- conflict: **false**
- combined norm: **1.8453009129**
- physical shared surface: **49,152**

Gradient routing:
- primary triadic logit gradient nonzero: PASS
- primary direct relation-logit gradient zero: PASS
- relation auxiliary gradient nonzero: PASS

## Diagnostic semantic metrics

A0 is not model-selection evidence.

- primary canonical accuracy: **0.375**
- primary paraphrase accuracy: **0.4375**
- raw primary cross-view agreement: **0.5**
- primary canonical mean gold margin: **-0.0227174778**
- relation canonical accuracy: **0.34375**
- relation paraphrase accuracy: **0.40625**
- relation canonical margin: **-0.2580002248**
- fused canonical accuracy: **0.375**
- fused paraphrase accuracy: **0.4375**
- fused paired both-correct: **0.0**
- fused cross-view agreement: **0.5625**
- fused mean JS: **0.0434330814**
- fused canonical margin: **-0.5857416391**
- same-option signature cosine: **0.6897160411**
- signature same-vs-wrong margin: **-0.0023326110**

These numbers do not authorize tuning because A0 is diagnostic-only.

## Authorization consequence

S21-A0 is QUALIFIED.

Fresh S21 TRAIN/DEV may be opened only after:
1. the interpretation plan is frozen;
2. TRAIN/DEV workflow is staged and exact-head CI passes.

No role temperature/weight/operator change is authorized after this A0.
