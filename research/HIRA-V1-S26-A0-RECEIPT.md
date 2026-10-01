# HIRA V1 S26 A0 receipt — Factorized Role-Value Relation Signatures

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #235  
PR: #236

## Canonical authority

- run `36849791728`
- artifact `11155305741`
- artifact digest `sha256:1cfe67edb48649a3cf7931403f1b3abd50b47d0030f05c0242a18b11f52564be`
- authority head `0df6f21b3e5b3a065e19a83c975e3ed3d4da7411`
- outcome `HIRA_V1_S26_A0_FACTORIZED_RELATION_READY`

Earlier run:
- `36848720848`: non-result harness abort caused by comparing the synthetic 4→4 court's 8D factorized signature against the real M4 256D signature.

That run produced no qualified receipt/artifact and has no scientific authority.

## Frozen inherited surface

- shared A13 LoRA: **16,384**
- primary-private 256→128 projection: **32,768**
- relation-private 256→128 projection: **32,768**
- exact physical trainable surface: **81,920**
- S26 relation operator added params: **0**
- A0 runtime trainable params: **0**
- original A13 trainable params: **0**
- HIRACore trainable params: **0**
- private projection storage distinct: PASS

Primary path identity against inherited S25 initialization: **PASS**.

The S26 relation intervention is genuinely active:
- relation intervention max abs: **5.0234084129**

## Factorized relation court

Real M4 factorized signature:
- dimension: **256**

Synthetic dimensional court:
- projection dimension: **4**
- signature dimension: **8**
- exact factorization rule: signature width = **2 × relation width**

Hard-negative quadrants:
- correct vs same-role/wrong-value margin: **+5.0**
- correct vs wrong-role/same-value margin: **+9.9998397827**
- correct vs wrong-role/wrong-value margin: **+10.0**

Thus the pre-DEV structural hypothesis is mechanically represented rather than collapsing the role/value quadrants.

## Gradient ownership

Cross-private leakage:
- primary → relation-private max abs: **0**
- relation → primary-private max abs: **0**

Own-private gradients:
- primary-private L1: **239.5990600586**
- relation-private L1: **68.0608596802**

Shared-LoRA gradients:
- primary block L1: **7.0240483284**
- relation block L1: **2.1269371510**

Shared neutral-bisector:
- primary norm: **0.1798840463**
- relation norm: **0.0553057417**
- projection coefficient: **0**
- combined norm: **0.1175949052**

S25 ownership semantics remain intact.

## Full-K / permutation / numerical court

- primary option permutation max abs: **0**
- relation option permutation max abs: **0**
- signature option permutation max abs: **0**
- mapped fused option-order flip rate: **0**
- frozen-S14 fused logit permutation diagnostic: **8.9406967e-7**
- max probability-mass error: **1.1920929e-7**
- full-K: PASS
- state-once views: **32**

## A0 semantic diagnostics

Not model-selection evidence:
- primary canonical: **34.375%**
- primary paraphrase: **37.5%**
- relation canonical: **40.625%**
- relation paraphrase: **25.0%**
- fused canonical: **40.625%**
- fused paraphrase: **31.25%**
- same-option signature cosine: **0.4765717983**
- signature same-vs-strongest-wrong margin: **0.0086444849**

Component diagnostics:
- canonical mean role compatibility: **0.6050544977**
- canonical mean value compatibility: **0.1033952981**
- paraphrase mean role compatibility: **0.7652090192**
- paraphrase mean value compatibility: **0.2883956134**

These semantic values are diagnostic only and MUST NOT tune role/value weights, temperatures, losses, capacity, seed, selector or DEV gates.

## Consequence

S26-A0 is QUALIFIED.

Fresh S26 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. the S26 interpretation plan is frozen;
3. wholly fresh S26 TRAIN/DEV generator is frozen;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. a separate one-shot TRAIN/DEV authorization is committed.

No additional S26-A0 run is authorized.
