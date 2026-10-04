# HIRA V1 S53 contract — Token-Level Query↔Option Late Interaction

Status: **PREREGISTERED / NO S53-A0 EXPOSURE**

Issue: #289

Parent:
- S52 fresh run `37204041655`
- artifact `11303604910`
- Case **D**
- global pooled query-relation canonicalization worsened fused correctness and stability.

## Scientific question

> Does preserving token-local query evidence and forming an option-conditioned query context improve paraphrase stability without sacrificing useful correctness, compared with the current single pooled-query readout?

## Frozen backbone

Reuse:
- sealed S51 persisted native authority run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- query-free state↔option identity
- one encoder/state-once
- immutable shared cache
- native trainability **0**.

Native retraining is forbidden.

## Reference

Exact pooled-query private correction:
- query-free option identity
- masked-mean normalized query vector
- correction adapter A/B + bilinear W.

## Treatment

No pooled global query is available to the correction path.

For option identity `s_k` and normalized cached query token `q_t`:

`score(k,t)=cosine(s_k,q_t)/0.10`

Mask inactive tokens and softmax over query tokens:

`weight(k,t)=softmax_t(score(k,t))`

Per-option query context:

`c_k=normalize(sum_t weight(k,t) q_t)`

Use `c_k` in the exact same correction adapter/bilinear surfaces.

## Matched parameter surface

Both arms:
- adapter A **32,768**
- adapter B **16,384**
- bilinear W **65,536**
- correction total **114,688**
- identity trainable params **0**
- treatment token-context params **0**
- total private trainable **114,688**
- bit-identical correction initialization.

## Frozen token-context operator

- cosine token↔option score
- temperature **0.10**
- masked softmax weighted sum
- normalization epsilon **1e-12**
- no trainable token projection
- no global-query bypass.

## Required S53-A0

Ownership/capacity:
- reference/treatment correction count 114,688;
- treatment late-interaction params 0;
- identity params 0;
- bit-identical correction initialization;
- native/cache gradient 0;
- one encoder/state-once.

Mechanics:
- K=3/7/255;
- query padding/mask invariance;
- query-token permutation invariance under joint token+mask permutation;
- option permutation equivariance;
- all-masked query rejected;
- per-option context finite and unit normalized;
- informative-token perturbation changes treatment context/logits;
- treatment correction path succeeds even if inherited pooled `query_summary` is disabled;
- full-K probability mass <=1e-6.

## Fresh S53 authority

Intended:
- seed **74001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S53 domains
- K=4
- private epochs 24
- batch 16
- one DEV only.

## Prohibited

No:
- temperature sweep
- aggregation-family sweep
- trainable token projection
- pooled-query bypass in treatment
- native retraining
- identity variant
- correction capacity change
- selector change
- retry
- gate weakening
- second S53 DEV.

Scientific failure is valid.
