# HIRA V1 S36 contract — Native Signature Linear Correctness Readout

Status: **OPEN / PREREGISTERED BEFORE S36-A0 EXPOSURE**

Issue: #255

Parent:
- S35 issue #253 / PR #254
- S35 outcome `HIRA_V1_S35_MATCHED_NATIVE_DEV_COMPLETE`
- merged main `8001e2a7de9f90dddd5dcd7d315fb91e080d5d54`
- preregistered S35 interpretation: **Case B**

## 1. Scientific question

S35 established a clean transport/correctness decoupling.

Native 256D relation geometry materially improved:
- same-option cross-wording signature cosine;
- same-option vs other-option signature discrimination;
- relation/fused cross-view agreement;
- fused JS.

But the fixed S13-equivalent native relation logits became less correct.

The S35 signature metrics are **option-identity transport metrics**, not gold-option correctness metrics.

S36 therefore asks:

> Does the wording-stable native 256D relation signature contain a transferable **linear correctness direction** that the fixed S35/S13 scoring formula fails to read out?

S36 is a readout-localization court.
It is not permission to tune S35 geometry.

## 2. Frozen shared shell

Both arms use exact S35/S17:
- final A13 attention LoRA **16,384**
- shared primary 256->128 projection **32,768**
- original A13 frozen
- HIRACore frozen
- S14 equal standardized full-K fusion
- S15 fused-primary relation-logit detach
- S17 primary/relation partition
- S17 relation-priority norm-balanced gradient rule
- local relation CE coefficient **0.10**
- local relation signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- state-once
- full-K
- opaque option IDs
- exact S35 native 256D relation representation
- no projected relation path in either arm
- no global cross-case contrastive objective

S35 native operator remains frozen in formulation:
- native dimension **256**
- role temperature **0.10**
- pair temperature **0.10**
- contrastive temperature **0.10**
- exact S13-equivalent native relation algorithm.

## 3. Control

Exact S35 native relation logits/signatures.

Trainable:
- A13 LoRA **16,384**
- shared primary projection **32,768**

Total:
**49,152**.

No relation readout parameters.

## 4. Treatment

Exact S35 native relation logits/signatures plus one shared residual correctness readout.

Let native option signature be:
`s_k in R^256`.

One shared vector:
`w in R^256`.

Frozen readout:
- zero initialization;
- no bias;
- no nonlinearity;
- no LayerNorm;
- no learned scale;
- residual scale **1.0**;
- no option-specific parameters;
- no option-ID embeddings;
- no domain-specific parameters;
- no K-specific parameters.

Treatment:
`readout_k = s_k dot w`

`logit_k = native_logit_k + readout_k`

Added parameters:
**256**.

Treatment total trainable surface:
**49,408**.

The readout is shared across arbitrary K and must be logical-option permutation equivariant.

## 5. Zero-init identity requirement

Before training:
- treatment native relation logits must equal control exactly;
- signatures must equal exactly;
- primary logits must equal exactly;
- fused logits must equal exactly;
- selected choices must match exactly.

Zero initialization is part of the scientific contract.

## 6. Gradient ownership

Treatment:
- relation block -> readout gradient: nonzero finite;
- primary block -> readout gradient: exactly zero;
- native relation block -> shared projection direct gradient: exactly zero;
- primary block -> shared projection gradient: nonzero finite;
- relation block -> A13 LoRA gradient: nonzero finite.

No readout-only optimizer or LR.

The readout uses the same AdamW LR/weight decay as the other trainable treatment parameters.

## 7. Required S36-A0

Fresh S36-A0 rows only.

Must prove:
- readout parameter count exactly **256**;
- control trainable surface **49,152**;
- treatment trainable surface **49,408**;
- original A13 trainable params **0**;
- HIRACore trainable params **0**;
- zero-init relation-logit identity max abs **0**;
- signature identity max abs **0**;
- primary-logit identity max abs **0**;
- fused-logit identity max abs **0**;
- exact selected-choice identity rate **1.0**;
- readout relation-gradient L1 > 0;
- readout primary-gradient L1 = 0;
- native relation -> shared projection direct gradient L1 = 0;
- primary -> shared projection gradient L1 > 0;
- relation -> LoRA gradient L1 > 0;
- logical-option permutation equivariance;
- arbitrary-K support for at least K=3 and K=7 controlled tensors;
- native representation remains exactly shared-projection independent;
- full-K / state-once / probability-mass / checkpoint mechanics PASS.

A0 semantic scores are diagnostic only.

## 8. Fresh matched TRAIN/DEV

Only after:
1. qualified A0;
2. frozen A0 receipt;
3. frozen interpretation plan;
4. frozen trainer/workflow;
5. exact-head generic CI;
6. separate one-shot TRAIN/DEV marker.

Frozen intended authority:
- seed **57001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S36 domains
- K=4
- two state views
- two question wording views per semantic query
- two option semantic views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- same per-epoch row order across arms
- independent runtime/optimizer state

Freshness:
- exact S0-S35 exposed rows excluded;
- S36-A0 rows excluded;
- M5 final/confirmatory rows excluded;
- W29-W34 sealed rows excluded.

## 9. Selector / gates

Use exact S17 selector independently per arm.

Keep existing DEV_READY gates:
- fused canonical >= .85
- paired >= .75
- question-swap >= .80
- fused agreement >= .95
- fused JS <= .05
- fused canonical margin >= .15
- relation canonical >= .80
- relation margin >= .15
- same-option signature cosine >= .90
- signature discrimination >= .15
- mechanics/capacity/freeze gates PASS.

Treatment additionally requires exact readout capacity **256**.

## 10. Frozen interpretation

### A — coherent linear-readout gain
Treatment improves relation accuracy/margins and fused endpoints while preserving native transport.
Then the readout family remains viable for a new fresh confirmation track.

### B — correctness improves but transport/fusion regresses
Close S36 as split evidence.
Do not tune scale/bias/nonlinearity on exposed DEV.

### C — little/no correctness gain
Conclude the stable native signature does not expose a transferable single linear correctness direction.
Move beyond a one-vector readout.

### D — treatment DEV_READY
Freeze immediately.
No second S36 DEV.
Separate confirmation required before external matched evaluation.

## 11. Stop rule

After one S36 DEV:
- no residual-scale tuning;
- no bias;
- no MLP/nonlinearity;
- no low-rank or multi-vector readout;
- no readout-only LR;
- no projected/native mixing;
- no native width/normalization/temperature tuning;
- no seed/LR/epoch/batch retry;
- no selector/gate weakening;
- no second DEV.

Scientific FAIL is valid.
No Laya/Jev benchmark before DEV_READY.
