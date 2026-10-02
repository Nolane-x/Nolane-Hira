# HIRA V1 S37 contract — Query-Gated Native Signature Correctness Readout

Status: **OPEN / PREREGISTERED BEFORE S37-A0 EXPOSURE**

Issue: #257

Parent:
- S36 issue #255 / PR #256
- S36 merged main `44619ae45654c260d734e8c7609ee1fdbd256b3f`
- S36 outcome `HIRA_V1_S36_MATCHED_READOUT_DEV_COMPLETE`
- preregistered S36 interpretation: **Case C**

## 1. Scientific question

S36 showed that one shared query-independent 256D correctness vector is insufficient.

The global readout improved several fused/stability quantities, but direct relation correctness moved only weakly and fused paraphrase accuracy did not improve.

S37 asks:

> Is the missing correctness direction query-dependent rather than globally shared?

S37 is a conditioning-localization court.
It is not permission to increase readout capacity.

## 2. Frozen shared shell

Both arms use:
- exact S35 native 256D relation representation/signature
- exact S17 primary path/fusion/optimizer shell
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
- no projected relation path

## 3. Control

Exact S35 native relation logits/signatures.

Trainable:
- A13 LoRA **16,384**
- shared primary projection **32,768**

Total:
**49,152**.

## 4. Treatment

Exact same native relation logits/signatures plus one query-gated residual correctness readout.

Given adapted A13 question tokens `q_i in R^256` and mask `m_i`:

`q_mean = sum_i(m_i * q_i) / sum_i(m_i)`

`q_hat = L2Normalize(q_mean, eps=1e-12)`

For native option signature `s_k in R^256`:

`feature_k = s_k * q_hat`

One shared vector:

`w in R^256`

Readout:

`readout_k = feature_k @ w`

Treatment:

`logit_k = native_logit_k + readout_k`

Frozen:
- zero initialization
- no bias
- no nonlinearity
- no LayerNorm
- fixed L2-normalization epsilon **1e-12**
- no learned scale
- residual scale **1.0**
- no rank expansion
- no option-specific parameters
- no option-ID embeddings
- no domain-specific parameters
- no K-specific parameters

Added parameters:
**256**.

Treatment total trainable surface:
**49,408**.

## 5. Zero-init identity

Before training:
- treatment native relation logits equal control exactly
- signatures equal exactly
- primary logits equal exactly
- fused logits equal exactly
- selected choices match exactly

## 6. Query-conditioning discriminator

With a fixed nonzero readout vector:
- changing the valid query content while holding a controlled native signature fixed must change the readout residual
- permuting query token order must not change the readout
- adding masked query padding must not change the readout

This distinguishes query conditioning from S36's global readout.

## 7. Gradient ownership

Treatment:
- relation block -> readout gradient: nonzero finite
- primary block -> readout gradient: exactly zero
- native relation block -> shared projection direct gradient: exactly zero
- primary block -> shared projection gradient: nonzero finite
- relation block -> A13 LoRA gradient: nonzero finite

No readout-only optimizer or LR.

## 8. Required S37-A0

Fresh S37-A0 rows only.

Must prove:
- readout parameter count exactly **256**
- control trainable surface **49,152**
- treatment trainable surface **49,408**
- original A13 trainable params **0**
- HIRACore trainable params **0**
- zero-init relation-logit identity max abs **0**
- signature identity max abs **0**
- primary-logit identity max abs **0**
- fused-logit identity max abs **0**
- exact selected-choice identity rate **1.0**
- readout relation-gradient L1 > 0
- readout primary-gradient L1 = 0
- native relation -> shared projection direct gradient L1 = 0
- primary -> shared projection gradient L1 > 0
- relation -> LoRA gradient L1 > 0
- query intervention residual change > 0
- logical-option permutation equivariance
- question-token permutation invariance
- masked query-padding invariance
- arbitrary-K support for at least K=3 and K=7
- native representation remains exactly shared-projection independent
- full-K / state-once / probability-mass / checkpoint mechanics PASS

A0 semantic scores are diagnostic only.

## 9. Fresh matched TRAIN/DEV

Only after:
1. qualified A0
2. frozen A0 receipt
3. frozen interpretation plan
4. frozen trainer/workflow
5. exact-head generic CI
6. separate one-shot TRAIN/DEV marker

Intended authority:
- seed **58001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S37 domains
- K=4
- two state views
- two question wording views per semantic query
- two option semantic views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- same rows/order across arms
- independent runtime/optimizer state

Freshness:
- exact S0-S36 exposed rows excluded
- S37-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 10. Selector / gates

Use exact existing S17/S36 selector independently per arm.

Keep existing DEV_READY gates unchanged.
Treatment additionally requires exact readout capacity **256**.

## 11. Frozen interpretation

### A — coherent query-conditioned correctness gain
Treatment improves relation canonical/paraphrase accuracy and margins, improves fused canonical/paraphrase endpoints, and preserves native signature transport.

Then query-conditioned readout remains viable for a new fresh confirmation track.

### B — correctness improves but transport/fusion regresses
Close S37 as split evidence.
Do not tune exposed DEV.

### C — little/no direct correctness gain
Conclude one diagonal query-signature interaction is insufficient.
Move beyond a single 256D diagonal interaction.

### D — treatment DEV_READY
Freeze immediately.
No second S37 DEV.
Separate confirmation required before external matched evaluation.

## 12. Stop rule

After one S37 DEV:
- no query-summary normalization tuning
- no residual-scale tuning
- no bias
- no MLP/nonlinearity
- no rank expansion/multiple vectors
- no readout-only LR
- no projected/native mixing
- no seed/LR/epoch/batch retry
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.
No Laya/Jev benchmark before confirmed DEV_READY.
