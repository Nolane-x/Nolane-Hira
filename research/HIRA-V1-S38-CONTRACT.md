# HIRA V1 S38 contract — Full Bilinear Native Query-Signature Correctness Readout

Status: **OPEN / PREREGISTERED BEFORE S38-A0 EXPOSURE**

Issue: #259

Parent:
- S37 issue #257 / PR #258
- S37 merged main `deaf7778caa0594affa992f076191d5e7aea9577`
- S37 outcome `HIRA_V1_S37_MATCHED_QUERY_GATED_DEV_COMPLETE`
- preregistered S37 interpretation: **Case C**

## 1. Scientific question

S36 ruled out one global query-independent 256D correctness direction.

S37 ruled out one query-conditioned **diagonal** bilinear interaction:

`(signature * query_summary) @ w`.

S38 asks:

> Does correctness require cross-coordinate interactions between the native option signature and native query summary?

This is a representation/readout localization court.
It is not permission to add a generic MLP.

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

Query summary is frozen exactly as S37:
- adapted A13 question tokens
- masked arithmetic mean
- L2 normalize
- epsilon **1e-12**
- no learned query projection
- no attention reducer
- no positional dependence

## 3. Control

Exact S35 native relation logits/signatures.

Trainable:
- A13 LoRA **16,384**
- shared primary projection **32,768**

Total:
**49,152**.

## 4. Treatment

Exact same native relation logits/signatures plus one full bilinear residual correctness map.

For normalized native question summary `q in R^256` and native option signature `s_k in R^256`:

`residual_k = s_k^T W q`

`logit_k = native_logit_k + residual_k`

Frozen:
- `W in R^(256 x 256)`
- W initialized exactly zero
- no bias
- no MLP/nonlinearity
- no learned scale
- residual scale **1.0**
- no factorization
- no rank constraint
- no option-specific parameters
- no option-ID embeddings
- no domain-specific parameters
- no K-specific parameters

Added parameters:
**65,536**.

Treatment total trainable surface:
**114,688**.

## 5. Why full bilinear now

S37 is the diagonal special case:
`s^T diag(w) q`.

S38 permits off-diagonal terms:
`s_i W_ij q_j`.

This is the smallest conceptually complete test of whether the missing correctness relation requires cross-coordinate interactions while preserving:
- exact zero-init identity
- one shared scorer
- arbitrary K
- option-permutation equivariance
- direct first-order gradients at initialization

## 6. Zero-init identity

Before training:
- treatment relation logits equal control exactly
- signatures equal exactly
- primary logits equal exactly
- fused logits equal exactly
- selected choices match exactly

## 7. Bilinear discriminator

At zero W:
- relation loss must produce a finite nonzero gradient on W
- off-diagonal gradient L1 must be >0

With a fixed nonzero off-diagonal W:
- changing query content while holding signatures fixed changes residual
- changing signature content while holding query fixed changes residual
- permuting query-token order does not change output
- adding masked query padding does not change output

## 8. Gradient ownership

Treatment:
- relation block -> W gradient: nonzero finite
- primary block -> W gradient: exactly zero
- off-diagonal relation -> W gradient: nonzero
- native relation block -> shared projection direct gradient: exactly zero
- primary block -> shared projection gradient: nonzero finite
- relation block -> A13 LoRA gradient: nonzero finite

No W-only optimizer or LR.

## 9. Required S38-A0

Fresh S38-A0 rows only.

Must prove:
- bilinear parameter count exactly **65,536**
- control trainable surface **49,152**
- treatment trainable surface **114,688**
- original A13 trainable params **0**
- HIRACore trainable params **0**
- zero-init relation-logit identity max abs **0**
- signature identity max abs **0**
- primary-logit identity max abs **0**
- fused-logit identity max abs **0**
- selected-choice identity rate **1.0**
- W relation-gradient L1 > 0
- W primary-gradient L1 = 0
- off-diagonal W relation-gradient L1 > 0
- native relation -> shared projection direct gradient L1 = 0
- primary -> shared projection gradient L1 > 0
- relation -> LoRA gradient L1 > 0
- query intervention residual change > 0
- signature intervention residual change > 0
- logical-option permutation equivariance
- question-token permutation invariance
- masked query-padding invariance
- arbitrary K at least K=3 and K=7
- native representation remains shared-projection independent
- full-K / state-once / probability / checkpoint mechanics PASS

A0 semantic scores are diagnostic only.

## 10. Fresh matched TRAIN/DEV

Only after:
1. qualified S38-A0
2. frozen A0 receipt
3. frozen interpretation plan
4. frozen trainer/workflow
5. exact-head generic CI
6. separate one-shot TRAIN/DEV marker

Intended authority:
- seed **59001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S38 domains
- K=4
- two state views
- two question wording views per semantic query
- two option views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- same rows/order across arms
- independent runtime/optimizer state

Freshness:
- exact S0-S37 exposed rows excluded
- S38-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 11. Selector / gates

Use exact existing S17/S37 selector independently per arm.

Keep existing DEV_READY gates unchanged.
Treatment additionally requires exact bilinear capacity **65,536**.

## 12. Frozen interpretation

### A — coherent cross-coordinate correctness gain
Treatment improves relation canonical/paraphrase accuracy and margins, improves both fused wording endpoints, and preserves native transport.

Then full bilinear readout remains viable for a new fresh confirmation track.

### B — correctness improves but transport/fusion regresses
Close S38 as split evidence.
Do not tune exposed DEV.

### C — little/no direct correctness gain
Conclude the full bilinear query-signature readout is insufficient.
Close this readout family rather than expanding capacity on exposed DEV.

### D — treatment DEV_READY
Freeze immediately.
No second S38 DEV.
Separate confirmation required before external matched evaluation.

## 13. Stop rule

After one S38 DEV:
- no matrix regularization tuning
- no rank/factorization retry
- no residual-scale tuning
- no bias/nonlinearity/MLP
- no W-only LR/scheduler
- no projected/native mixing
- no seed/LR/epoch/batch retry
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.
No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
