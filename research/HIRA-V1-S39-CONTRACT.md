# HIRA V1 S39 contract — Gradient-Isolated Full Bilinear Correctness Readout

Status: **OPEN / PREREGISTERED BEFORE S39-A0 EXPOSURE**

Issue: #261

Parent:
- S38 issue #259 / PR #260
- S38 merged main `f4111e671df35d58b8b43707f5d436fc49dad86c`
- S38 outcome `HIRA_V1_S38_MATCHED_FULL_BILINEAR_DEV_COMPLETE`
- preregistered S38 interpretation: **Case B**

## 1. Scientific question

S38 proves that a full 256x256 native query-signature bilinear map contains substantial correctness signal, but unrestricted joint optimization damages transport/fusion stability.

S39 asks:

> Can the S38 correctness gain be retained when the bilinear correction head is prevented from steering the native runtime trajectory?

S39 changes gradient ownership, not readout expressivity.

## 2. Frozen shared shell

Both arms use:
- exact S35 native 256D relation representation/signature
- exact S17 primary/fusion shell
- final A13 attention LoRA **16,384**
- shared primary 256->128 projection **32,768**
- original A13 frozen
- HIRACore frozen
- exact S38 query summary:
  - adapted A13 question tokens
  - masked arithmetic mean
  - L2 normalization
  - epsilon **1e-12**
- S14 equal standardized full-K fusion
- S15 relation-logit detach in fused-primary objective
- S17 primary/relation partition
- S17 relation-priority norm-balanced runtime gradients
- local native relation CE coefficient **0.10**
- local signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- state-once
- full-K
- opaque option IDs
- no projected relation path

## 3. Control

Exact native S35/S17 training trajectory.

Runtime training:
- primary block uses native relation evidence
- relation block uses exact native relation CE + signature canonicalization
- norm-balanced primary/native-relation gradients
- runtime gradient clip **1.0**

Trainable:
- A13 LoRA **16,384**
- shared primary projection **32,768**

Total:
**49,152**.

## 4. Treatment evaluation

Exact S38 bilinear evaluation family:

`residual_k = signature_k^T W q_hat`

`logit_k = native_logit_k + residual_k`

where:
- `W in R^(256 x 256)`
- exact zero initialization
- no bias
- no nonlinearity
- no factorization
- no rank constraint
- no learned scale
- residual scale **1.0**
- no option/domain/K-specific parameters

Added parameters:
**65,536**.

Treatment total trainable surface:
**114,688**.

## 5. Hard gradient isolation

### 5.1 Runtime route

Treatment runtime training is numerically identical to control:
- exact same primary block
- exact same native relation block
- exact same rows/order
- exact same norm-balanced gradient rule
- exact same runtime gradient clipping threshold **1.0**
- bilinear residual is NOT inserted into runtime-training losses

### 5.2 Correction route

On the same encoded batch:
- native relation logits are detached
- native signatures are detached
- native query tokens/summary are detached
- correction logits = detached native logits + bilinear residual
- correction objective = **0.10 * mean(canonical relation CE, paraphrase relation CE)**
- correction objective updates **W only**
- no correction canonicalization term

### 5.3 Update rule

One AdamW optimizer may contain runtime params and W:
- same LR **2e-4**
- same weight decay **0.01**
- no W-only LR/scheduler

Before `optimizer.step()`:
- runtime params receive the existing norm-balanced primary/native-relation gradient
- runtime params are clipped as one set at **1.0**
- W receives correction-objective gradient only
- W is clipped as its own set at **1.0**

W is excluded from runtime norm-balancing.
Runtime params are excluded from correction gradients.

## 6. Strong trajectory invariant

Given identical initialization, rows and order:
- control and treatment LoRA state must remain exactly equal under the same backend
- control and treatment primary projection state must remain exactly equal
- native relation logits/signatures before applying W must remain equal

S39 treatment is therefore a correction adapter on a shared native trajectory, not representation co-adaptation.

## 7. Required S39-A0

Fresh S39-A0 only.

Must prove:
- zero-init evaluation identity for relation/signature/primary/fused outputs
- selected-choice identity **1.0**
- W params exactly **65,536**
- control/treatment trainable surfaces **49,152 / 114,688**
- correction CE -> W gradient finite and nonzero
- correction CE -> off-diagonal W gradient finite and nonzero
- correction CE -> A13 LoRA gradient exactly **0**
- correction CE -> shared projection gradient exactly **0**
- correction CE -> HIRACore gradient exactly **0**
- primary block -> W gradient exactly **0**
- native relation block -> W gradient exactly **0**
- with a nonzero W present, runtime primary/native gradients exactly match control
- one matched synthetic optimizer update leaves runtime params equal across arms while W changes
- nonzero-W query intervention changes residual
- nonzero-W signature intervention changes residual
- logical-option permutation equivariance
- question-token permutation invariance
- masked query-padding invariance
- arbitrary K at least K=3/K=7
- native representation remains shared-projection independent
- checkpoint / probability / full-K / state-once mechanics PASS

A0 semantic scores are diagnostic only.

## 8. Fresh matched TRAIN/DEV

Only after:
1. qualified A0
2. frozen A0 receipt
3. frozen interpretation plan
4. frozen trainer/workflow
5. exact-head generic CI
6. separate one-shot TRAIN/DEV marker

Frozen intended authority:
- seed **60001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S39 domains
- K=4
- two state views
- two question wording views per semantic query
- two option semantic views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- runtime grad clip **1.0**
- W grad clip **1.0**
- identical rows/order across arms
- independent runtime/optimizer state

Freshness:
- exact S0-S38 exposed rows excluded
- S39-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 9. Selector / gates

Use exact existing S35/S38 selector independently per arm.
Keep existing DEV_READY gates unchanged.

Treatment additionally requires:
- exact W capacity **65,536**
- runtime trajectory identity to matched control

## 10. Frozen interpretation

### A — correctness retained + transport recovers
Relation/fused correctness remains meaningfully above control while agreement/JS/signature discrimination recover from S38 Case-B failure.

Then gradient isolation remains viable for a wholly fresh confirmation track.

### B — correctness retained but transport remains degraded
Gradient isolation is insufficient.
Close S39.

### C — transport recovers but correctness gain largely disappears
S38 gain depended on representation co-adaptation.
Close isolated readout family.

### D — treatment DEV_READY
Freeze immediately.
No second S39 DEV.
Separate confirmation required before external matched evaluation.

## 11. Stop rule

After one S39 DEV:
- no partial detach coefficient
- no gradient-mixing coefficient
- no W-only LR/scheduler
- no rank/factorization retry
- no residual-scale/bias/nonlinearity retry
- no projected/native mixing
- no seed/LR/epoch/batch retry
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.
No Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
