# HIRA V1 S25 contract — Decoupled Expert Projection Surfaces

Status: **OPEN / PREREGISTERED BEFORE S25-A0 EXPOSURE**

Issue: #233

Base main:
`ce78ef07f33d22cfd039ff6f7da71a14273e4250`

S24 is frozen as `HIRA_V1_S24_RELIABILITY_FUSION_DEV_FAIL`.

## 1. Hypothesis

S21-S24 force the role-gated primary expert and canonical-relation expert through one learned 256->128 projection. Optimizer priority and post-hoc reliability weighting did not establish DEV_READY.

S25 tests exactly one representation claim:

> Keeping one shared A13/LoRA encoder while giving primary and relation experts separate private 256->128 projection surfaces reduces destructive representational coupling enough to improve fresh semantic generalization.

## 2. Frozen architecture

Shared:
- A13 encoder revision and W28 T0 source
- last-attention LoRA rank/alpha/dropout exactly S6+
- shared LoRA trainable params **16,384**
- original A13 weights frozen
- HIRACore frozen
- state-once packing unchanged

Primary expert:
- S21 role-gated content triadic operator unchanged
- private bias-free projection 256->128
- trainable params **32,768**
- initialized exactly from W28 T0

Relation expert:
- S13 CrossViewRelationCanonicalizer unchanged
- private bias-free projection 256->128
- trainable params **32,768**
- initialized bit-identically from the same W28 T0

Total physical trainable surface:
**81,920**.

No learned router, fusion gate, calibrator, head, expert coefficient or temperature is added.

## 3. Fusion

S24 reliability weighting is CLOSED.

S25 restores frozen S14 fusion:
- center/RMS standardize each full-K expert independently
- exact equal **0.5 / 0.5** average
- epsilon **1e-6**
- full-K retained
- relation logits detached from the fused-primary objective

Thus S25 changes representation ownership only.

## 4. Gradient ownership

Let:
- `L_p` = fused decision + option alignment + fused cross-view JS primary block
- `L_r` = relation CE + signature canonicalization relation block

Shared LoRA:
- receives gradients from both `L_p` and `L_r`
- combine only these shared gradients with S23 neutral-bisector norm balancing
- epsilon **1e-12**

Primary private projection:
- receives only `L_p`
- direct gradient, no neutral-bisector mixing

Relation private projection:
- receives only `L_r`
- direct gradient, no neutral-bisector mixing

Forbidden:
- primary block -> relation-private update
- relation block -> primary-private update
- concatenating private gradients into the shared neutral-bisector court

## 5. Frozen training objective

Exactly S23, except projection ownership:
- fused canonical + paraphrase CE
- swap coefficient **0.25**
- swap margin **0.20**
- option alignment **0.05**
- option temperature **0.10**
- fused cross-view symmetric JS **0.25**
- relation CE **0.10**
- signature canonicalization **0.15**
- signature separation margin **0.20**
- role temperature **0.10**
- role/content weights **0.50 / 0.50**

## 6. Frozen optimizer shell

- AdamW lr **2e-4**
- weight decay **0.01**
- batch size **16**
- epochs **24**
- grad clip **1.0**
- shared LoRA neutral-bisector epsilon **1e-12**
- private gradients remain direct

No asymmetric gradient projection.

## 7. S25 A0

Use **16 wholly fresh English S25-A0 cases**.

A0 must prove:
- exact physical trainable surface **81,920**
- shared LoRA **16,384**
- primary projection **32,768**
- relation projection **32,768**
- primary/relation projections are distinct storage
- both are bit-identical to W28 T0 at initialization
- primary inference at initialization matches frozen S21/S23 primary
- relation inference at initialization matches frozen S13/S23 relation
- equal S14 fusion identity at initialization
- primary block -> relation-private gradient exactly zero
- relation block -> primary-private gradient exactly zero
- own-private gradients nonzero on controlled court
- both blocks produce nonzero shared-LoRA gradients on controlled court
- shared neutral-bisector court finite and deterministic
- full-K/state-once/option permutation preserved
- original A13/HIRACore frozen
- no downstream learned state beyond the two projections + shared LoRA

A0 semantic accuracy is diagnostic only.

## 8. Fresh TRAIN / DEV

To be generated only after A0 qualifies:
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S25 domains
- K=4
- two state views
- two question views
- two option views
- one preregistered seed

Forbidden:
- exact S0-S24 exposed rows
- S25-A0 rows
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 9. DEV protocol

Before fresh DEV exposure, freeze:
- exact S25 TRAIN/DEV generator
- seed
- selection key
- DEV_READY gates
- interpretation plan
- workflow
- one-shot authorization

No post-DEV:
- capacity change
- projection ownership change
- shared/private optimizer change
- seed/LR/template retry
- fusion change
- gate weakening

Scientific FAIL is valid.

No sealed confirmation, multilingual transfer or Laya/Jev unless DEV_READY.
