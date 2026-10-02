# HIRA V1 S35 contract — Native A13 Relation Geometry Court

Status: **OPEN / PREREGISTERED BEFORE S35-A0 EXPOSURE**

Issue: #253

Parent:
- S34 issue #251 / PR #252
- S34 outcome `HIRA_V1_S34_MATCHED_TRANSPORT_DEV_COMPLETE`
- merged main `d6df51427d128a688e70205dbaea36526ac9c5be`

## 1. Scientific question

S34 showed that a mechanically valid many-to-many transport operator still fails badly inside the inherited shared 128D W28-style relation geometry.

S5 relearned the shared 256->128 projection and failed.
S7 jointly adapted A13 + shared projection and improved semantics without reaching robust fresh generalization.
S13-S34 relation operators all consume the compact shared projection geometry.

S35 isolates one remaining representation question:

> Is relation information being lost by the shared 128D projection even when it remains available in adapted A13 native 256D token embeddings?

## 2. Frozen shared shell

Both arms use exact S17:
- final A13 attention LoRA **16,384**
- shared bias-free 256->128 primary projection **32,768**
- exact physical trainable surface **49,152**
- original A13 frozen
- HIRACore frozen
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 primary/relation partition
- S17 relation-priority norm-balanced gradient rule
- local relation CE coefficient **0.10**
- local relation signature canonicalization coefficient **0.15**
- signature separation margin **0.20**
- state-once / full-K / opaque option IDs
- no learned downstream head/router/gate/calibrator

The primary scorer is identical in both arms and always uses the shared 128D projection.

## 3. Control

Exact S13 `CrossViewRelationCanonicalizer`:
- shared projected 128D state/question/option token geometry
- role temperature **0.10**
- pair temperature **0.10**
- contrastive temperature **0.10**
- relation signature width **128**

## 4. Treatment

**NativeA13RelationCanonicalizer**

Zero learned parameters/state.

Use the exact S13 relation algorithm directly in adapted A13 **256D native token geometry**:
- L2-normalize state/question/option native content-token embeddings;
- compute question-conditioned state role weights identically to S13;
- compute question-conditioned option role weights identically to S13;
- compute state and option role anchors identically;
- compute direct and relative token-pair cosine terms identically;
- compute pair softmax identically;
- compute role delta and relation signature identically;
- average active semantic option views identically;
- compute relation logits identically.

The only scientific change is:
- S13 control first applies the shared 256->128 relation projection;
- S35 treatment does **not** use the projection in the relation path.

Frozen treatment:
- native dimension **256**
- role temperature **0.10**
- pair temperature **0.10**
- contrastive temperature **0.10**
- no learned native metric
- no whitening
- no projection/native mixing
- no extra normalization beyond S13-equivalent L2 normalization
- no global contrastive objective
- no transport operator
- zero added learned params/state

## 5. Gradient ownership

Control:
- relation block may train both LoRA and shared projection.

Treatment:
- relation block must have **zero direct gradient** to the shared projection;
- relation block must have nonzero finite gradient to adapted A13 LoRA;
- primary block must retain nonzero finite gradient to the shared projection;
- primary block remains identical between arms.

No projection freeze is introduced: the shared projection remains trainable through the primary S17 path.

## 6. Required S35-A0

Use fresh S35-A0 rows only.

A0 must prove:
- treatment operator params **0**
- exact physical trainable surface **49,152**
- original A13 trainable params **0**
- HIRACore trainable params **0**
- treatment signature width **256**
- control signature width **128**
- primary logits exact control/treatment identity
- native operator equals S13 algorithm under an exact 256D identity projection on controlled geometry
- strong shared-projection perturbation materially changes control relation logits/signatures
- the same perturbation changes native treatment relation logits/signatures by **exactly 0**
- logical-option permutation equivariance
- state-token permutation equivariance
- question-token permutation equivariance
- option-token permutation equivariance
- masked-padding invariance
- finite one-token / duplicate-token geometry
- treatment relation block -> shared projection direct gradient **0**
- treatment relation block -> LoRA gradient nonzero
- primary block -> projection gradient nonzero
- local relation CE/signature losses finite
- full-K/state-once/probability/checkpoint mechanics PASS

A0 semantic accuracy is diagnostic only.

## 7. Fresh matched TRAIN/DEV

Only after qualified A0 + frozen receipt + exact-head CI + separate one-shot marker.

Frozen intended authority:
- seed **56001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S35 domains
- K=4
- two state views
- two question wording views per semantic query
- two option semantic views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**
- identical rows and per-epoch batch order across arms
- independent runtime/optimizer state

Freshness:
- exact S0-S34 exposed rows excluded
- S35-A0 rows excluded
- M5 final/confirmatory rows excluded
- W29-W34 sealed rows excluded

## 8. Selector and gates

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
- full-K/state-once/capacity/freeze mechanics

## 9. Frozen interpretation

### A — coherent native-space gain
Native treatment improves relation accuracy/margins and fused endpoints without sacrificing transport.
Then native relation geometry remains viable for a fresh confirmation track.

### B — native transport improves but semantic decision quality regresses
Native space preserves wording identity but does not identify the correct relation.
Close S35.

### C — semantics improve but signature transport weakens
Native geometry contains useful relation information but wording invariance remains the bottleneck.
Close S35 and localize native-space invariance; do not mix projected/native geometry on exposed DEV.

### D — little/no coherent gain
Reject projection-bypass as the solution and move away from relation geometry localization within the current A13 representation.

### E — treatment DEV_READY
Freeze immediately.
No second S35 DEV.
A separate confirmation protocol is required before external matched evaluation.

## 10. Stop rule

After one fresh S35 DEV:
- no projected/native logit mixing
- no dimensional-reduction retry
- no native whitening
- no learned native metric
- no temperature/normalization retry
- no seed/LR/epoch/batch retry
- no selector/gate weakening
- no second DEV

Scientific FAIL is valid.

No Laya/Jev benchmark unless a later confirmed candidate reaches DEV_READY.
