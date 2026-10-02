# HIRA V1 S33 closure — Query-Explicit Relation Coordinates

Status: **CLOSED — DEV FAIL / PREREGISTERED CASE D / QUERY-EXPLICIT ADDITIVE COORDINATES REJECTED**

Issue: #249
PR: #250

## Canonical authority

A0:
- run `36988606786`
- artifact `11219055373`
- digest `sha256:99c4be1b4e2badc4aa5f6591d8c044377f62f8a958d7c51a0bee241a803596ea`
- authority head `5569fb670e72cfae42517875b95e0ea29522ed73`
- outcome `HIRA_V1_S33_A0_QUERY_EXPLICIT_RELATION_READY`

Fresh matched TRAIN/DEV:
- run `36989462713`
- artifact `11220650899`
- digest `sha256:398f173ea3945b600bc9a71449ccc5667ebad13ba5f7b6a3b0621d19f30c2d85`
- scientific head `1ec79633d8b6532c2cbfc65cab61f6b043dbf931`
- outcome `HIRA_V1_S33_MATCHED_QUERY_EXPLICIT_DEV_COMPLETE`

No second DEV.
No post-DEV score-weight tuning.
No query-pooling/signature-block retry.
No global-loss stacking.
No sealed confirmation.
No multilingual probe.
No Laya/Jev evaluation.

## Frozen matched setup

Both arms:
- exact S17 final-attention LoRA: **16,384**
- shared bias-free 256->128 projection: **32,768**
- exact physical trainable surface: **49,152**
- original A13 frozen
- HIRACore frozen
- S14 equal standardized full-K fusion
- S15 relation-logit detach
- S17 relation-priority norm-balanced gradients
- local relation CE + local signature canonicalization
- same fresh TRAIN **768**
- same fresh DEV **192**
- same 12 domains
- seed **54001**
- same epoch batch order
- 24 epochs
- batch 16
- AdamW 2e-4 / wd .01 / clip 1.0

Control:
- exact S13 CrossViewRelationCanonicalizer.

Treatment:
- exact S13 base logits/signatures retained;
- explicit projected pooled question anchor;
- query-option residual block;
- signature width **256 = 128 S13 + 128 query-option**;
- score `0.50 * base_S13 + 0.50 * query_option`;
- zero added learned parameters.

## A0 result

A0 established the treatment was mechanically real before DEV:

- inherited S13 base-logit max abs **0**
- inherited S13 base-signature max abs **0**
- query-anchor reference max abs **0**
- treatment signature width **256**
- added learned params **0**
- physical trainable surface **49,152**
- question-anchor intervention max abs **1.0**
- query-option signature intervention max abs **0.7071068287**
- query-option logits intervention max abs **10.0**
- qA option0-option1 controlled margin **+10.0**
- qB option1-option0 controlled margin **+10.0**
- option-permutation logit/signature error **0 / 0**
- probability-mass error **1.1920929e-7**
- full-K/checkpoint PASS

Real semantic gradients:
- Q LoRA-B L1 **0.0209561940**
- K **0.0252935886**
- V **0.2000730783**
- attention output **1.3588542938**
- projection L1 **43.4579086304**

Therefore DEV failure is not an inactive-treatment or harness result.

## Selected DEV — control

Selected epoch: **13**
Checkpoint:
`880bdf12a7abd5ca299d38de71a43e80a4d691e33d20e8322f76e43f23030de1`

Fused:
- canonical **0.5104166667**
- paraphrase **0.3567708333**
- paired **0.2135416667**
- question-swap **0.484375**
- cross-view agreement **0.4713541667**
- JS **0.0439519394**
- canonical margin **+0.0597386840**
- paraphrase margin **-0.2806819053**

Primary:
- canonical **0.4296875**
- paraphrase **0.3385416667**
- agreement **0.4921875**

Relation:
- canonical **0.421875**
- paraphrase **0.3151041667**
- canonical margin **-0.1305607222**
- paraphrase margin **-0.3917472288**
- agreement **0.6432291667**

Signatures:
- same-option cosine **0.8920793931**
- same-vs-strongest-wrong margin **0.1311184944**

Gate:
- PASS **13/22**
- FAIL **9/22**
- DEV_READY false

## Selected DEV — query-explicit treatment

Selected epoch: **6**
Checkpoint:
`dc77c926064aaa9d5ef8403ca74ff7253a5f1a15c3c3d8889701ab96c83457ad`

Fused:
- canonical **0.390625**
- paraphrase **0.2760416667**
- paired **0.1666666667**
- question-swap **0.5260416667**
- cross-view agreement **0.203125**
- JS **0.1523825129**
- canonical margin **-0.2864707964**
- paraphrase margin **-0.9238851070**

Primary:
- canonical **0.3463541667**
- paraphrase **0.2421875**
- agreement **0.1953125**

Relation:
- canonical **0.4401041667**
- paraphrase **0.3046875**
- canonical margin **-0.0911216897**
- paraphrase margin **-1.2646731014**
- agreement **0.2213541667**

Signatures:
- same-option cosine **0.6780493011**
- same-vs-strongest-wrong margin **0.0439207170**

Gate:
- PASS **12/22**
- FAIL **10/22**
- DEV_READY false

## Exact selected delta — query-explicit minus control

Small improvements:
- canonical relation accuracy **+0.0182291667**
- canonical relation margin **+0.0394390325**
- question-swap **+0.0416666667**

Major regressions:
- fused canonical **-0.1197916667**
- fused paraphrase **-0.0807291667**
- paired **-0.046875**
- fused canonical margin **-0.3462094804**
- fused paraphrase margin **-0.6432032017**
- fused agreement **-0.2682291667**
- fused JS **+0.1084305735** (worse)
- raw canonical **-0.0833333333**
- raw paraphrase **-0.0963541667**
- relation paraphrase **-0.0104166667**
- relation paraphrase margin **-0.8729258726**
- relation agreement **-0.421875**
- signature cosine **-0.2140300920**
- signature discrimination **-0.0871977773**

This is not a coherent gain.

## Best observed DEV — control

Across 24 epochs:
- fused canonical **0.5104166667**, epoch 13
- fused paraphrase **0.4010416667**, epoch 4
- paired **0.2135416667**, epoch 13
- question-swap **0.6302083333**, epoch 20
- agreement **0.625**, epoch 4
- minimum JS **0.0145279617**, epoch 1
- canonical margin **+0.0672321683**, epoch 12
- relation canonical **0.4505208333**, epoch 8
- relation paraphrase **0.4244791667**, epoch 5
- relation canonical margin best **-0.0620233143**, epoch 2
- relation paraphrase margin best **-0.0666614498**, epoch 5
- relation agreement **0.7239583333**, epoch 10
- signature cosine **0.9506630103**, epoch 7
- signature discrimination **0.1444530835**, epoch 6

## Best observed DEV — query-explicit

Across 24 epochs:
- fused canonical **0.4088541667**, epoch 18
- fused paraphrase **0.3307291667**, epoch 1
- paired **0.1666666667**, epoch 6
- question-swap **0.6510416667**, epoch 7
- agreement **0.6640625**, epoch 1
- minimum JS **0.0138145324**, epoch 1
- canonical margin best **-0.1113212352**, epoch 7
- relation canonical **0.4557291667**, epoch 7
- relation paraphrase **0.3411458333**, epoch 3
- relation canonical margin best **-0.0669018800**, epoch 7
- relation paraphrase margin best **-0.0601459518**, epoch 1
- relation agreement **0.7057291667**, epoch 1
- signature cosine best only **0.8141244451**, epoch 2
- signature discrimination **0.0837717950**, epoch 3

Even the best treatment epochs do not establish a coherent advantage.

## Optimization dynamics

Control epoch 1 -> 24:
- total TRAIN loss **1.7198981643 -> 0.8066370673**
- relation block **0.1795353821 -> 0.0981415422**
- canonicalization **0.2750124916 -> 0.0545827836**
- LoRA-B norm **0.4479116797 -> 2.6959481239**
- mean conflict **0.3072916667**

Query-explicit epoch 1 -> 24:
- total TRAIN loss **1.7085862036 -> 0.9064967930**
- relation block **0.1774262901 -> 0.0830388035**
- canonicalization **0.2622907255 -> 0.1537273082**
- LoRA-B norm **0.4968935847 -> 3.4098510742**
- mean conflict **0.2612847222**

The treatment is trainable, but its fresh representation generalization is worse.

## Preregistered interpretation

This matches **Case D — little/no coherent gain**.

The query-explicit coordinate:
- is mechanically active;
- slightly improves canonical relation accuracy/margin;
- but damages primary semantics, fused semantics, cross-view agreement, JS, relation paraphrase behavior and signature transport.

Therefore:

> Explicit query retention at the end of the S13 additive/anchor relation construction is not the missing solution.

The problem is deeper than a discarded final query vector.

## Closed directions

Do not create S33b by:
- changing 0.50/0.50 score weights
- changing query pooling
- reweighting signature blocks
- temperature tuning
- adding S31/S32 global loss
- seed/LR/epoch/batch retry
- selecting another exposed epoch
- weakening gates

Also do not return to:
- S1 two-slot evidence routing
- S2 query-token residual fusion into frozen W34
- S3 compact CP triadic factor scorer
- S10 pooled state-to-option grounding
- S11 fixed-window role/value binding
- S12 role-relative best-pair matching
- S31/S32 global contrastive variants

## Next structural residual

The common failure family from S10-S13 and S33 repeatedly reduces a state/question/option relation to:
- one or a few anchors;
- local windows;
- independently selected best token pairs;
- additive/residual pooled relation signatures.

That destroys multi-token correspondence structure before the final relation decision.

The next clean hypothesis should preserve a **many-to-many state↔option transport plan conditioned by the question**, and only aggregate after the relation matching itself is formed.

## S34 direction

**S34 — Query-Conditioned Entropic Relation Transport**

Use exact S17 optimization shell and 49,152 physical trainable surface.

Control:
- exact S13 relation operator.

Treatment:
- zero-parameter token-level transport relation operator;
- project state/question/option tokens through the same shared projection;
- compute question-conditioned state-token marginals from max question↔state cosine;
- compute question-conditioned option-token marginals from max question↔option cosine;
- compute state↔option token kernel from cosine similarity;
- use fixed entropic Sinkhorn normalization to obtain a many-to-many transport plan for each option view;
- score each option by expected state↔option similarity under its transport plan;
- derive relation signature only **after** transport from transport-weighted pair deltas;
- average only over valid semantic option views.

Frozen before A0:
- state relevance temperature **0.10**
- option relevance temperature **0.10**
- transport kernel temperature **0.10**
- Sinkhorn iterations **8**
- relation-logit temperature **0.10**
- zero added learned params/state
- no S13 additive/base-logit mixing
- no local-window assumption
- no global cross-case contrastive loss

A0 must prove:
- controlled query switch changes transport marginals/plan and selected option;
- option-token permutation equivariance;
- state-token permutation equivariance;
- valid-mask invariance;
- Sinkhorn marginal residual bounded;
- finite degenerate geometry;
- nonzero real LoRA/projection gradients;
- exact physical surface 49,152;
- full-K/state-once/checkpoint mechanics.

Fresh matched S34 data only.

Production-ready remains false.
Laya/Jev parity remains unestablished.
