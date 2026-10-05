# HIRA V1 S65 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #317  
PR: #318

## Parent S64

Merged main:
`c5063da0365d49dc4f9dc2c64f94bc309caef0ba`

Fresh S64:
- run `37319541951`
- artifact `11348594503`
- digest `sha256:f3d04d452a51b7358a23a9391579bca6ce1ab2c16fa107dc9f7f79740dba90ea`
- verdict **Case B**.

## Qualified S65-A0

Run: `37322871071`  
Artifact: `11349983389`  
Digest: `sha256:1abecf58d75be991cd89dd3bb6732a859b1159426023a209758b1d5a7d459dea`

Outcome:
`HIRA_V1_S65_A0_CROSS_VIEW_SOFT_AND_RELIABILITY_READY`

Qualified:
- reference gate **60 params**
- treatment gate **60 params**
- added treatment params **0**
- bit-identical initialization
- bit-identical context projection
- same single-view inference path
- exact S62 target retained
- soft-AND exact-min error **1.49e-8**
- pair symmetry exact
- lower-view gradient routing exact
- objective added trainable params **0**
- staged phi gradient path PASS
- upstream gradients zero
- K=3/7/255 PASS
- bounded residual PASS
- probability mass error **1.19e-7**
- one encoder/state-once
- checkpoint replay exact.

## Fresh S65 authority

- seed **86001**
- TRAIN **768**
- DEV **192**
- **12 fresh S65 domains**
- exact S64 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Matched scientific trajectory

Shared exactly:
- immutable S51 native/cache evidence
- one correction trajectory, **114,688 params**
- one S59 pairwise-head trajectory, **32,832 params**
- exact S64 contextual single-view gate, **60 params** per arm
- same contextual projection and representation
- same parameter initialization
- exact S62 reliability target
- alpha probe **0.35**
- tolerance **1e-8**
- bounded residual alpha <= **0.35**
- same optimizer/LR/weight decay
- same TRAIN order
- same frozen S17 selector.

Reference:
- per-view independent BCE.

Treatment:
- cross-view exact-min soft-AND BCE during TRAIN only.

Inference:
- both arms use the same single-view contextual gate;
- no paired/cross-view inference dependency.

Per batch:
1. update shared correction;
2. update shared pairwise head;
3. recompute detached fused/pairwise surfaces;
4. reuse detached S59 contextual representation;
5. compute exact S62 pair-level target;
6. reference updates with independent-view BCE;
7. treatment updates with exact-min soft-AND BCE.

Reference/treatment target counts and labels must match every batch.

Fused shadow is diagnostic only and never participates in selection.

## Stop rule

Once S65 TRAIN begins:
- no soft-min temperature
- no min/mean/max sweep
- no objective mixing coefficient
- no BCE weighting
- no target change
- no context/projection/width/pooling change
- no optimizer/LR/weight-decay change
- no regularizer
- no gradient coupling
- no native retraining
- no selector change
- no retry for scientific weakness
- no second S65 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S65-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
