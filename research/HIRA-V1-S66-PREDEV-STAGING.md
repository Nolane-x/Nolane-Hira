# HIRA V1 S66 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #319  
PR: #320

## Parent S65

Merged main:
`6bae27159fc2f9577f72540c7070da3e41ef3d15`

Fresh S65:
- run `37325995163`
- artifact `11352920320`
- digest `sha256:a44c560d304537bb7acaa1b5d569160ec744f897ae728edf95ceca7701be9dfc`
- verdict **Case B**.

## Qualified S66-A0

Run: `37329441437`  
Artifact: `11353806431`  
Digest: `sha256:61d76394e79457f26251468fea6e73be53e0507d979aa64d5b24c67f18562c39`

Outcome:
`HIRA_V1_S66_A0_PER_VIEW_RESPONSIBILITY_READY`

Qualified:
- reference gate **60 params**
- treatment gate **60 params**
- added treatment params **0**
- bit-identical initialization
- bit-identical context projection
- responsibility states **00/01/10/11**
- probe-bank disagreement **0.40234375**
- real-cache disagreement **0.25**
- view-swap target error **0**
- treatment target distinct from pair target
- correctness veto observed for both views
- stability veto observed for both views
- output gradients live
- staged phi gradients live
- upstream gradients zero
- K=3/7/255 PASS
- probability mass error **5.96e-8**
- one encoder/state-once
- checkpoint replay exact.

## Fresh S66 authority

- seed **87001**
- TRAIN **768**
- DEV **192**
- **12 fresh S66 domains**
- exact S65 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Matched scientific trajectory

Shared exactly:
- immutable S51 native/cache evidence
- one correction trajectory, **114,688 params**
- one S59 pairwise-head trajectory, **32,832 params**
- exact S64 contextual single-view gate, **60 params** per arm
- same context projection
- bit-identical parameter initialization
- fixed probe alpha **0.35**
- tolerance **1e-8**
- bounded residual alpha <= **0.35**
- same optimizer/LR/weight decay
- same TRAIN order
- same frozen S17 selector
- same single-view inference.

Reference supervision:
- exact S62 shared pair-level target
- independent BCE on canonical and paraphrase gate logits.

Treatment supervision:
- per-view counterfactual responsibility targets
- canonical target uses own CE safety + canonical-only JS improvement
- paraphrase target uses own CE safety + paraphrase-only JS improvement
- independent BCE against those two targets.

No extra treatment parameters.

Per batch:
1. update shared correction;
2. update shared pairwise head;
3. recompute detached fused/pairwise surfaces;
4. reuse detached S59 contextual representations;
5. compute reference shared pair target;
6. compute treatment per-view responsibility targets;
7. update matched gates independently.

Reference pair target and the treatment diagnostic pair target must remain identical.

## DEV comparison

Selection arms:
- reference = shared pair-target gate
- treatment = per-view responsibility gate.

Fused shadow is diagnostic only and not a selection arm.

Frozen S17 selector is used independently for both arms.

## Stop rule

Once S66 TRAIN begins:
- no responsibility tolerance sweep
- no CE/JS coefficient
- no target smoothing
- no target mixing
- no context/projection/width/pooling change
- no alpha-probe change
- no optimizer/LR/weight-decay change
- no regularizer
- no gradient coupling
- no native retraining
- no selector change
- no retry for scientific weakness
- no second S66 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S66-ENABLE-TRAIN-DEV`

It MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.

Scientific failure is valid.
