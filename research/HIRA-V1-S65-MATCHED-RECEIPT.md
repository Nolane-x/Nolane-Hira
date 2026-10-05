# HIRA V1 S65 matched scientific receipt — Cross-View Soft-AND Reliability Objective

Status: **FROZEN / CASE B**

Scientific run: `37325995163`  
Artifact: `11352920320`  
Artifact digest: `sha256:a44c560d304537bb7acaa1b5d569160ec744f897ae728edf95ceca7701be9dfc`  
Scientific head: `1dc1ff01ef85d76f6937c3f12fbe4f2c2bab0bfe`

Outcome:
`HIRA_V1_S65_CROSS_VIEW_SOFT_AND_RELIABILITY_DEV_COMPLETE`

## Controlled variable

Reference:
- exact S64 contextual single-view gate
- **60 trainable params**
- independent per-view reliability BCE.

Treatment:
- exact same S64 contextual single-view gate
- **60 trainable params**
- cross-view exact-min soft-AND BCE during TRAIN only.

Treatment parameter advantage: **0**.

Shared exactly:
- correction trajectory **114,688 params**
- pairwise-head trajectory **32,832 params**
- context representation/projection
- bit-identical gate initialization
- exact S62 pair-level reliability target
- alpha probe **0.35**
- target tolerance **1e-8**
- bounded residual alpha <= **0.35**
- optimizer/LR/weight decay
- frozen S17 selector
- immutable native/cache evidence
- single-view inference path.

## Selected checkpoints

Reference epoch: **18**  
Treatment epoch: **18**

## Reference

- canonical accuracy **0.4791666667**
- paraphrase accuracy **0.4348958333**
- paired both-correct **0.2604166667**
- question-swap **0.5989583333**
- agreement **0.3463541667**
- JS **0.1243128323**
- canonical gold margin **-0.0881838499**
- paraphrase gold margin **-0.4007128078**
- mean alpha **0.0741744523**
- alpha range **0.0670361444–0.0853542313**
- alpha std **0.0034994693**
- pairwise gold-pair accuracy **0.5993923611**

## Treatment

- canonical accuracy **0.4791666667**
- paraphrase accuracy **0.4348958333**
- paired both-correct **0.2604166667**
- question-swap **0.5989583333**
- agreement **0.3463541667**
- JS **0.1243827490**
- canonical gold margin **-0.0880918371**
- paraphrase gold margin **-0.4006766300**
- mean alpha **0.0751679099**
- alpha range **0.0675445870–0.0880381688**
- alpha std **0.0042641923**
- pairwise gold-pair accuracy **0.5993923611**

## Treatment minus reference

- canonical accuracy **0.00 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap **0.00 pp**
- selected-choice agreement **0.00 pp**
- cross-view JS **+0.0000699167 worse**
- canonical gold margin **+0.0000920128**
- paraphrase gold margin **+0.0000361778**
- relation-binding metrics **unchanged**.

## Frozen interpretation

**Case B — cross-view exact-min soft-AND changes the gate policy slightly but does not recover the missing stability signal.**

Evidence:
- treatment mean alpha differs from reference (**0.07517 vs 0.07417**);
- treatment alpha spread is larger;
- canonical/paraphrase correctness, both-correct, question-swap and agreement are exactly unchanged;
- JS is microscopically worse by about **6.99e-5**.

Therefore pair-level reliability target routing through a hard soft-AND is insufficient. The next family must assign **per-view responsibility** using TRAIN-only counterfactual evidence, rather than forcing both views to share a single pair label or relying on a min aggregation.

No S65 temperature sweep, min/mean/max sweep, objective mixture, target change, context/projection/width/pooling change, retry, second DEV, selector change, native retraining or external Laya/Jev evaluation is authorized.
