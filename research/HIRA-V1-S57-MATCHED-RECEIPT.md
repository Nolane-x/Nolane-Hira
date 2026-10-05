# HIRA V1 S57 matched scientific receipt — Discrete Pairwise Ranking Consistency

Status: **FROZEN / CASE B**

Scientific run: `37271509208`  
Artifact: `11327849211`  
Artifact digest: `sha256:868faf45b1994286e52cbac4c35adfbbb420e2ba8a6c561545067da013c05147`  
Scientific head: `02d1b0155cdd56dbae90832dcb7743618881f668`

Outcome:
`HIRA_V1_S57_DISCRETE_PAIRWISE_RANKING_CONSISTENCY_DEV_COMPLETE`

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- native trainable params **0**
- native retraining **false**
- S57-A0 run `37266891469`
- S57-A0 artifact `11326757222`
- shared TRAIN cache digest `b1271c1bcda199dd7300996b4d59ff0316d5e9efe4224d16c8b846919e405c7d`
- shared DEV cache digest `1f9e1cb5c2a7d202d6349570384437440f0c38bb7c2fd745cad2b84d15f4f9c5`
- reference/treatment same cache bytes **true**
- private state-view encodes **0**

Matched private surface:
- correction params **114,688 / arm**
- added trainable params **0**
- initialization bit-identical **true**

Controlled variable:
- reference ordinal coefficient **0**
- treatment ordinal coefficient **0.05**
- standardized-logit epsilon **1e-6**
- active threshold **0.25**
- preserved margin floor **0.05**
- detached anchors + gold-order protection.

## Selected checkpoints

Reference selected epoch: **19**  
Treatment selected epoch: **22**

## Reference

- fused canonical accuracy **0.4557291666666667**
- fused paraphrase accuracy **0.4713541666666667**
- paired both-correct **0.17708333333333334**
- fused agreement **0.4869791666666667**
- fused JS **0.035293725784868**
- canonical relation accuracy **0.421875**
- paraphrase relation accuracy **0.4453125**
- relation agreement **0.5260416666666666**
- relation JS **0.029611536379282672**

## Treatment

- fused canonical accuracy **0.359375**
- fused paraphrase accuracy **0.4192708333333333**
- paired both-correct **0.125**
- fused agreement **0.5677083333333334**
- fused JS **0.034106852719560266**
- canonical relation accuracy **0.3515625**
- paraphrase relation accuracy **0.4036458333333333**
- relation agreement **0.4505208333333333**
- relation JS **0.009777729418904832**

## Treatment minus reference

Stability:
- fused selected-choice agreement **8.07 pp**
- fused JS **-0.001187**
- relation selected-choice agreement **-7.55 pp**
- relation JS **-0.019834**

Correctness/discrimination:
- fused canonical accuracy **-9.64 pp**
- fused paraphrase accuracy **-5.21 pp**
- paired both-correct **-5.21 pp**
- relation canonical accuracy **-7.03 pp**
- relation paraphrase accuracy **-4.17 pp**
- question-swap change **-17.71 pp**
- fused canonical margin **-0.157942**
- fused paraphrase margin **-0.171732**

## Frozen interpretation

**Case B — fused stability improves materially, but correctness/discrimination still collapses and relation stability does not generalize.**

Treatment improves:
- fused selected-choice agreement **+8.07 pp**
- fused JS **-0.001187**

But:
- fused canonical accuracy **-9.64 pp**
- fused paraphrase accuracy **-5.21 pp**
- paired both-correct **-5.21 pp**
- question-swap discrimination **-17.71 pp**
- relation canonical accuracy **-7.03 pp**
- relation paraphrase accuracy **-4.17 pp**
- relation selected-choice agreement **-7.55 pp**.

So selective ordinal pressure avoids the extreme distribution collapse of S56, but the active anchors are still not safe enough. It can stabilize the final fused choice while pushing incorrect or view-specific pairwise orderings into the private relation path.

The next family must keep decision-level pairwise consistency but replace self-anchoring with **consensus/teacher anchoring** so only cross-view-supported orderings become targets.

No S57 coefficient/threshold/filter sweep or second DEV is authorized.
