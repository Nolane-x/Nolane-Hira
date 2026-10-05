# HIRA V1 S58 matched scientific receipt — Consensus-Teacher Pairwise Ranking

Status: **FROZEN / CASE B**

Scientific run: `37276076841`  
Artifact: `11330721014`  
Artifact digest: `sha256:cea8203e98899feda6c1ca1558b9941a8900b987124dfb43958c90e5d0285e50`  
Scientific head: `ceb8c28ee68574e130f0a20761bc64e7413140ac`

Outcome:
`HIRA_V1_S58_CONSENSUS_TEACHER_PAIRWISE_RANKING_DEV_COMPLETE`

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- S58-A0 run `37274963437`
- S58-A0 artifact `11330280320`
- frozen S57 teacher run `37271509208`
- frozen teacher artifact `11327849211`
- frozen teacher branch **reference**
- frozen teacher selected epoch **19**
- teacher checkpoint SHA `804dc0b31ca75a77400c4a658ae893058d39916ed287d9fe2d45643798b739aa`
- teacher trainable params in S58 **0**
- teacher selected using S58 data **false**
- TRAIN **768**, DEV **192**, 12 fresh domains
- exact S57 state/question/option overlap **0**
- correction params **114,688 / arm**
- added trainable params **0**
- reference/treatment initialization bit-identical **true**
- teacher targets precomputed once before student training
- reference/treatment same teacher target bytes **true**.

Controlled variable:
- reference teacher-consensus coefficient **0.0**
- treatment teacher-consensus coefficient **0.05**
- teacher threshold **0.25**
- student target margin **0.05**
- wrong-gold teacher pairs filtered
- detached teacher consensus signs.

## Selected checkpoints

Reference selected epoch: **18**  
Treatment selected epoch: **21**

Reference checkpoint SHA:
`8cc14841d3511f8f40b74f6be15c3b40595c54341757d02346d2e92bdfcfacdd`

Treatment checkpoint SHA:
`6c9bdc54739c7c6ca3d840aa2357eac8d1eb72693d2a85aa842a1d04680c6b30`

## Reference

- fused canonical accuracy **0.5390625**
- fused paraphrase accuracy **0.4348958333333333**
- paired both-correct **0.23958333333333334**
- question-swap change **0.7552083333333334**
- fused agreement **0.4817708333333333**
- fused JS **0.07187499245628715**
- canonical relation accuracy **0.5546875**
- paraphrase relation accuracy **0.4583333333333333**
- relation agreement **0.5130208333333334**
- relation JS **0.08333782564538221**

## Treatment

- fused canonical accuracy **0.4765625**
- fused paraphrase accuracy **0.4401041666666667**
- paired both-correct **0.171875**
- question-swap change **0.6145833333333334**
- fused agreement **0.5182291666666666**
- fused JS **0.06182682925524811**
- canonical relation accuracy **0.4401041666666667**
- paraphrase relation accuracy **0.4427083333333333**
- relation agreement **0.5**
- relation JS **0.00813306881658112**

## Treatment minus reference

Stability:
- fused selected-choice agreement **+3.65 pp**
- fused JS **-0.010048**
- relation selected-choice agreement **-1.30 pp**
- relation JS **-0.075205**

Correctness/discrimination:
- fused canonical accuracy **-6.25 pp**
- fused paraphrase accuracy **+0.52 pp**
- paired both-correct **-6.77 pp**
- question-swap change **-14.06 pp**
- canonical relation accuracy **-11.46 pp**
- paraphrase relation accuracy **-1.56 pp**
- fused canonical margin **-0.164449**
- fused paraphrase margin **+0.007488**.

Teacher-target behavior at selected treatment:
- teacher active consensus fraction **0.486979**
- student violation fraction **0.112220**
- mean teacher-consensus pairwise loss **0.337317**.

## Frozen interpretation

**Case B — teacher-consensus pressure improves some fused/distribution stability, but correctness and discrimination still materially collapse.**

S58 successfully removes the S57 self-anchor mechanism:
- frozen teacher replay is exact;
- teacher has no gradients;
- wrong-gold teacher pairs are filtered;
- treatment follows teacher targets more closely.

However that is not enough. The treatment gains only **+3.65 pp** fused agreement while losing **6.25 pp** canonical accuracy, **6.77 pp** paired both-correct, **14.06 pp** question-swap discrimination, and **11.46 pp** canonical relation accuracy.

The key localization is now stronger:
**the problem is not merely unsafe self-generated targets. A fixed teacher target itself is too restrictive and transfers a decision boundary that does not preserve the student's correctness/discrimination frontier.**

Per the preregistered S58 interpretation, the next family is an **explicit learned pairwise decision head** rather than another teacher/threshold/consensus-mask variant.

No S58 teacher swap, threshold/margin/coefficient sweep, retry, selector change, second DEV, or external Laya/Jev evaluation is authorized.
