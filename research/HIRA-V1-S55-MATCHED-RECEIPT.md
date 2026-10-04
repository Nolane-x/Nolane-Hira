# HIRA V1 S55 matched scientific receipt — Learned Joint Relation Interaction

Status: **FROZEN / CASE C**

Scientific run: `37212962160`  
Artifact: `11307227880`  
Artifact digest: `sha256:7059cf8477eb4bd775e89badb0c734a2d351afe2e1a1d23ceaf4f7189c7744e4`  
Scientific head: `ad91370b20ee4127383c2f50801430a04aa72267`

Outcome:
`HIRA_V1_S55_LEARNED_JOINT_RELATION_INTERACTION_DEV_COMPLETE`

## Governance

Initial run `37212252407` was a mechanical pre-DEV abort:
- reference TRAIN_BEGIN reached;
- completed epochs 0;
- treatment TRAIN_BEGIN not reached;
- private DEV scoring 0;
- model selection 0.

A single mechanical replay was then authorized after repair + exact-head CI. The scientific evidence below comes only from run `37212962160`.

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- native trainable params **0**
- native optimizer absent
- native retraining **false**
- S55-A0 run `37211474503`
- S55-A0 artifact `11306837784`
- shared TRAIN cache digest `e8252d493fc45129c30caf2a207ada9fb89a2791beae686d87b8786ac4be3a9f`
- shared DEV cache digest `bb67546bc9f7c75ae3c280f3330a00671dddee4184cfc7558a7bd66d0c7d32c7`
- reference/treatment same cache bytes **true**
- private state-view encodes **0**

Matched private surface:
- correction **114,688 / arm**
- learned joint transform **65,536 / arm**
- total private trainable **180,224 / arm**
- identity params **0**
- initialization bit-identical **true**

## Selected checkpoints

Reference selected epoch: **22**  
Treatment selected epoch: **24**

## Reference — learned transform with zero explicit state channel

- fused canonical accuracy **0.3880208333333333**
- fused paraphrase accuracy **0.4270833333333333**
- paired both-correct **0.13020833333333334**
- fused agreement **0.4947916666666667**
- fused JS **0.044713844234744705**
- relation canonical accuracy **0.3515625**
- relation paraphrase accuracy **0.4453125**
- relation agreement **0.4479166666666667**
- relation JS **0.19614692963659763**

## Treatment — learned transform with S54 state-conditioned channel

- fused canonical accuracy **0.4322916666666667**
- fused paraphrase accuracy **0.3932291666666667**
- paired both-correct **0.17708333333333334**
- fused agreement **0.4661458333333333**
- fused JS **0.04184923398618897**
- relation canonical accuracy **0.4192708333333333**
- relation paraphrase accuracy **0.421875**
- relation agreement **0.3984375**
- relation JS **0.18485931415731707**

## Treatment minus reference

Correctness:
- fused canonical **4.43 pp**
- fused paraphrase **-3.39 pp**
- paired both-correct **4.69 pp**
- canonical relation accuracy **6.77 pp**
- paraphrase relation accuracy **-2.34 pp**

Stability:
- fused selected-choice agreement **-2.86 pp**
- fused JS **-0.002865**
- relation selected-choice agreement **-4.95 pp**
- relation JS **-0.011288**

Margins:
- fused canonical **+0.037912**
- fused paraphrase **-0.077496**
- relation canonical **+0.354751**
- relation paraphrase **-0.555471**

## Learned-transform diagnostics

Reference:
- relation-code cross-view cosine **0.49573742349942523**
- q↔relation-code cosine **0.24194980350633463**
- mean residual norm **1.2218775153160095**
- zero-state-channel max abs **0**

Treatment:
- relation-code cross-view cosine **0.4788971741994222**
- q↔relation-code cosine **0.3205200427522262**
- mean residual norm **1.1588203112284343**
- explicit joint-channel norm **1**
- joint-channel sensitivity **0.0496011795476079**

## Frozen interpretation

**Case C — correctness remains mixed/useful, but selected-choice stability does not materially improve.**

Treatment improves:
- fused canonical **+4.43 pp**
- paired both-correct **+4.69 pp**
- canonical relation accuracy **+6.77 pp**
- canonical relation margin strongly improves.

But:
- fused paraphrase **-3.39 pp**
- fused agreement **-2.86 pp**
- relation agreement **-4.95 pp**
- paraphrase relation accuracy **-2.34 pp**.

JS decreases in both fused and relation views, but this does not translate into selected-choice agreement. The learned relation code itself is also *less* cross-view aligned than reference.

Therefore adding matched learned joint capacity does not solve the remaining instability.

The S51–S55 correction/factorization family is now exhausted.

The next experiment must operate at the **decision-consistency level directly**, rather than continuing to redesign identity/query/context/relation representations.

No S55 retry or second DEV is authorized.
