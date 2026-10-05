# HIRA V1 S59 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #302  
PR: #304

## Qualified S59-A0

Run: `37278474963`  
Artifact: `11331042601`  
Digest: `sha256:5271081c4bcacbced9ca85c146ddd36b5dc9ecdc0c6dc9038c9b3fb6219a11b6`  
Authorization head: `cff44fd7fead90f9e7d31f39b099760f4445a278`

Outcome:
`HIRA_V1_S59_A0_EXPLICIT_LEARNED_PAIRWISE_DECISION_HEAD_READY`

Qualified:
- pairwise params **32,832**
- trainable tensors exactly **A/u**
- no bias
- anti-symmetry max error **0**
- diagonal max error **0**
- K=3/7/255 PASS
- probability mass error <= 1e-6
- gold-vs-distractor supervision only
- distractor-vs-distractor supervised count **0**
- A/u gradients live
- representation requires-grad **false**
- correction received pairwise gradient **false**
- correction state unchanged after pairwise backward
- native trainable params **0**
- no teacher
- no native/fused/corrected logit input to head
- checkpoint replay max error **0**
- one encoder/state-once.

Observed A0:
- A gradient L1 **3.1314144134521484**
- u gradient L1 **0.06544577330350876**
- correct-sign loss **0.1629256755**
- flipped-sign loss **2.1629257202**.

## Fresh S59 authority

- seed **80001**
- TRAIN **768**
- DEV **192**
- **12 wholly fresh S59 domains**
- exact S58 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Shared scientific trajectory

One shared correction/head training trajectory:
- `JointStateQueryOptionPrivateCorrectionFork`
- correction params **114,688**
- explicit pairwise params **32,832**
- pair representation **512D detached**
- same immutable TRAIN/DEV cache
- same optimizer constants
- same base correction objective
- same pairwise gold objective
- same frozen S17 selector
- no teacher/pseudo-target/self-anchor.

At each epoch the court freezes:
- correction-state SHA
- pairwise-head-state SHA
- reference DEV metrics
- treatment DEV metrics.

Reference decision:
- existing fused decision shell.

Treatment decision:
- explicit learned anti-symmetric pairwise full-K aggregate.

Thus decision shell is the scientific variable; training trajectory is shared.

## Fresh court ownership

Parent S51:
- run `37192490832`
- artifact `11299783210`
- runtime digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`.

S59 A0:
- run `37278474963`
- artifact `11331042601`
- digest `sha256:5271081c4bcacbced9ca85c146ddd36b5dc9ecdc0c6dc9038c9b3fb6219a11b6`.

## Stop rule

Once S59 TRAIN begins:
- no width/activation sweep
- no loss variant
- no tiebreak variant
- no base-logit blend
- no pairwise/correction joint-gradient variant
- no architecture/capacity change
- no native retraining
- no selector change
- no retry for scientific weakness
- no gate weakening
- no second S59 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S59-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
