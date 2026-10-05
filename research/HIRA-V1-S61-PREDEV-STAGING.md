# HIRA V1 S61 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #307  
PR: #308

## Qualified S61-A0

Run: `37299903484`  
Artifact: `11341361300`  
Digest: `sha256:2313d461197d8a635842a86220b55cf249600f58e0d044ba6d016edf3d9ca4fb`  
Authorization head: `ce8c64daa70b744c3a040443cd01ff783f007a52`

Outcome:
`HIRA_V1_S61_A0_CONFIDENCE_ADAPTIVE_BOUNDED_HYBRID_READY`

Qualified:
- gate trainable params **5**
- trainable tensors exactly **w,b**
- feature dimension **4**
- features detached
- initial w exactly zero
- initial alpha exactly **0.10**
- alpha max **0.35**
- gate gradients reach w/b
- gate gradient upstream **0**
- adaptive probe produces distinct per-query alpha
- residual bound violation **0**
- option permutation PASS
- feature affine invariance PASS
- K=3/7/255 PASS
- probability mass PASS
- no pairwise-only final path
- no teacher / pseudo-target / self-anchor
- one encoder/state-once
- checkpoint replay exact.

Observed A0:
- adaptive probe alpha 0: **0.193052276968956**
- adaptive probe alpha 1: **0.11539456993341446**
- w gradient L1 **0.0236570984**
- b gradient L1 **0.0066764997**
- feature affine error **3.8743019104003906e-7**
- option permutation error **1.1920928955078125e-7**
- max probability-mass error **5.960464477539063e-8**.

## Fresh S61 authority

- seed **82001**
- TRAIN **768**
- DEV **192**
- **12 fresh S61 domains**
- exact S60 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Shared scientific trajectory

One shared trajectory contains:
- correction params **114,688**
- explicit pairwise params **32,832**
- adaptive gate params **5**
- immutable native/cache evidence
- detached pairwise representation
- same TRAIN order
- same frozen S17 selector.

At each TRAIN batch:
1. update correction only from frozen base private objective;
2. update pairwise head only from TRAIN gold pair objective;
3. recompute current fused/pairwise surfaces;
4. detach both decision surfaces;
5. update gate `w,b` only from TRAIN gold canonical+paraphrase CE.

Gate features per query:
1. normalized fused top1-top2 confidence;
2. normalized pairwise top1-top2 confidence;
3. fused/pairwise argmax agreement ±1;
4. centered surface alignment.

Treatment:
`fused + alpha(q) * fused_rms * tanh(standardized_pairwise)`

Reference:
exact existing fused decision.

Frozen:
- gate feature dimension **4**
- no hidden layer
- alpha max **0.35**
- initial alpha **0.10**
- gate AdamW LR = existing S35 LR
- gate weight decay **0.0**
- pairwise-only path forbidden.

At every epoch freeze/report:
- correction-state SHA
- pairwise-head-state SHA
- gate-state SHA
- gate mean/min/max/std alpha
- reference DEV metrics
- treatment DEV metrics.

## Pinned authority chain

Parent S60:
- merged main `c795c6c448af942f597a8d33131557338def7fa9`
- scientific run `37297784471`
- artifact `11340397535`
- digest `sha256:c55205e9b74aac6196346a0ce9b705900aaec7fe4c6b30380af8ee0093d1b18b`
- verdict **Case B**.

S51 native:
- run `37192490832`
- artifact `11299783210`
- runtime digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`.

S61 A0:
- run `37299903484`
- artifact `11341361300`
- digest `sha256:2313d461197d8a635842a86220b55cf249600f58e0d044ba6d016edf3d9ca4fb`.

## Stop rule

Once S61 TRAIN begins:
- no feature addition/removal
- no hidden layer
- no feature scaling sweep
- no alpha max sweep
- no initialization sweep
- no calibration objective variant
- no optimizer/LR/weight-decay change
- no gate regularizer
- no gradient coupling
- no architecture/capacity change
- no native retraining
- no selector change
- no retry for scientific weakness
- no gate weakening
- no second S61 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S61-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
