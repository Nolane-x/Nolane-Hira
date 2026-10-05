# HIRA V1 S60 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #305  
PR: #306

## Qualified S60-A0

Run: `37291693731`  
Artifact: `11337255933`  
Digest: `sha256:0a8f44e150932c0df4393cb74af94a380f93680e7b3fde2e0ea642295356c182`  
Authorization head: `24399c0766bd5d72da0ac00ef368a0b05c469b37`

Outcome:
`HIRA_V1_S60_A0_TRAIN_CALIBRATED_BOUNDED_HYBRID_READY`

Qualified:
- correction params **114,688**
- pairwise params **32,832**
- composer params exactly **1**
- trainable tensor exactly **a**
- alpha initial **0.10**
- alpha max **0.35**
- alpha_override=0 exact fused identity
- flat pairwise residual exactly zero
- adversarial residual bound PASS
- option permutation PASS
- fused offset/scale contract PASS
- pairwise offset/scale invariance PASS
- K=3/7/255 PASS
- full-K probability mass PASS
- composer gradient live
- composer gradient upstream **0**
- no pairwise-only final path
- no teacher / pseudo-target / self-anchor
- one encoder/state-once
- checkpoint replay exact.

Observed A0:
- composer gradient L1 **0.0086300261**
- real-cache alpha **0.1000000015**
- real-cache canonical residual max **0.0904251486**
- real-cache canonical bound max **0.0999275297**
- max probability-mass error **1.7881393432617188e-7**.

## Fresh S60 authority

- seed **81001**
- TRAIN **768**
- DEV **192**
- **12 fresh S60 domains**
- exact S59 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Shared scientific trajectory

One shared training trajectory contains:
- correction `114,688` params
- explicit pairwise head `32,832` params
- bounded hybrid composer `1` param
- immutable native/cache evidence
- detached pairwise representation
- same TRAIN order
- same frozen S17 selector.

Training each batch:
1. update correction only from frozen base private objective;
2. update pairwise head only from TRAIN gold pair objective;
3. recompute current fused/pairwise surfaces;
4. update composer scalar only from TRAIN gold canonical+paraphrase CE.

Composer inputs are detached; composer gradient cannot enter correction/head/native/cache.

Reference decision:
- exact existing fused logits.

Treatment decision:
- `fused + alpha * fused_rms * tanh(standardized_pairwise)`.

Frozen:
- `alpha_max = 0.35`
- initial `alpha = 0.10`
- composer AdamW LR = existing S35 LR
- composer weight decay **0.0**
- no pairwise-only path.

At every epoch freeze/report:
- correction state SHA
- pairwise-head state SHA
- composer state SHA
- alpha
- reference DEV metrics
- treatment DEV metrics.

## Pinned authority chain

Parent S59:
- merged main `867056d999bab707a1d8f4b1ea9ca774ccaf514d`
- run `37289522292`
- artifact `11335428630`
- verdict **Case C**.

S51 native:
- run `37192490832`
- artifact `11299783210`
- runtime digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`.

S60 A0:
- run `37291693731`
- artifact `11337255933`
- digest `sha256:0a8f44e150932c0df4393cb74af94a380f93680e7b3fde2e0ea642295356c182`.

## Stop rule

Once S60 TRAIN begins:
- no alpha_max sweep
- no alpha initialization sweep
- no calibration objective change
- no optimizer/LR/weight-decay change
- no residual nonlinearity change
- no normalization change
- no per-query gate
- no gradient coupling
- no architecture/capacity change
- no native retraining
- no selector change
- no retry for scientific weakness
- no gate weakening
- no second S60 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S60-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.


## Pre-DEV reconciliation

Initial exact-head CI `37296806343` on `72b6dd04929b9112353df8da7ea56fa1dcf5068d` correctly rejected the staging because one DEV state-view literal inherited the S59 wording, creating exact S59 state overlap.

The authority was repaired at implementation head:
`a64f2236563cb41921c94bb7a759b2035ef35070`

Repair scope:
- only the leaked S59 DEV state-view literal was renamed into the S60 namespace;
- no TRAIN/DEV labels, examples, counts, optimizer, architecture, alpha, loss, selector, or evaluation rule changed;
- fresh S60 DEV remained unscored and unexposed.

The TRAIN/DEV marker remains absent. A new exact-head generic CI is required before authorization.
