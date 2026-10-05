# HIRA V1 S63 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #311  
PR: #312

## Parent S62

Merged main:
`798f5c4d933635b36f4e0eaf76503f8213afedd9`

Fresh S62:
- run `37307658289`
- artifact `11344855774`
- digest `sha256:7812c348d707ea8675fc9370af573a87d9e5b1a0b10e55a1f335bf890ddb3305`
- verdict **Case B**.

## Qualified S63-A0

Run: `37312740736`  
Artifact: `11346084572`  
Digest: `sha256:39c9c84a93e5da0075b13018b7a5a4151dda4563476834478524e042b1cfac9f`

Outcome:
`HIRA_V1_S63_A0_LEARNED_SET_RELIABILITY_GATE_READY`

Qualified:
- reference gate **5 params**
- treatment gate **61 params**
- added treatment params **56**
- treatment tensors exactly `W_phi,b_phi,w_out,b_out`
- shapes **[8,4], [8], [20], scalar**
- option input dim **4**
- hidden width **8**
- pooled dim **16**
- final representation dim **20**
- init seed **63063**
- output weights zero
- initial alpha exact **0.10**
- alpha max **0.35**
- permutation invariance PASS
- surface affine invariance PASS
- phi gradient path proven after isolated warm step
- upstream gradients zero
- K=3/7/255 PASS
- bounded residual PASS
- exact S62 reliability target retained
- one encoder/state-once
- checkpoint replay exact.

Observed A0:
- option permutation alpha max error **1.49e-8**
- affine alpha max error **1.49e-8**
- post-warm W_phi gradient L1 **0.0220926**
- post-warm b_phi gradient L1 **0.00602429**
- full-K probability mass error **1.19e-7**.

## Fresh S63 authority

- seed **84001**
- TRAIN **768**
- DEV **192**
- **12 fresh S63 domains**
- exact S62 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Matched scientific trajectory

Shared exactly:
- immutable S51 native/cache evidence
- one correction trajectory, **114,688 params**
- one S59 pairwise-head trajectory, **32,832 params**
- exact S62 reliability target
- alpha probe **0.35**
- target tolerance **1e-8**
- bounded residual, alpha <= **0.35**
- same optimizer/LR/weight decay
- same TRAIN ordering
- same frozen S17 selector.

Reference:
- exact S62 four-scalar reliability gate
- **5 params**
- TRAIN-only reliability BCE.

Treatment:
- learned permutation-invariant set reliability gate
- **61 params**
- exact same TRAIN-only reliability BCE target.

Per batch:
1. update shared correction from existing private objective;
2. update shared pairwise head from gold-pair objective;
3. recompute and detach current fused/pairwise surfaces;
4. compute exact S62 reliability target;
5. update reference 5-param gate;
6. update treatment 61-param gate.

Reference/treatment target count and positive count must match every batch.

Fused shadow is diagnostic only and never participates in model selection.

## Stop rule

Once S63 TRAIN begins:
- no option-feature addition/removal
- no hidden-width sweep
- no mean/max pooling change
- no initialization sweep
- no alpha-probe/tolerance/target change
- no BCE weighting
- no optimizer/LR/weight-decay change
- no regularizer
- no gradient coupling
- no native retraining
- no selector change
- no retry for scientific weakness
- no second S63 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S63-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
