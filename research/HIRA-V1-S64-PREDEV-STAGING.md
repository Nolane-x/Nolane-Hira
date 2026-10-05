# HIRA V1 S64 pre-DEV staging receipt

Status: **STAGED / TRAIN-DEV NOT AUTHORIZED**

Issue: #315  
PR: #316

## Parent S63

Merged main:
`4fcb58f0c35e2feca841b08f6f560c092ae600fd`

Fresh S63:
- run `37314015283`
- artifact `11347706059`
- digest `sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e`
- verdict **Case B**.

## Qualified S64-A0

Run: `37318082102`  
Artifact: `11349331302`  
Digest: `sha256:462c4b845ba60b870a5ead4a26dbc9e4ba661426c6d2a23112484214d7c6d14e`

Outcome:
`HIRA_V1_S64_A0_CONTEXTUAL_RELIABILITY_GATE_READY`

Qualified:
- reference gate **60 params**
- treatment gate **60 params**
- added treatment params **0**
- parameter init bit-identical
- context projection **[4,512]**
- projection seed **64064**
- projection trainable params **0**
- encoder init seed **64164**
- surface dim **4**
- context dim **4**
- option input dim **8**
- hidden width **5**
- pooled dim **10**
- final representation **14**
- reference contextual channels exact zero
- treatment contextual channels non-degenerate
- contextual sensitivity PASS
- permutation invariance PASS
- surface affine invariance PASS
- staged gradient path PASS
- upstream gradients zero
- K=3/7/255 PASS
- exact S62 target retained
- bounded residual PASS
- one encoder/state-once
- replay exact.

Observed A0:
- treatment context L1 **70.80506134**
- contextual sensitivity max abs **0.003652066**
- permutation alpha error **2.98e-8**
- surface affine alpha error **7.45e-9**
- full-K probability mass error **1.19e-7**.

## Fresh S64 authority

- seed **85001**
- TRAIN **768**
- DEV **192**
- **12 fresh S64 domains**
- exact S63 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Matched scientific trajectory

Shared exactly:
- immutable S51 native/cache evidence
- one correction trajectory, **114,688 params**
- one S59 pairwise-head trajectory, **32,832 params**
- exact same 60-param reliability architecture
- exact same parameter initialization
- exact same fixed context projection buffer
- exact S62 reliability target
- alpha probe **0.35**
- target tolerance **1e-8**
- bounded residual, alpha <= **0.35**
- same optimizer/LR/weight decay
- same TRAIN order
- same frozen S17 selector.

Reference:
- surface4 + exact zero context4
- 60 params
- TRAIN-only reliability BCE.

Treatment:
- surface4 + fixed projected S59 context4
- 60 params
- exact same TRAIN-only reliability BCE.

Per batch:
1. update shared correction from existing private objective;
2. update shared pairwise head from gold-pair objective;
3. recompute and detach current fused/pairwise surfaces;
4. reuse detached S59 context representation;
5. compute exact S62 reliability target;
6. update reference gate;
7. update treatment gate.

Reference/treatment reliability target count and positive count must match every batch.

Fused shadow is diagnostic only and never participates in selection.

## Stop rule

Once S64 TRAIN begins:
- no context projection seed/dim sweep
- no hidden-width sweep
- no surface/context feature retrofit
- no pooling change
- no initialization sweep
- no alpha-probe/tolerance/target change
- no BCE weighting
- no optimizer/LR/weight-decay change
- no regularizer
- no gradient coupling
- no native retraining
- no selector change
- no retry for scientific weakness
- no second S64 DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S64-ENABLE-TRAIN-DEV`

The marker MUST remain absent until the exact final pre-DEV staging head passes generic CI on Python 3.10 and 3.12.
