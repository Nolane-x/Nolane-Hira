# HIRA V1 S45 contract — Cross-View Consistent Private Correction

Status: **PREREGISTERED / NO S45-A0 EXPOSURE**

Issue: #273

Parent scientific evidence:
- S44 fresh run `37123003224`
- artifact `11274687323`
- interpretation **Case A**
- native trajectory exact
- relation canonical/paraphrase gains **+0.1770833 / +0.1223958**
- fused canonical/paraphrase gains **+0.0546875 / +0.0078125**
- relation agreement delta **-0.1458333**
- fused JS delta **+0.0310173** worse
- treatment DEV_READY false

## Scientific question

> Can Hira retain S44's detached private-correction correctness while making that private expert cross-view consistent, without changing capacity, native ownership, fusion weighting, or encoder count?

## Architecture

S45 reuses the exact S44 private correction architecture unchanged.

Native path:
- exact S35/S17 native runtime
- LoRA 16,384
- primary projection 32,768
- native trainable total **49,152**
- reference and treatment must follow identical native trajectories
- correction gradients may never enter native LoRA/projection/signatures/query tokens

Private correction path:
- detached native signature [256]
- detached normalized query [256]
- concat [512]
- A 64x512 = **32,768**
- GELU
- B 256x64 = **16,384**
- normalized private signature
- W 256x256 = **65,536**
- correction-only total **114,688**
- treatment total **163,840**
- no new trainable parameter
- no bias, gate, layer norm, learned scale or learned fusion weight
- one encoder batch only

## Frozen correction objective

S44:
`CE_corr = 0.5 * (CE(c_c, gold) + CE(c_p, gold))`

S45 adds exactly one existing consistency primitive:

`JS_corr = JS(softmax(c_c), softmax(c_p))`

`L_corr = 0.10 * CE_corr + 0.25 * JS_corr`

where:
- `c_c` and `c_p` are corrected relation logits for exact semantic canonical/paraphrase pairs;
- JS is symmetric Jensen-Shannon divergence using natural log;
- no temperature;
- epsilon uses the existing numerical clamp convention only;
- **0.10** reuses frozen S17/S39/S44 binding coefficient;
- **0.25** reuses the existing Hira cross-view JS coefficient;
- no scalar was chosen from S44 DEV.

No fused-specific correction loss.
No primary correction loss.
No learned fusion change.

## Optimizers

Native reference/treatment:
- exact S44/S35/S17 AdamW semantics and matched initialization/order.

Private correction:
- AdamW lr **2e-4**
- weight decay **0.01**
- existing betas/eps
- grad clip **1.0** over A/B/W
- no separate schedule.

## Required S45-A0

Fresh diagnostic authority must prove:
- exact S44 parameter surface;
- native reference/treatment initial and one-step update identity;
- CE correction gradients reach W and deterministic warm-start makes B then A live;
- JS correction gradients reach A/B/W under a deterministic non-identical paired probe;
- JS correction gradients to native LoRA/projection/signatures/query are zero;
- native objective gradients to A/B/W are zero;
- identical corrected pair gives JS zero within frozen tolerance;
- deterministic different pair gives finite positive JS;
- semantic pair indexing cannot cross cases;
- option permutation equivariance;
- question/padding invariance;
- arbitrary K=3/K=7;
- checkpoint roundtrip;
- full-K/probability mass;
- one encoder batch / state-once;
- no extra parameters;
- no learned fusion gate;
- no projected relation path.

A0 semantic values are diagnostic only.

## Intended fresh matched authority

Only after contract + interpretation plan + qualified A0 + receipt + frozen authority + trainer/workflow + exact staged-head CI + separate one-shot marker.

Intended:
- seed **66001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S45 domains
- K=4
- 24 epochs
- batch 16
- identical native rows/order between reference and treatment
- one DEV only

Freshness:
- no exact S0-S44 exposed rows
- no S45-A0 rows
- no M5 final/confirm rows
- no W29-W34 sealed rows.

## Stop rule

After one S45 DEV:
- no JS coefficient sweep
- no temperature sweep
- no alternate divergence
- no width/activation/capacity change
- no learned fusion gate/weight
- no native-gradient leakage
- no second encoder
- no seed/LR/epoch/batch retry
- no gate weakening
- no second DEV.

No external Laya/Jev benchmark before separately confirmed DEV_READY.
