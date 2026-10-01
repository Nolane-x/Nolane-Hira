# HIRA V1 S29 contract — Final-Block Attention+FFN LoRA Semantic Adaptation

Status: **OPEN / PREREGISTERED BEFORE S29-A0 EXPOSURE**

Issue: #241

Parent:
- S28 issue #239 / PR #240
- S28 outcome `HIRA_V1_S28_ANCHOR_FACTOR_TRANSPORT_DEV_FAIL`

## 1. Motivation

S17 remains the strongest clean v1 fresh semantic frontier:
- fused canonical **0.7213541667**
- paired both-correct **0.5104166667**
- question-swap **0.984375**
- fused canonical margin **+0.3138313380**
- relation canonical **0.640625**
- relation margin **+0.2073315941**

S18-S28 test downstream consistency, primary factorization, optimizer priority, fusion, projection ownership, factorized relation representations, output canonicalization and latent anchor transport. None reaches DEV_READY.

S28 specifically proves that cross-view latent transport can become strong while same-vs-wrong semantic discrimination remains weak.

Historical review forbids relabeling old ideas as S29:
- S11/S12/S21 already test query-role / role-value localization and binding;
- W7 already tests explicit smooth-AND/conjunctive factor scoring and fails against an equal-data free-form control;
- S0-S3/R8 already contain generic query-gating/low-rank bridge failures.

One v1 semantic encoder surface remains untested:
**the feed-forward sublayer of A13's final BERT block**.

Every S6-S28 A13 adaptation wraps only the four final attention linears.

## 2. Controlled hypothesis

> S17's remaining generalization gap requires limited nonlinear token-wise semantic remapping in the final A13 FFN. Expanding LoRA coverage from final attention only to the complete final Transformer block can improve semantic discrimination without unfreezing original A13 weights or adding downstream task heads.

S29 is an **encoder adaptation surface** experiment.

It is not:
- a new loss family;
- a new fusion rule;
- a new relation scorer;
- an optimizer-priority retry;
- a rank/alpha search.

## 3. Frozen S17 inference/training shell

Keep exactly:
- S13 cross-view relation expert/canonicalizer;
- S14 equal standardized full-K fusion, epsilon **1e-6**;
- S15 fused-primary relation-logit detach;
- S17 relation-priority norm-balanced shared-gradient update;
- S17 loss partition and coefficients;
- shared bias-free 256→128 projection;
- state-once;
- full-K;
- opaque option IDs;
- all original A13 parameters frozen;
- HIRACore frozen;
- zero learned downstream scorer/router/gate/calibrator.

S29 changes only which final-block A13 linear layers receive LoRA.

## 4. Frozen LoRA configuration

All S29 LoRA:
- rank **8**
- alpha **8.0**
- dropout **0.0**
- zero-initialized B matrix
- original linear frozen

Final BERT block index:
- **5** of 6.

### 4.1 Existing attention LoRA

- `attention.self.query`, 256→256: **4,096**
- `attention.self.key`, 256→256: **4,096**
- `attention.self.value`, 256→256: **4,096**
- `attention.output.dense`, 256→256: **4,096**

Attention subtotal: **16,384**.

### 4.2 New FFN LoRA

- `intermediate.dense`, 256→1024:
  - A: 8×256 = 2,048
  - B: 1024×8 = 8,192
  - subtotal **10,240**
- `output.dense`, 1024→256:
  - A: 8×1024 = 8,192
  - B: 256×8 = 2,048
  - subtotal **10,240**

FFN subtotal: **20,480**.

A13 LoRA total:
**36,864**.

Shared relation projection:
**32,768**.

Exact physical trainable total:
**69,632**.

No bias, LayerNorm, embedding, earlier layer or HIRACore parameter may become trainable.

## 5. Required implementation isolation

S29 must use a new full-block LoRA helper rather than changing the semantics of the existing S6 attention-only helper.

Required:
- existing `inject_a13_last_attention_lora` behavior remains unchanged;
- existing S6-S28 tests remain green;
- S29 full-block iterator returns exactly six LoRA modules in a frozen order;
- checkpoint keys cover all six modules;
- load/save roundtrip preserves tensors exactly.

## 6. S29-A0

A0 uses wholly fresh English semantic rows and is diagnostic only.

Required identity:
- A13 token output exact identity to attention-only S17 initialization;
- A13 pooled output exact identity;
- raw primary logits exact identity;
- relation logits exact identity;
- fused logits exact identity;
- selected choices exact identity.

Because every S29 LoRA B matrix is zero at A0, FFN expansion must be a true no-op.

Required capacity:
- attention LoRA **16,384**
- FFN LoRA **20,480**
- total A13 LoRA **36,864**
- projection **32,768**
- total physical trainable **69,632**
- runtime trainable at frozen A0 **0**
- original A13 trainable **0**
- HIRACore trainable **0**

Required gradient court on the trainable S29 core:
- at least one attention-LoRA B gradient nonzero;
- FFN intermediate LoRA B gradient nonzero;
- FFN output LoRA B gradient nonzero;
- projection gradient nonzero;
- primary/relation shared-gradient partition still measurable;
- S17 norm-balanced rule finite and valid.

Required mechanics:
- full-K;
- state-once;
- option permutation;
- probability-mass error <= **1e-6**;
- relation delta **0**;
- S14 fusion unchanged;
- six-module checkpoint roundtrip/replay.

A0 semantic accuracy cannot tune S29.

## 7. Fresh TRAIN/DEV

Only after A0 QUALIFIED + frozen interpretation + exact-head CI + one-shot authorization.

Intended authority:
- seed **50001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S29 domains
- K=4
- two state views
- two question views per semantic query
- two semantic option views
- 24 epochs
- batch 16
- AdamW lr 2e-4
- weight decay 0.01
- grad clip 1.0

All S17 losses/coefficients and selector/gates remain frozen unless explicitly preregistered before any S29-A0 semantic exposure.

Forbidden:
- exact S0-S28 exposed rows;
- S29-A0 rows in TRAIN/DEV;
- M5 final/confirmatory rows;
- W29-W34 sealed rows.

## 8. Stop rule

After S29 DEV exposure:
- no rank change;
- no alpha/dropout change;
- no additional layer adaptation;
- no loss coefficient change;
- no seed/LR/epoch/batch retry;
- no fusion/optimizer/scorer change;
- no selector/gate weakening;
- no second DEV run.

Scientific FAIL is valid.

Only DEV_READY may open sealed confirmation, multilingual transfer or matched Laya/Jev evaluation.
