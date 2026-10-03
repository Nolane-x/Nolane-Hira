# HIRA V1 S44 contract — Private Correction Representation Fork

Status: **PREREGISTERED / NO S44-A0 EXPOSURE**

Issue: #271

Parent:
- S43 merged main `0689990b618a8c3a1b7200469fe55deb19e55405`
- S43 fresh DEV run `37112732199`
- S43 artifact `11270379708`
- S43 interpretation **Case C**

## Scientific question

> Can Hira preserve the native transport representation on the exact matched reference trajectory while giving correctness a separate post-encoder nonlinear representation that co-adapts only inside the correction branch?

S38 showed that correctness benefits from representation co-adaptation.
S39 showed that W-only correction on frozen native signatures is insufficient.
S43 showed that constraining transport-critical native movement suppresses useful fused/primary correctness.

S44 changes representation ownership rather than adding another anchor.

## Native path

Reference and treatment native runtimes use exactly:
- S35 native 256D relation/signature geometry;
- S17 primary/fusion shell;
- A13 LoRA 16,384;
- shared primary projection 32,768;
- native trainable surface **49,152**;
- original A13 frozen;
- HIRACore frozen;
- same initialization;
- same rows/order;
- same native S35/S17 objective;
- same AdamW configuration.

Treatment native runtime MUST remain trajectory-identical to reference:
- LoRA state;
- primary projection state;
- native pre-correction relation logits;
- native signatures.

Correction loss MUST NOT backpropagate into native LoRA/projection/signatures/query tokens.

## Private correction representation

Inputs:
- detached native option signature `s [256]`;
- detached S38 masked-mean + L2 query summary `q [256]`.

Concatenate:
`x = concat(s, q)` width **512**.

Private adapter:
`h = GELU(A x)`
`d = B h`
`s_private = L2Norm(s + d)`

Frozen sizes:
- hidden width **64**;
- A shape **64 x 512** = **32,768**;
- B shape **256 x 64** = **16,384**;
- private adapter total **49,152**;
- no bias;
- no layer norm;
- no gate;
- no learned scale;
- query norm epsilon **1e-12**;
- private signature norm epsilon **1e-12**.

Initialization:
- A uses deterministic local seed **65044**;
- A entries use zero-mean normal with std `sqrt(2/512)`;
- B is exact zero;
- W is exact zero.

The zero-B state makes the residual exactly zero.
Private/native signature equality is checked under the frozen normalization tolerance; corrected logits are exactly native at W=0.

## Correction head

Full bilinear W:
- shape **256 x 256**;
- **65,536 params**;
- zero-init;
- no bias;
- residual scale **1.0**.

Correction residual:
`r_k = s_private_k^T W q`

Corrected relation logits:
`l_corr = stopgrad(l_native) + r`.

Correction-only trainable surface:
- A 32,768
- B 16,384
- W 65,536
- total **114,688**

Treatment total trainable:
- native 49,152
- correction-only 114,688
- total **163,840**

## State-once

No second encoder pass.

The private correction adapter consumes tensors already emitted by the single native encoded batch.
State-view encoder count must therefore remain identical to the reference arm.

## Frozen correction objective

Reuse the exact S39 correction objective.

For corrected canonical/paraphrase relation logits:
`CE_corr = 0.5 * (CE(l_corr_c, gold) + CE(l_corr_p, gold))`

`L_corr = 0.10 * CE_corr`

The coefficient **0.10** is the already-frozen S17/S39 `BINDING_COEFFICIENT`.

No fused-specific loss.
No new coefficient.
No primary loss in the correction branch.

Native treatment runtime receives only the exact native S35/S17 objective.

## Optimizers

Native reference and treatment:
- exact existing S35/S17 AdamW semantics;
- identical initialization/state/order;
- correction params excluded.

Private correction optimizer:
- AdamW;
- lr **2e-4**;
- weight decay **0.01**;
- same frozen default betas/eps as the existing S39 correction path;
- global correction grad clip **1.0** over A/B/W together;
- no W-only LR/scheduler.

The native optimizer and correction optimizer are separate by ownership, not by tuned learning rate.

## S44-A0 authority

Wholly fresh S44-A0 diagnostic cases only.

Must prove:
- native reference/treatment zero-init identity;
- native runtime update identity under one matched native step;
- A count 32,768;
- B count 16,384;
- private adapter 49,152;
- W 65,536;
- correction-only 114,688;
- treatment total 163,840;
- B exactly zero at initialization;
- W exactly zero at initialization;
- corrected/native relation logits exact identity at zero W;
- private/native signature difference within frozen normalization tolerance;
- one encoded batch only / no second encoder pass;
- correction loss -> native LoRA gradient exactly zero;
- correction loss -> native projection gradient exactly zero;
- correction loss -> reference runtime exactly zero;
- native objective -> A/B/W exactly zero;
- deterministic warm-start sequence from zero initialization causes:
  - W to receive/update first;
  - B to become live after W is nonzero;
  - A to become live after B and W are nonzero;
- after warm start, correction loss gradients into A/B/W are finite/nonzero;
- nonzero private branch differs from the W-only S39 residual under a deterministic probe;
- logical-option permutation equivariance;
- question permutation/padding invariance;
- arbitrary K=3/K=7;
- native projection independence;
- checkpoint roundtrip;
- probability mass/full-K;
- state-once mechanics.

A0 semantic values are diagnostic only and MUST NOT tune:
- hidden width;
- activation;
- init scale/seed;
- correction coefficient;
- optimizer;
- capacity;
- selector/gates.

## Intended fresh matched TRAIN/DEV

Only after:
1. contract frozen;
2. interpretation plan frozen;
3. S44-A0 qualified;
4. A0 receipt frozen;
5. wholly fresh S44 authority frozen;
6. trainer/workflow frozen;
7. exact staged-head CI PASS;
8. separate one-shot marker.

Intended:
- seed **65001**;
- TRAIN 768;
- DEV 192;
- 12 wholly fresh S44 domains;
- K=4;
- two state views;
- two question wording views;
- two option views;
- 24 epochs;
- batch 16;
- identical native rows/order;
- one DEV only.

Freshness:
- no exact S0-S43 exposed rows;
- no S44-A0 rows;
- no M5 final/confirm rows;
- no W29-W34 sealed rows.

## Stop rule

After one S44 DEV:
- no hidden-width sweep;
- no activation sweep;
- no A-init sweep;
- no bias/gate/layernorm;
- no residual-scale tuning;
- no correction-loss change;
- no native-gradient leakage;
- no second encoder path;
- no native/private mixing coefficient;
- no W-only LR/scheduler;
- no seed/LR/epoch/batch retry;
- no gate weakening;
- no second DEV.

Scientific FAIL is valid.
No external Laya/Jev benchmark before a separately confirmed DEV_READY candidate.
