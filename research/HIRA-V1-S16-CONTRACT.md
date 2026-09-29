# HIRA V1 S16 contract — Shared-Surface Gradient Surgery

Status: **OPEN / PREREGISTERED BEFORE S16-A0 EXPOSURE**

Issue: #210

Base main:

`744bcd246e520970743701eba5ffe8a4a194bb00`

S15 is frozen as `HIRA_V1_S15_GRADIENT_ISOLATED_FUSION_DEV_FAIL`.

## 1. Motivation

S15 proved that direct fused-primary relation-logit gradients can be eliminated exactly while preserving S14 inference.

Yet fresh DEV relation quality collapsed:
- S14 relation canonical accuracy: **0.5833333333**
- S15 relation canonical accuracy: **0.2890625**
- S15 relation signed margin: **-0.1557084620**

At the same time S15 raw triadic canonical accuracy rose to **0.5755208333**, while fused accuracy fell to **0.5182291667**.

Therefore the dominant remaining interference is on the **shared physical A13-LoRA + projection surface**, not merely the relation-logit autograd edge.

## 2. Frozen physical and inference surface

Trainable:
- A13 final-attention LoRA: **16,384**
- shared bias-free 256->128 projection: **32,768**
- total: **49,152**

Frozen:
- original A13
- HIRACore
- reliability/calibration
- adaptive budget
- relation refinement

Excluded:
- learned optimizer
- learned gradient router
- learned fusion head
- fitted gradient coefficient
- task/domain/language head
- W34

S16 surgery adds **0 learned parameters**.

Inference remains numerically identical to S14/S15:

`fused = 0.5 * standardized(triadic) + 0.5 * standardized(relation)`

with fusion epsilon **1e-6**.

## 3. Frozen loss partition

For each TRAIN batch, compute one forward graph and split the exact S15 scalar objective into two blocks.

### Primary block

`L_primary = L_fused_decision + 0.05 * L_option_view + 0.25 * L_fused_cross_view_JS`

where:
- `L_fused_decision` is canonical+paraphrase fused CE plus paired swap margin at coefficient **0.25**, margin **0.20**;
- option-view InfoNCE uses temperature **0.10**;
- S15 protected fusion remains in force: relation logits are detached from fused-primary fusion.

### Relation-preservation block

`L_relation = 0.10 * L_relation_CE + 0.15 * L_signature_canonicalization`

where:
- relation CE uses canonical+paraphrase relation logits;
- signature canonicalization uses separation margin **0.20**;
- role temperature **0.10**;
- pair temperature **0.10**;
- relation contrastive temperature **0.10**.

No loss coefficient differs from S15.

## 4. Frozen gradient extraction

Ordered trainable surface:
1. all A13 LoRA parameters in existing deterministic module/state order;
2. shared projection weight.

For each batch:
- compute `g_p = grad(L_primary, surface, retain_graph=True)`;
- compute `g_r = grad(L_relation, surface)`;
- replace any structurally unused gradient with an all-zero tensor of the exact parameter shape;
- flatten only conceptually for global dot/norm diagnostics; implementation may keep tensor list form;
- no per-module weighting.

## 5. Frozen relation-priority surgery

Fixed epsilon:

`1e-12`

Global dot:

`d = dot(g_p, g_r)`

If `d < 0` and `||g_r|| > 0`:

`g_p' = g_p - d / (||g_r||^2 + 1e-12) * g_r`

Else:

`g_p' = g_p`

Combined update:

`g = g_p' + g_r`

Then:
- write `g` into the real parameter `.grad` buffers;
- apply the existing global grad clip **1.0**;
- AdamW step with the same LR/weight decay as S15.

No random PCGrad order.
No symmetric second projection.
No learned/fitted gradient scale.

This deliberately gives relation preservation priority only when the raw primary update is actively anti-aligned with it.

## 6. Required surgery invariants

A0/unit contracts must prove:
- conflicting primary/relation gradients become non-negative after projection, within numeric tolerance;
- non-conflicting primary gradients remain bit-identical;
- zero relation gradient leaves primary unchanged;
- combined update equals projected primary + relation;
- common positive joint rescaling is equivariant;
- no learned/module state exists;
- same inference as S15;
- exact 49,152 trainable surface.

A0 must additionally record on wholly fresh cases:
- primary/relation global dot
- gradient cosine
- conflict boolean
- projection coefficient
- primary/relation/projected/combined norms
- post-projection primary-relation dot

A0 is diagnostic only.

## 7. Fresh TRAIN / DEV

Frozen:
- TRAIN: **768 wholly fresh semantic cases**
- DEV: **192 wholly fresh semantic cases**
- 12 fresh domains
- fresh lexical banks
- unseen DEV wording families
- K=4
- two state views
- two question wording views per semantic query
- two semantic views per option

Forbidden:
- every S0-S15 A0/TRAIN/DEV row
- M5 final/confirmatory rows
- W29-W34 sealed rows

## 8. Optimizer

- seed: **25001**
- AdamW
- 24 epochs
- batch size: 16 semantic cases
- lr: **2e-4**
- weight decay: **0.01**
- post-surgery grad clip: **1.0**

No optimizer hyperparameter changes from S15 except seed/fresh evidence identity.

## 9. Frozen DEV selection order

1. fused canonical paired both-correct
2. fused canonical accuracy
3. canonical relation-binding accuracy
4. canonical relation-binding signed margin
5. fused canonical signed margin
6. fused question-swap choice-change
7. fused cross-view selected-choice agreement
8. mean same-option signature cosine
9. mean signature same-vs-strongest-wrong margin
10. lower fused canonical decision loss
11. earlier epoch

Relative to S15, relation quality is deliberately moved ahead of later primary diagnostics because S16 specifically tests preservation of relation semantics while retaining fusion.

## 10. Frozen DEV gate

`HIRA_V1_S16_SHARED_GRADIENT_SURGERY_DEV_READY` requires all:
- fused canonical accuracy >= **0.85**
- fused canonical paired both-correct >= **0.75**
- fused question-swap choice-change >= **0.80**
- fused cross-view selected-choice agreement >= **0.95**
- fused cross-view mean JS <= **0.05**
- fused canonical signed margin >= **0.15**
- canonical relation-binding accuracy >= **0.80**
- canonical relation-binding signed margin >= **0.15**
- mean same-option signature cosine >= **0.90**
- mean signature same-vs-strongest-wrong margin >= **0.15**
- fused option-order flip <= **0.02**
- fused probability-mass error <= **1e-6**
- full-K
- relation delta = 0
- state-once
- exact 49,152 trainable params
- original A13 frozen
- HIRACore frozen
- surgery/fusion/canonicalizer/downstream learned params = 0

Additional diagnostic, not a success gate:
- conflict rate over TRAIN batches.

## 11. No post-DEV tuning

Forbidden after DEV exposure:
- surgery epsilon changes
- switching to symmetric/random PCGrad
- loss repartition
- loss coefficient changes
- fusion weight/epsilon changes
- seed/LR/template retry
- gate weakening

Scientific FAIL is valid and must be frozen.

## 12. Sealed boundary

Only DEV READY may open:
1. one-shot sealed English confirmation
2. separately preregistered zero-training Vietnamese transfer

S16 cannot independently establish Laya/Jev parity, multilingual qualification, reliability/OOD readiness, or production readiness.
