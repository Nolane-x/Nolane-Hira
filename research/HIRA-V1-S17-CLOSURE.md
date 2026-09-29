# HIRA V1 S17 closure — Norm-Balanced Shared Gradient Optimization

Status: **CLOSED — DEV FAIL / STRONG POSITIVE RESULT, CROSS-VIEW PAIRED CONSISTENCY BOTTLENECK**

Issue: #214  
PR: #215

## Authority

A0:
- run `36574252783`
- artifact `11034934999`
- artifact digest `sha256:4a29beb1cf410f3f98585367d6a0dc0dc6f73be37aee64b91e4030d814f0f0fe`
- outcome `HIRA_V1_S17_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run `36576470197`
- artifact `11039530381`
- artifact digest `sha256:79d86431618e7f832d38a4c2e46942c7af391788660635aa5eab13523602a2d1`
- outcome `HIRA_V1_S17_NORM_BALANCED_GRADIENT_DEV_FAIL`
- selected epoch **23**
- checkpoint SHA256 `b0a292524c2bbf7b8eb56d583cbe3196aa739ffbca6eccea80240ef057c3851e`

One earlier TRAIN/DEV attempt `36575533502` aborted in the freshness guard before training because an exact prior question overlap was detected. The fresh wording set was corrected before scientific exposure; the model surface, optimizer, norm-balance rule, loss coefficients, selection order, gates and seed remained unchanged.

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen surface / optimizer mechanism

Inference remains numerically identical to S14-S16.

Trainable physical surface:
- A13 final-attention LoRA: **16,384**
- shared bias-free 256->128 projection: **32,768**
- total: **49,152**

Added learned balancing/router parameters: **0**.

For nonzero primary/relation shared gradients:

`u_p = g_p / ||g_p||`

`u_r = g_r / ||g_r||`

If `dot(u_p,u_r) < 0`, relation-priority project `u_p` away from `u_r`.

Then:

`d = normalize(u_p' + u_r)`

`s = 0.5 * (||g_p|| + ||g_r||)`

`g = s * d`

Fixed epsilon: **1e-12**.

## Selected DEV — epoch 23

Fused primary:
- canonical accuracy: **0.7213541667**
- paraphrase accuracy: **0.5546875**
- paired both-correct: **0.5104166667**
- question-swap choice-change: **0.984375**
- cross-view selected-choice agreement: **0.5286458333**
- cross-view mean JS: **0.0416121766**
- canonical signed margin: **0.3138313380**

Raw triadic:
- canonical accuracy: **0.5911458333**
- paraphrase accuracy: **0.4427083333**
- cross-view agreement: **0.5208333333**

Relation:
- canonical accuracy: **0.640625**
- paraphrase accuracy: **0.6380208333**
- canonical signed margin: **0.2073315941**
- paraphrase signed margin: **0.2545196755**
- cross-view agreement: **0.6380208333**

Relation signatures:
- same-option cosine: **0.8373316179**
- same-vs-strongest-wrong margin: **0.0794288889**

Mechanical invariants:
- full-K: PASS
- option-order flip: **0.0**
- relation delta: **0.0**
- state-once: PASS
- max probability-mass error: **1.7881393433e-07**

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **1.7188058421 -> 0.8336605690**
- decision loss: **1.4601607819 -> 0.6914654970**
- relation-binding loss: **1.4047878062 -> 0.5074664789**
- canonicalization loss: **0.3045701105 -> 0.1537828504**
- primary block: **1.5326415449 -> 0.7598464936**
- relation block: **0.1861643018 -> 0.0738140771**

Unlike S16, relation binding now learns strongly.

Norm-balanced TRAIN conflict rate:
- mean: **0.3220486111**
- rule: `equal_direction_relation_priority_projection`
- scale: arithmetic mean raw norm

## Fresh DEV extrema

- fused canonical accuracy best: **0.7213541667** at epoch 23
- paired both-correct best: **0.5104166667** at epoch 23
- question-swap best: **0.984375** at epoch 23
- fused canonical margin best: **0.3138313380** at epoch 23
- relation canonical accuracy best: **0.6640625** at epoch 22
- relation signed margin best: **0.2775976347** at epoch 17
- same-option signature cosine best: **0.9469363044** at epoch 5
- signature same-vs-wrong margin best: **0.1205706494** at epoch 5
- fused cross-view agreement best: **0.6380208333** at epoch 6

## Comparison

S14:
- fused canonical: **0.6354166667**
- relation canonical: **0.5833333333**

S16:
- fused canonical: **0.5520833333**
- relation canonical: **0.34375**

S17:
- fused canonical: **0.7213541667**
- relation canonical: **0.640625**

S17 is the strongest fresh DEV result in the current v1 semantic-rescue sequence.

## Gate result

S17 materially exceeds several semantic gates:
- question-swap >= 0.80: PASS
- fused mean JS <= 0.05: PASS
- fused signed margin >= 0.15: PASS
- relation signed margin >= 0.15: PASS
- mechanical gates: PASS

It still fails:
- fused canonical accuracy >= 0.85
- paired both-correct >= 0.75
- fused cross-view selected-choice agreement >= 0.95
- relation binding accuracy >= 0.80
- same-option signature cosine >= 0.90 at selected checkpoint
- signature same-vs-wrong margin >= 0.15

Therefore DEV_READY is not authorized.

## Scientific conclusion

S17 strongly supports the norm-balancing hypothesis.

Equalizing objective direction before shared-surface conflict handling:
- restores strong relation learning;
- raises fused canonical accuracy to **72.14%**;
- raises relation canonical accuracy to **64.06%**;
- produces positive relation and fused margins;
- drives question sensitivity to **98.44%**.

The dominant remaining failure is no longer basic semantic discrimination.

The new bottleneck is:

> **Cross-view paired consistency: the model can answer the queried fact correctly with strong margins, but equivalent wording views still choose different top options too often.**

Evidence:
- canonical fused accuracy **72.14%**
- paraphrase fused accuracy **55.47%**
- paired both-correct **51.04%**
- cross-view selected-choice agreement **52.86%**
- while question-swap is **98.44%**

## Next controlled hypothesis

S18 should preserve S17 norm-balanced gradient optimization and directly train **paired worst-view consistency** without adding parameters or changing inference.

A parameter-free supervised paired-margin objective is the strongest next test:

For canonical/paraphrase fused logits and gold option, compute each view's gold-vs-hardest-wrong margin `m_c, m_p`.

`m_pair = min(m_c, m_p)`

`L_pair = relu(M - m_pair)`

with fixed margin `M = 0.20`.

This directly requires both wording views to carry a positive gold separation instead of allowing one strong view to hide a weak paired view.

Production-ready remains false.
Laya/Jev parity remains unestablished.
