# HIRA V1 S21 closure — Role-Gated Content Triadic Scoring

Status: **CLOSED — DEV FAIL / FACTORIZATION IMPROVES VIEW ROBUSTNESS BUT RELATION-PRIORITY CONFLICT PRESSURE LIMITS CANONICAL PERFORMANCE**

Issue: #225  
PR: #226

## Authority

A0:
- run `36701343136`
- artifact `11089967708`
- digest `sha256:27219a7f99f8ef8f6f2552ada65ed97e6a99a6bc2234147177fdffdbb365c31b`
- head `8eafefb9cc49dc1c8e8662b760b0526742645117`
- outcome `HIRA_V1_S21_A0_ROLE_CONTENT_READY`

Canonical fresh TRAIN/DEV:
- run `36703344115`
- artifact `11091248529`
- digest `sha256:af58a3fd91c282dab02641bb8082e1f66b9d8bff007b6fe92fafc652ee5dcf49`
- scientific head `fddc2aab1fe71952bf8a26606d7ea8a5d344f23e`
- outcome `HIRA_V1_S21_ROLE_GATED_CONTENT_DEV_FAIL`
- selected epoch **13**
- checkpoint SHA256 `727fd536e819cce6235e34f47abbdfd8778d68213c5b0858ae58c8b8e1aae103`

Harness aborts preserved separately:
- A0 run `36692441039`: py_compile failure before A0 job;
- TRAIN/DEV run `36702356642`: stale exact-logit-identity guard before case generation / TRAIN / DEV.

Neither abort produced scientific authority evidence.

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S21 changed the **primary scorer itself** while keeping the physical trainable surface fixed.

Role-gated content factorization:
- question-conditioned state/option role distributions;
- role anchors;
- explicit removal of role direction from token content;
- symmetric residual state↔option content binding;
- per-view score = **0.50 role compatibility + 0.50 content binding**;
- role temperature **0.10**;
- learned factorization params **0**.

Kept:
- A13 LoRA **16,384**
- shared projection **32,768**
- exact trainable surface **49,152**
- S13 relation expert / signature canonicalization
- S14 equal-weight standardized fusion
- S17 fused-output JS objective
- S17 relation-priority norm-balanced shared-gradient optimization

## A0 result

The structural mechanism was proven before DEV:
- same-role/same-content > same-role/wrong-content margin: **+0.2051138282**
- same-role/same-content > wrong-role/same-content margin: **+0.5093060732**
- question-role switch selected option: **0 -> 2**
- option permutation max abs: **0.0**
- degenerate residual geometry finite
- factorization added params: **0**
- physical surface: **49,152**
- full-K/state-once/fusion mechanics PASS

## Selected DEV — epoch 13

Fused:
- canonical accuracy: **0.6145833333**
- paraphrase accuracy: **0.5651041667**
- paired both-correct: **0.3697916667**
- question-swap choice-change: **0.8333333333**
- cross-view selected-choice agreement: **0.6588541667**
- cross-view mean JS: **0.0395950909**
- canonical signed margin: **+0.1440208815**
- paraphrase signed margin: **+0.0138455929**

Primary role-gated content expert:
- canonical accuracy: **0.5807291667**
- paraphrase accuracy: **0.5182291667**
- cross-view agreement: **0.6484375**

Relation expert:
- canonical accuracy: **0.6484375**
- paraphrase accuracy: **0.5963541667**
- canonical signed margin: **+0.2356135895**
- paraphrase signed margin: **+0.1948801869**
- cross-view agreement: **0.6744791667**

Signatures:
- same-option cosine: **0.7969388465**
- same-vs-strongest-wrong margin: **0.0455976037**

Expert interaction:
- canonical expert top-1 agreement: **0.8958333333**
- paraphrase expert top-1 agreement: **0.8671875**

Mechanical:
- full-K: PASS
- state-once: PASS
- relation delta: **0**
- option-order flip: **0**
- max fused probability-mass error: **1.1920928955e-07**
- exact trainable surface: **49,152**

## Gate result

PASS:
- question-swap >= 0.80
- fused mean JS <= 0.05
- relation signed margin >= 0.15
- option-order / mass / full-K / state-once / capacity gates

FAIL:
- fused canonical >= 0.85
- paired >= 0.75
- fused cross-view agreement >= 0.95
- fused canonical margin >= 0.15 (misses narrowly at 0.1440)
- relation canonical >= 0.80
- signature cosine >= 0.90
- signature discrimination margin >= 0.15

DEV_READY is not authorized.

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **1.9167083229 -> 0.7739952703**
- decision loss: **1.6494018187 -> 0.6274795768**
- relation-binding loss: **1.3657605623 -> 0.5195634067**
- canonicalization loss: **0.3105653316 -> 0.1764784005**
- fused consistency JS: **0.0675813666 -> 0.0038957067**
- primary block: **1.7335474590 -> 0.6955671745**
- relation block: **0.1831608613 -> 0.0784281023**

The structural scorer is trainable and relation learning remains strong.

## Gradient-conflict evidence

S21 retained S17's **relation-priority** normalized conflict projection.

Conflict rate by epoch:
- epoch 1: **0.5416666667**
- epoch 6: **0.1458333333**
- selected epoch 13: **0.7291666667**
- epoch 21: **0.7916666667**
- epoch 22: **0.8333333333**
- epoch 24: **0.8125**
- mean over 24 epochs: **0.5998263889**

Selected epoch 13:
- primary gradient norm: **1.3187688440**
- relation gradient norm: **0.3862016831**
- normalized pre-dot: **-0.1691433275**
- normalized post-dot: **+0.1412833923**
- mean projection coefficient: **-0.3104267173**

Epoch 24:
- normalized pre-dot: **-0.3396972206**
- conflict rate: **0.8125**

This is materially more conflict-heavy than S17 (mean conflict ~**0.3220**).

## Fresh DEV extrema

- fused canonical best: **0.6145833333** at epoch 13
- fused paraphrase best: **0.6041666667** at epoch 23
- paired best: **0.3697916667** at epochs 7/8/13
- question-swap best: **0.8854166667** at epoch 7
- fused agreement best: **0.7734375** at epoch 4
- fused JS minimum: **0.0150162739** at epoch 4
- fused canonical margin best: **+0.1585271782** at epoch 8
- primary canonical best: **0.6015625** at epoch 8
- primary paraphrase best: **0.546875** at epoch 23
- primary agreement best: **0.75** at epoch 6
- relation canonical best: **0.6484375** at epoch 13
- relation paraphrase best: **0.6380208333** at epoch 23
- relation canonical margin best: **+0.2478801484** at epoch 11
- relation paraphrase margin best: **+0.2202820207** at epoch 23
- signature cosine best: **0.9347053816** at epoch 4
- signature margin best: **0.1177686416** at epoch 5

## Comparison to S17 frontier

S17 selected:
- fused canonical **0.7213541667**
- fused paraphrase **0.5546875**
- paired **0.5104166667**
- question-swap **0.984375**
- fused agreement **0.5286458333**
- fused margin **+0.3138313380**
- primary canonical/paraphrase **0.5911458333 / 0.4427083333**
- relation canonical/paraphrase **0.640625 / 0.6380208333**
- relation canonical margin **+0.2073315941**
- mean gradient-conflict rate **~0.3220**

S21 selected:
- fused canonical **0.6145833333**
- fused paraphrase **0.5651041667**
- paired **0.3697916667**
- question-swap **0.8333333333**
- fused agreement **0.6588541667**
- fused margin **+0.1440208815**
- primary canonical/paraphrase **0.5807291667 / 0.5182291667**
- relation canonical/paraphrase **0.6484375 / 0.5963541667**
- relation canonical margin **+0.2356135895**
- mean gradient-conflict rate **0.5998263889**

Important:
- primary canonical is near S17 (-1.04 pp);
- primary paraphrase improves by **+7.55 pp**;
- fused paraphrase improves by **+1.04 pp**;
- fused cross-view agreement improves by **+13.02 pp**;
- relation canonical and margin are not destroyed;
- canonical fused and paired performance regress substantially.

## Scientific conclusion

S21 rejects the claim that role/content factorization alone is sufficient for DEV_READY.

But S21 does **not** support discarding the factorization immediately.

The factorized primary:
- generalizes better to paraphrase;
- improves view agreement;
- preserves strong relation canonical accuracy/margin;
- remains within the same 49,152-parameter budget.

The strongest new failure signal is optimizer interaction:

> S17's relation-priority conflict projection is activated on ~60% of S21 steps and >80% late in training. The new primary geometry is therefore frequently forced to discard components that conflict with the relation objective.

This makes the optimizer-priority assumption the next clean controlled variable.

## Next controlled hypothesis

S22 should keep the S21 role/content architecture and all losses/capacity fixed, changing **only conflict priority**.

When unit-normalized primary/relation gradients conflict:

S17/S21:
- project primary away from relation;
- relation is untouched.

S22:
- **project relation away from primary**;
- primary is untouched.

Then combine the primary direction with the projected relation direction, normalize, and scale by the same arithmetic mean raw norm.

Everything else remains frozen:
- exact 49,152 params
- role temperature 0.10
- role/content weights 0.50/0.50
- all S17/S21 losses
- S14 fusion
- seed/data wholly fresh
- no new learned state

This directly tests whether relation-priority optimization was suppressing the new factorized primary geometry.

Production-ready remains false.
Laya/Jev parity remains unestablished.
