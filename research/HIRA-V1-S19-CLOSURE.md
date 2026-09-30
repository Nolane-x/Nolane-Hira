# HIRA V1 S19 closure — Expert-Separated Triadic View Consistency

Status: **CLOSED — DEV FAIL / RAW-LOGIT JS IS EFFECTIVELY INERT ON FLAT TRIADIC EVIDENCE**

Issue: #221  
PR: #222

## Authority

A0:
- run `36588963863`
- artifact `11042832648`
- digest `sha256:f54819e15bc642f3424272b05ba3dc66e2418b34314e536709d72171ce858276`
- outcome `HIRA_V1_S19_A0_IDENTITY_READY`

One first TRAIN invocation:
- run `36643441796`
- terminated on the first TRAIN batch before any DEV evaluation/artifact
- cause: stale harness metric key `consistency_js` versus S19 `triadic_consistency_js`
- not scientific exposure

Canonical fresh TRAIN/DEV:
- run `36644174421`
- artifact `11068688127`
- digest `sha256:2f2f70f4d83704f5701238171faa0e23187bfe18293d8a96ffb793cc904762be`
- outcome `HIRA_V1_S19_TRIADIC_VIEW_CONSISTENCY_DEV_FAIL`
- selected epoch **20**
- checkpoint SHA256 `af302063a4188fdaa0c61d29810b2ae15dd77059767b002d83d61d4adaf930af`

No post-DEV tuning.
No sealed confirmation.
No multilingual probe.
No Laya/Jev reopening.

## Frozen intervention

S19 returned to the S17 optimizer/objective and changed only the cross-view consistency target:

- S17 norm-balanced shared-gradient optimization retained
- S18 paired margin removed
- fused-output JS removed
- raw-triadic canonical/paraphrase symmetric JS added with coefficient **0.25**
- relation CE/canonicalization unchanged
- inference unchanged
- exact **49,152** trainable physical params
- 0 learned consistency/router params

## Selected DEV — epoch 20

Fused:
- canonical accuracy: **0.6302083333**
- paraphrase accuracy: **0.453125**
- paired both-correct: **0.359375**
- question-swap choice-change: **0.71875**
- cross-view selected-choice agreement: **0.6067708333**
- cross-view mean JS: **0.0423815645**
- canonical signed margin: **+0.2158164382**
- paraphrase signed margin: **-0.2138849869**

Raw triadic:
- canonical accuracy: **0.4609375**
- paraphrase accuracy: **0.3567708333**
- cross-view agreement: **0.5494791667**

Relation:
- canonical accuracy: **0.4791666667**
- paraphrase accuracy: **0.4166666667**
- canonical signed margin: **-0.0836246746**
- paraphrase signed margin: **-0.1740764827**
- cross-view agreement: **0.7317708333**

Relation signatures:
- same-option cosine: **0.9174721142**
- same-vs-strongest-wrong margin: **0.1374289120**

Mechanical:
- full-K: PASS
- state-once: PASS
- relation delta: **0**
- option-order flip: **0**
- fused probability-mass error: **1.1920928955e-07**
- exact trainable surface: **49,152**

## Gate result

PASS:
- fused canonical signed margin >= 0.15
- fused cross-view mean JS <= 0.05
- same-option signature cosine >= 0.90
- all mechanical/capacity gates

FAIL:
- fused canonical >= 0.85
- paired >= 0.75
- question-swap >= 0.80
- fused cross-view agreement >= 0.95
- relation canonical >= 0.80
- relation signed margin >= 0.15
- signature discrimination margin >= 0.15

Therefore DEV_READY is not authorized.

## TRAIN dynamics

Epoch 1 -> 24:
- total loss: **1.7200225964 -> 0.8558772678**
- decision loss: **1.4708443855 -> 0.6898663193**
- relation-binding loss: **1.3878978789 -> 0.8670776623**
- canonicalization loss: **0.2841384380 -> 0.0785273352**
- primary block: **1.5386120453 -> 0.7573903985**
- relation block: **0.1814105570 -> 0.0984868687**

But the actual S19 intervention stayed effectively zero:

- TRAIN raw-triadic JS minimum across epochs: **4.5597310279e-09**
- maximum: **1.1515045787e-08**
- epoch 1: **8.4439193323e-09**
- epoch 24: **1.0502139177e-08**

This is many orders below the other objective terms.

## Fresh DEV extrema

- fused canonical best: **0.6302083333** at epoch 20
- fused paraphrase best: **0.53125** at epoch 19
- paired best: **0.359375** at epoch 20
- fused cross-view agreement best: **0.6927083333** at epochs 6/8
- raw-triadic cross-view agreement best: **0.8854166667** at epoch 1
- relation canonical best: **0.5104166667** at epoch 23
- relation cross-view agreement best: **0.78125** at epoch 10
- signature cosine best: **0.9470630984** at epoch 5
- signature discrimination margin best: **0.1642986350** at epoch 7

The raw-triadic top-1 agreement moves substantially while raw-logit JS remains ~1e-8.

## Comparison to S17 frontier

S17 selected:
- fused canonical **0.7213541667**
- fused paraphrase **0.5546875**
- paired **0.5104166667**
- question-swap **0.984375**
- fused agreement **0.5286458333**
- raw triadic canonical/paraphrase **0.5911458333 / 0.4427083333**
- relation canonical/paraphrase **0.640625 / 0.6380208333**
- relation canonical margin **+0.2073315941**

S19 selected:
- fused canonical **0.6302083333**
- fused paraphrase **0.453125**
- paired **0.359375**
- question-swap **0.71875**
- raw triadic canonical/paraphrase **0.4609375 / 0.3567708333**
- relation canonical/paraphrase **0.4791666667 / 0.4166666667**
- relation canonical margin **-0.0836246746**

S19 is a regression versus S17.

## Scientific conclusion

S19 rejects the hypothesis that symmetric JS on **raw** triadic logits is an effective branch-specific consistency regularizer.

The failure mode is specific:

> **Raw triadic distributions are nearly flat, so softmax/JS sees almost identical probability vectors even when tiny relative logit differences change the top-ranked option.**

Therefore:
- the S19 consistency term contributes essentially no training signal;
- top-1/ranking instability can remain large while raw-logit JS is nearly zero;
- the next consistency target must operate on the same scale-free relative evidence geometry actually used by fusion.

## Next controlled hypothesis

S20 should start from the **S17 scientific frontier**, not retain S18 or S19 interventions.

Use **Standardized Triadic Evidence Consistency**:

For each raw triadic full-K vector `x`:

`c = x - mean_K(x)`

`rms = sqrt(mean_K(c^2))`

If `rms <= 1e-6`, standardized evidence is the all-zero neutral vector.
Otherwise:

`z = c / rms`

This is the same center/RMS evidence normalization family used by S14+ inference fusion.

Cross-view loss:

`L_std = mean((z_c - z_p)^2)`

Frozen coefficient:
- **0.25**, replacing the S17 fused-output JS term
- do not duplicate consistency terms

Keep:
- S17 norm-balanced gradients
- relation CE/signature canonicalization unchanged
- exact 49,152 trainable params
- inference unchanged
- 0 learned consistency parameters

This directly exposes relative ranking/evidence differences that raw softmax JS failed to see.

Production-ready remains false.
Laya/Jev parity remains unestablished.
