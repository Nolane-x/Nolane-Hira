# HIRA V1 S41 matched DEV receipt

Status: **FROZEN FROM SEALED SCIENTIFIC RUN LOG**

Issue: #265
PR: #266

## Canonical court

- run `37102038337`
- scientific head `ae60b7dfe7e587efeaf5a0c0b06dbd5799d808de`
- seed **62001**
- step 10 fresh matched TRAIN/DEV: **SUCCESS**
- outcome `HIRA_V1_S41_MATCHED_OPTIMIZER_STEP_ANCHORED_DEV_COMPLETE`
- scientific authority `V1_S41_FRESH_MATCHED_NATIVE_REFERENCE_VS_OPTIMIZER_STEP_ANCHORED_FULL_BILINEAR`

The court completed all 24 epochs and emitted a complete result receipt.

## Post-court verifier failure

Step 11 failed before integrity freeze/artifact upload.

Cause:
- emitted schema: `hira-v1-s41-matched-optimizer-step-anchored-coadaptation-train-dev-v1`
- verifier expected: `hira-v1-s41-matched-optimizer-step-anchored-train-dev-v1`

The verifier omitted the frozen `coadaptation` token and failed on its first result assertion.

Therefore:
- **no rerun is authorized**;
- scientific DEV exposure is valid and final;
- no uploaded S41 matched artifact exists;
- the exact emitted receipt is frozen in `research/HIRA-V1-S41-MATCHED-RECEIPT.json`;
- checkpoint hashes recorded inside the receipt remain evidence, but the ephemeral checkpoint files were not uploaded because step 13 was skipped.

## Selected reference — epoch 24

- gates: 14/22
- DEV_READY: **false**
- checkpoint sha256: `3c232d0c4b7a44de5250a6dc11ba3ef92b00a46b34beb415a9546a4fca417f1f`

Fused:
- canonical **0.4739583333333333**
- paraphrase **0.4010416666666667**
- paired **0.18229166666666666**
- question-swap **0.5520833333333334**
- agreement **0.6276041666666666**
- JS **0.026823726560299594**

Relation:
- canonical **0.3229166666666667**
- paraphrase **0.2786458333333333**
- agreement **0.5911458333333334**

Signature:
- same-option cosine **0.8467475871245066**
- discrimination margin **0.20094152788321176**

## Selected treatment — epoch 15

- gates: 19/27
- DEV_READY: **false**
- checkpoint sha256: `08ebfdb6e4e0b94ac73c0fbd97ee6862242fe02e3b56ee9515dd5bf3f5be1c05`

Fused:
- canonical **0.5963541666666666**
- paraphrase **0.484375**
- paired **0.3333333333333333**
- question-swap **0.8697916666666666**
- agreement **0.6432291666666666**
- JS **0.029318218934349716**

Relation:
- canonical **0.5651041666666666**
- paraphrase **0.4609375**
- agreement **0.71875**

Signature:
- same-option cosine **0.882604663570722**
- discrimination margin **0.0808326993137598**

## Treatment minus reference

Correctness:
- fused canonical **0.12239583333333331**
- fused paraphrase **0.08333333333333331**
- relation canonical **0.24218749999999994**
- relation paraphrase **0.18229166666666669**
- primary canonical **0.13541666666666669**
- primary paraphrase **0.09114583333333331**
- paired **0.15104166666666666**
- question-swap **0.31770833333333326**

Transport/stability:
- fused agreement **0.015625**
- relation agreement **0.12760416666666663**
- fused JS **0.0024944923740501217**
- same-option signature cosine **0.035857076446215386**
- signature discrimination margin **-0.12010882856945196**

## Optimizer-step anchor diagnostics

- mean conflict rate **0.734375**
- mean pre-dot **0.0006909580706071162**
- mean post-dot **-0.00003552509259545435**
- mean applied anchor dot **-0.00003552508143682874**
- mean applied rounding bound **0.00003554596970020622**
- max runtime rounding ratio **0.4995121951219512**
- max W rounding ratio **0**
- mean anchor **0.20131434196971695**
- selected-epoch W norm **12.501283645629883**

## Frozen interpretation

**Case B — correctness retained but transport is not fully protected.**

Why:
- correctness is retained strongly and exceeds the matched reference by double-digit points on key endpoints;
- fused and relation agreement no longer collapse as in S38 and instead improve;
- however signature discrimination margin falls by **-0.12010882856945196**, leaving treatment absolute margin at only **0.0808326993137598** and failing the frozen signature-margin gate;
- fused JS is also slightly worse.

So optimizer-step anchoring fixes a large part of the S38 transport failure, but a cosine-to-reference anchor is insufficient to preserve **relative option discrimination geometry**.

Treatment is not DEV_READY. No confirmation or external Laya/Jev evaluation is authorized.

## Safeguards

- post-DEV tuning: **false**
- second DEV: **false**
- sealed confirm: **false**
- multilingual probe: **false**
- production-ready claim: **false**
