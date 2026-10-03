# HIRA V1 S41 closure — Optimizer-Step-Anchored Joint Bilinear Co-Adaptation

Status: **CLOSED — ONE FRESH DEV COMPLETE / PREREGISTERED CASE B**

Issue: #265
PR: #266

## Canonical A0

- run `37101438505`
- artifact `11266051980`
- digest `sha256:b4312c61d6f1afa107192c004909579fd2b2677e651c73c2f82ce67fb4063af3`
- authority head `1b6ad4ef353a3d194b436d645fbe9c1a158e0f55`
- outcome `HIRA_V1_S41_A0_OPTIMIZER_STEP_ANCHORED_COADAPTATION_READY`

## Canonical fresh matched DEV

- run `37102038337`
- scientific head `ae60b7dfe7e587efeaf5a0c0b06dbd5799d808de`
- seed **62001**
- step 10 TRAIN/DEV: **SUCCESS**
- outcome `HIRA_V1_S41_MATCHED_OPTIMIZER_STEP_ANCHORED_DEV_COMPLETE`

The emitted receipt is frozen at:
- `research/HIRA-V1-S41-MATCHED-RECEIPT.json`
- `research/HIRA-V1-S41-MATCHED-RECEIPT.md`

No second DEV.

## Post-court verifier typo

Step 11 failed on its **first** result assertion because:
- emitted schema: `hira-v1-s41-matched-optimizer-step-anchored-coadaptation-train-dev-v1`
- workflow verifier expected: `hira-v1-s41-matched-optimizer-step-anchored-train-dev-v1`

The verifier accidentally omitted `coadaptation`.

This occurred **after** all 24 epochs and receipt emission.

Consequences:
- scientific exposure is final and valid;
- rerun is forbidden;
- integrity/artifact upload steps were skipped;
- the exact receipt is preserved from the sealed job log;
- checkpoint hashes remain recorded, but ephemeral checkpoint files were not uploaded.

This is a post-court verifier defect, not a model failure and not authority for another DEV.

## Matched result

Reference selected epoch **24**:
- fused canonical/paraphrase: **0.4739583 / 0.4010417**
- relation canonical/paraphrase: **0.3229167 / 0.2786458**
- fused agreement: **0.6276042**
- relation agreement: **0.5911458**
- signature cosine/margin: **0.8467476 / 0.2009415**
- gates **14/22**
- DEV_READY false

Treatment selected epoch **15**:
- fused canonical/paraphrase: **0.5963542 / 0.4843750**
- relation canonical/paraphrase: **0.5651042 / 0.4609375**
- fused agreement: **0.6432292**
- relation agreement: **0.7187500**
- signature cosine/margin: **0.8826047 / 0.0808327**
- gates **19/27**
- DEV_READY false

## Treatment minus reference

Correctness:
- fused canonical **+0.1223958**
- fused paraphrase **+0.0833333**
- relation canonical **+0.2421875**
- relation paraphrase **+0.1822917**
- primary canonical **+0.1354167**
- primary paraphrase **+0.0911458**
- paired **+0.1510417**
- question-swap **+0.3177083**

Transport/stability:
- fused agreement **+0.0156250**
- relation agreement **+0.1276042**
- fused JS **+0.0024945** (slightly worse)
- same-option signature cosine **+0.0358571**
- signature discrimination margin **-0.1201088**

## Actual-step anchor behavior

Across training:
- mean conflict rate **0.734375**
- mean pre-dot **+0.0006909581**
- mean post-dot **-0.0000355251**
- mean applied anchor dot **-0.0000355251**
- mean applied rounding bound **0.0000355460**
- max runtime rounding ratio **0.4995122**
- max W rounding ratio **0**
- mean anchor **0.2013143**
- selected-epoch W norm **12.5012836**

The optimizer-step guard was active frequently and remained within frozen float-precision bounds.

## Frozen interpretation

**Case B — correctness retained but transport is not fully protected.**

S41 materially retains/exceeds the S38-style correctness effect.

It also fixes a large portion of S38's transport failure:
- fused agreement no longer collapses;
- relation agreement improves strongly;
- same-option cosine improves.

But the signature discrimination margin still falls by **0.1201088**, almost the same qualitative failure mode that mattered in S38, and treatment ends at only **0.0808327**, below the frozen **0.15** gate.

Therefore:

> Constraining actual AdamW movement against a per-signature cosine-to-reference anchor is insufficient to preserve relative option discrimination geometry.

## Closed family

Do not post-DEV tune S41 by:
- changing anchor target;
- adding anchor coefficient/slack;
- changing optimizer-state semantics;
- partial detach;
- gradient mixing;
- W-only LR/scheduler;
- seed/LR/epoch/batch retry;
- gate weakening;
- second DEV.

S41 is scientifically closed.

No confirmation.
No multilingual probe.
No Laya/Jev evaluation.
Production-ready remains false.

## Next direction

**S42 — Cross-View Relational Signature-Geometry Anchoring**

The next mechanism changes the anchor target rather than readout capacity or optimizer.

Goal:
preserve the **relative option geometry** directly, especially same-option-vs-wrong-option discrimination across canonical/paraphrase views, while retaining S41 optimizer-faithful actual-step projection.
