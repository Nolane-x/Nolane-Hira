# HIRA V1 S43 A0 receipt — Cross-View Relative-Gap Signature Geometry Anchoring

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #269
PR: #270

## Canonical authority

- run `37112131898`
- artifact `11269189976`
- digest `sha256:8116aec424a7852a0749a0880f66ecdcfe633ee4c1bb6b570f322f409e338429`
- authority head `ff45888f822f8b6b06218e4758b15d1d57087462`
- outcome `HIRA_V1_S43_A0_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_READY`

No replacement S43-A0 is authorized after this qualified receipt.

## Capacity / zero-init identity

- W **65,536**
- reference trainable **49,152**
- treatment trainable **114,688**
- zero-init relation/signature/fused differences **0**
- selected-choice identity **1.0**

## Relative-gap court

Anchor:
`A_gap = mean((R_treatment - stopgrad(R_reference))^2)`

with:
`R[b,i,j] = G[b,i,i] - G[b,i,j]`, `j != i`.

Evidence:
- geometry shape **[3,4,4]**
- off-diagonal gap count **36**
- identical geometry loss **0**
- same-option diagonal perturbation loss **0.00390625**
- wrong-option perturbation loss **0.0013020834**
- shared option-permutation scalar error **0**

Equal-full-MSE contrast:
- absolute-shift full MSE **0.0133333337**
- gap-distortion full MSE **0.0133333337**
- absolute-shift gap loss **0**
- distorted gap loss **0.0266666692**

Therefore S43 directly distinguishes relative discrimination distortion that S42 full-matrix MSE treats as equally large absolute geometry movement.

## Anchor ownership / liveness

Synthetic signature drift max abs:
**0.0038859472**

Gap anchor:
**1.6543487e-7**

Gradients:
- anchor -> treatment runtime L1 **0.0006695528**
- anchor -> treatment LoRA L1 **0.0004509039**
- anchor -> reference runtime **0**
- anchor -> W **0**

Correctness:
- W gradient L1 **442.6993713**
- off-diagonal W gradient L1 **440.8872375**
- treatment LoRA correctness gradient L1 **225.4208136**

Both relative-gap constraint and correctness co-adaptation are live.

## Exact AdamW parity

- candidate movement error **0**
- first-moment error **0**
- second-moment error **0**
- step-counter error **0**
- reconstruction-only float rounding **1.4551915e-11**

## Actual-step gap projection

Live candidate:
- runtime anchor dot **+3.5904023e-8**
- projected **true**
- W candidate delta L1 **13.1052923**

Synthetic conflict:
- pre-dot **+1.3387653e-7**
- projected **true**
- post-dot **1.0292496e-9**

Safe actual step:
- projected **false**
- identity max abs **0**

W projection max abs:
**0**

## Float-representable applied movement

Runtime:
- movement error **3.6234269e-9**
- rounding bound **1.4901190e-8**

W:
- movement error **0**
- rounding bound **2.8025969e-45**

Actual applied gap-anchor dot:
- **1.0292496e-9**
- permitted rounding bound **1.0294896e-9**

Thus the representable movement remains inside the frozen precision guard.

## Mechanics

- arbitrary K=3/K=7 PASS
- option permutation logit/signature error **5.9604645e-8 / 0**
- question permutation logit/signature error **0 / 0**
- padding logit/signature error **0 / 0**
- native projection perturbation relation/signature **0 / 0**
- primary projection perturbation **0.0017167892**
- checkpoint roundtrip PASS
- probability mass error **1.1920929e-7**
- full-K PASS
- state-once expected views **32**

## Diagnostic semantic values

- relation accuracy **0.28125**
- fused accuracy **0.25**
- used for model selection **false**

These values MUST NOT tune:
- gap definition
- gap weighting
- strongest-wrong variant
- target margin
- norm
- projection epsilon
- optimizer semantics
- W capacity
- selector/gates
- fresh TRAIN/DEV settings.

## Consequence

S43-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. fresh S43 authority is frozen;
3. matched trainer/workflow are frozen;
4. exact staged-head generic CI PASS;
5. separate one-shot TRAIN/DEV marker.

No external Laya/Jev benchmark is authorized from A0.
