# HIRA V1 S33 A0 receipt — Query-Explicit Relation Coordinates

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #249
PR: #250

## Canonical authority

- run `36988606786`
- artifact `11219055373`
- artifact digest `sha256:99c4be1b4e2badc4aa5f6591d8c044377f62f8a958d7c51a0bee241a803596ea`
- authority head `5569fb670e72cfae42517875b95e0ea29522ed73`
- outcome `HIRA_V1_S33_A0_QUERY_EXPLICIT_RELATION_READY`

No replacement A0 is authorized.

## Exact inherited S13 identity

The treatment preserves the complete inherited base relation path exactly:

- base relation logits max abs: **0**
- base relation signatures max abs: **0**

Query-anchor implementation:
- reference formula max abs: **0**

Dimensions:
- base signature: **128**
- query-option block: **128**
- treatment signature: **256**

Frozen score weights:
- base S13 score: **0.50**
- query-option score: **0.50**

Added learned parameters:
- **0**

## Exact physical surface

- attention LoRA: **16,384**
- shared projection: **32,768**
- physical trainable surface: **49,152**
- frozen A0 runtime trainable params: **0**
- original A13 trainable params: **0**
- HIRACore trainable params: **0**

Checkpoint:
- LoRA key count: **8**
- roundtrip: PASS

## Controlled query-explicit intervention

With state/options held fixed:

- question-anchor intervention max abs: **1.0**
- query-option signature intervention max abs: **0.7071068287**
- query-option logits intervention max abs: **10.0**

Controlled expected-direction geometry:
- qA option0 - option1: **+10.0**
- qB option1 - option0: **+10.0**

Therefore the explicit query coordinate is not decorative; changing the question directly moves both the query-option representation and scoring term in the intended direction.

## Option/mechanics court

- option-permutation logit error: **0**
- option-permutation signature error: **0**
- treatment probability-mass error: **1.1920929e-7**
- full-K: PASS
- expected state views: **32**
- checkpoint roundtrip: PASS

## Real semantic gradient court

Relation block:
- relation CE: **1.4451451302**
- local signature canonicalization: **0.4694456756**
- total relation block: **0.2149313688**

Attention LoRA-B gradient L1:
- Q: **0.0209561940**
- K: **0.0252935886**
- V: **0.2000730783**
- attention output: **1.3588542938**

Projection gradient L1:
- **43.4579086304**

All intended trainable surfaces receive finite nonzero gradients.

## A0 semantic diagnostics

Diagnostic only; not model-selection authority:

- treatment relation accuracy: **28.125%**
- treatment same-option signature cosine: **0.7167657614**

These values MUST NOT tune:
- 0.50 / 0.50 score weights
- query pooling
- query-option residual formula
- signature block weighting
- role/pair/contrastive temperatures
- optimizer
- selector
- DEV gates
- seed / LR / epochs / batch

## Consequence

S33-A0 is **QUALIFIED**.

Fresh matched S33 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. interpretation plan remains unchanged;
3. query-explicit operator, authority, checkpoint and trainer remain unchanged;
4. TRAIN/DEV workflow remains staged;
5. exact-head generic CI passes after this receipt;
6. a separate one-shot TRAIN/DEV authorization marker is committed.

No second S33-A0 run is authorized.
