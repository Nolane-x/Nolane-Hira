# HIRA V1 S30 A0 receipt — Matched Attention-vs-FFN Adaptation Surface Court

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #243
PR: #244

## Canonical authority

- run `36940968046`
- artifact `11199314198`
- artifact digest `sha256:2ec456fc02ca498729358d21cc7a8205aa8807230514902ac73f2ddaaebad013`
- authority head `04100aad4158170bfcea8c0dd1b41a7a529f58c8`
- outcome `HIRA_V1_S30_A0_MATCHED_ADAPTATION_READY`

No replacement A0 is authorized.

## Exact zero-init arm identity

Attention-only and FFN-only are exact inference identities at A0:

- token output max abs **0**
- pooled output max abs **0**
- raw canonical/paraphrase logits max abs **0 / 0**
- relation canonical/paraphrase logits max abs **0 / 0**
- fused canonical/paraphrase logits max abs **0 / 0**
- relation signatures max abs **0 / 0**
- exact logit identity rate **1.0**
- exact selected-choice identity rate **1.0**

Thus S30 changes only optimization surface after training begins.

## Arm A — attention-only

Capacity:
- LoRA **16,384**
- projection **32,768**
- physical total **49,152**
- frozen A0 runtime trainable **0**
- original A13 trainable **0**
- HIRACore trainable **0**

Checkpoint:
- LoRA keys **8**
- exact roundtrip PASS

Real combined-gradient L1 by attention LoRA B:
- Q **0.2424240708**
- K **0.3624919653**
- V **2.2540881634**
- attention output **16.6883277893**

Projection gradient L1:
- **657.9580688477**

Gradient partition:
- primary norm **11.0922117233**
- relation norm **0.5571977496**
- normalized pre-dot **-0.0003226218**
- normalized post-dot approximately **0**
- projection coefficient **-0.0003226218**
- combined norm **5.8247046471**

## Arm B — FFN-only

Capacity:
- LoRA **20,480**
- projection **32,768**
- physical total **53,248**
- frozen A0 runtime trainable **0**
- original A13 trainable **0**
- HIRACore trainable **0**

Checkpoint:
- LoRA keys **4**
- exact roundtrip PASS

Real combined-gradient L1 by FFN LoRA B:
- intermediate.dense **3.6626973152**
- output.dense **1.9171190262**

Projection gradient L1:
- **659.6528320312**

Gradient partition:
- primary norm **11.0787248611**
- relation norm **0.5534449816**
- normalized pre-dot **+0.0006424960**
- normalized post-dot **+0.0006424960**
- projection coefficient **0**
- combined norm **5.8160843849**

## Mechanics

Both arms:
- full-K PASS
- state encode calls **32**
- option-order flip **0**
- probability-mass error **1.1920929e-7**
- relation refinement disabled
- fusion expert-swap exact

## A0 semantic diagnostics

A0 semantics are identical across arms and diagnostic only:

- fused canonical **31.25%**
- fused paraphrase **40.625%**
- fused agreement **78.125%**
- fused JS **0.0170529112**
- relation canonical **37.5%**
- relation paraphrase **40.625%**
- relation agreement **71.875%**
- signature cosine **0.5655265450**
- signature discrimination **-0.0133778509**

These values MUST NOT tune either S30 arm.

## Consequence

S30-A0 is **QUALIFIED**.

A fresh matched TRAIN/DEV court may open only after:
1. this receipt is frozen;
2. the frozen S30 interpretation plan remains unchanged;
3. the matched trainer/authority/checkpoint stack remains unchanged;
4. exact-head generic CI passes;
5. a separate one-shot TRAIN/DEV authorization marker is committed.

No second S30-A0 run is authorized.
