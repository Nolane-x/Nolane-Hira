# HIRA V1 S29 A0 receipt — Final-Block Attention+FFN LoRA Semantic Adaptation

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #241
PR: #242

## Canonical authority

- run `36883766054`
- artifact `11172806806`
- artifact digest `sha256:1f05cab39d88b481968d27c7260b9a7192f184bf650863c5fc7874c2d2f53f9f`
- authority head `ed3d7b9c64e6a3bf75fa62f577901c41b51dfcdd`
- outcome `HIRA_V1_S29_A0_FULL_BLOCK_LORA_READY`

No replacement A0 is authorized.

## Exact S17 initialization identity

S29 zero-init FFN extension is a true no-op at A0:

- A13 token output max abs: **0**
- A13 pooled output max abs: **0**
- raw canonical logits max abs: **0**
- raw paraphrase logits max abs: **0**
- relation canonical logits max abs: **0**
- relation paraphrase logits max abs: **0**
- fused canonical logits max abs: **0**
- fused paraphrase logits max abs: **0**
- canonical signature max abs: **0**
- paraphrase signature max abs: **0**
- exact decision-logit identity rate: **1.0**
- exact selected-choice identity rate: **1.0**

S29 therefore changes no S17 inference behavior before training.

## Exact trainable surface

Frozen physical surface:
- attention LoRA: **16,384**
- FFN intermediate LoRA: **10,240**
- FFN output LoRA: **10,240**
- FFN LoRA subtotal: **20,480**
- total A13 LoRA: **36,864**
- shared projection: **32,768**
- exact physical trainable surface: **69,632**

Frozen A0 runtime:
- runtime trainable params: **0**
- original A13 trainable params: **0**
- HIRACore trainable params: **0**

Six-module LoRA parameter layout:
- **[4096, 4096, 4096, 4096, 10240, 10240]**

Checkpoint:
- key count: **12**
- exact roundtrip: PASS

## Real gradient court

The new FFN surface is active on fresh S29-A0 text rows:

Attention LoRA B gradient L1:
- Q: **0.1920543015**
- K: **0.2026104927**
- V: **1.6929421425**
- attention output: **8.3233375549**

FFN:
- intermediate LoRA B gradient L1: **2.3468730450**
- output LoRA B gradient L1: **0.9416253567**

Shared projection:
- gradient L1: **358.0661621094**

S17 primary/relation partition:
- primary block gradient norm: **5.5852651596**
- relation block gradient norm: **0.6249681711**
- normalized pre-dot: **+0.0042904611**
- normalized post-dot: **+0.0042904611**
- projection coefficient: **0**
- combined norm: **3.1051163673**

The FFN adapters are therefore not decorative/dead parameters.

## Mechanics

- full-K: PASS
- state encode calls: **32** for 16 cases × 2 state views
- primary option-order flip: **0**
- fused option-order flip: **0**
- probability-mass error: **1.1920929e-7**
- fusion expert-swap max abs: **0**
- relation refinement: disabled

## A0 semantic diagnostics

Diagnostic only; not model-selection or tuning authority:

- raw primary canonical accuracy: **21.875%**
- relation canonical accuracy: **28.125%**
- relation paraphrase accuracy: **18.75%**
- fused canonical accuracy: **31.25%**
- fused paraphrase accuracy: **31.25%**
- paired both-correct: **0**
- fused cross-view agreement: **59.375%**
- fused JS: **0.0434342921**
- fused canonical margin: **-0.7812182903**
- same-option signature cosine: **0.4156530797**
- signature discrimination margin: **-0.0432921834**

These values MUST NOT tune rank, alpha, dropout, LoRA coverage, losses, optimizer, selector, seed, DEV gates or any other S29 setting.

## Consequence

S29-A0 is **QUALIFIED**.

Fresh S29 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. the preregistered interpretation plan remains unchanged;
3. fresh S29 authority/checkpoint/train script remain frozen;
4. TRAIN/DEV workflow remains staged;
5. exact-head generic CI passes after this receipt;
6. a separate one-shot TRAIN/DEV authorization marker is committed.

No additional S29-A0 run is authorized.
