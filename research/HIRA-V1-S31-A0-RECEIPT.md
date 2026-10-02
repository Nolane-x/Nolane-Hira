# HIRA V1 S31 A0 receipt — Global Cross-Case Relation Contrastive

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #245
PR: #246

## Canonical authority

- run `36970587382`
- artifact `11211836648`
- artifact digest `sha256:6520619fbcb1fb74464770848022d8658ac234f909451ed2338e819be3c12862`
- authority head `422b81f96ae0cd0c1affa547521842b7ce0e650c`
- outcome `HIRA_V1_S31_A0_GLOBAL_RELATION_CONTRASTIVE_READY`

Earlier run `36969870003` is a documented harness-only non-result:
- no artifact
- no qualified receipt
- no scientific outcome
- science settings unchanged

No additional S31-A0 is authorized.

## Exact control/treatment inference identity

Before training, local control and global treatment are exact identities:

- token output max abs **0**
- pooled output max abs **0**
- raw canonical/paraphrase logits max abs **0 / 0**
- relation canonical/paraphrase logits max abs **0 / 0**
- fused canonical/paraphrase logits max abs **0 / 0**
- signature canonical/paraphrase max abs **0 / 0**
- exact logit identity rate **1.0**
- exact selected-choice identity rate **1.0**

The S31 intervention is training-only.

## Exact physical surface

Both arms:
- final-attention LoRA **16,384**
- shared projection **32,768**
- exact physical trainable surface **49,152**
- frozen A0 runtime trainable **0**
- original A13 trainable **0**
- HIRACore trainable **0**

Checkpoint:
- LoRA key count **8**
- exact roundtrip PASS

Operator:
- learned parameter count **0**
- temperature **0.10**
- outer coefficient **0.15**

## Controlled operator court

Global cross-case relation contrastive:

- matched positive loss **0.0003177615**
- shuffled-positive loss **2.0794413090**
- shuffled - matched **+2.0791234970**
- matched query permutation error **0**
- canonical/paraphrase view-swap error **0**
- canonical-signature gradient L1 **9.3638286591**
- paraphrase-signature gradient L1 **8.6360034943**

Therefore:
- matched positives are strongly preferred;
- cross-query negatives are active;
- the operator is matched-permutation invariant;
- view symmetry holds;
- gradients reach both wording views;
- no learned state is added.

## Real S31-A0 semantic gradient court

- attention LoRA-B gradient L1:
  - Q **0.2731080353**
  - K **0.2124030143**
  - V **2.4728033543**
  - attention output **18.0734901428**
- projection gradient L1 **518.0814819336**
- primary gradient norm **8.1179714203**
- relation gradient norm **1.3129460812**
- normalized pre-dot **+0.0863989070**
- normalized post-dot **+0.0863989070**
- projection coefficient **0**
- combined norm **4.7154588699**

The treatment surface is live and optimization is finite.

## Mechanics

- full-K PASS
- state encode calls **32**
- option-order flip **0**
- probability-mass error **1.1920929e-7**
- relation refinement disabled
- checkpoint roundtrip PASS

## A0 semantic diagnostics

Diagnostic only; not model-selection authority:

- fused canonical **34.375%**
- fused paraphrase **37.5%**
- fused agreement **71.875%**
- fused JS **0.0285592377**
- relation canonical **37.5%**
- relation paraphrase **37.5%**
- relation agreement **56.25%**
- signature cosine **0.5783756375**
- signature discrimination **-0.0339841619**
- real semantic global contrastive loss **2.9082093239**

These values MUST NOT tune:
- temperature
- outer coefficient
- negative bank
- batch size
- loss composition
- optimizer
- selector
- gates
- seed or schedule

## Consequence

S31-A0 is **QUALIFIED**.

Fresh matched S31 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. the interpretation plan remains unchanged;
3. authority/trainer/operator remain unchanged;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. a separate one-shot TRAIN/DEV authorization marker is committed.

No second S31 DEV run is authorized.
