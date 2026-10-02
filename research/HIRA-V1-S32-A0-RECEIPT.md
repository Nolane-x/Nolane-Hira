# HIRA V1 S32 A0 receipt — Global Query×Option Relation Matrix Canonicalization

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #247
PR: #248

## Canonical authority

- run `36978637856`
- artifact `11214148806`
- artifact digest `sha256:19cab638c1dbb73b382aae669f42b3ab254ddb993145810dc26b3e12a61942a0`
- authority head `47be09c51dcd5b5d1307d911171266ca229c59f9`
- outcome `HIRA_V1_S32_A0_GLOBAL_QUERY_OPTION_MATRIX_READY`

No replacement A0 is authorized.

## Exact inference identity

Gold-only control and all-query×option treatment are exact inference identities before training:

- token output max abs **0**
- pooled output max abs **0**
- raw canonical/paraphrase logits max abs **0 / 0**
- relation canonical/paraphrase logits max abs **0 / 0**
- fused canonical/paraphrase logits max abs **0 / 0**
- relation signatures canonical/paraphrase max abs **0 / 0**
- exact decision-logit identity rate **1.0**
- exact selected-choice identity rate **1.0**

The S32 intervention is training-only.

## Exact physical surface

Both arms:
- final attention LoRA **16,384**
- shared projection **32,768**
- exact physical trainable surface **49,152**
- frozen A0 runtime trainable **0**
- original A13 trainable **0**
- HIRACore trainable **0**

Operator:
- learned parameters **0**
- temperature **0.10**
- outer coefficient **0.15**

Checkpoint:
- LoRA key count **8**
- exact roundtrip PASS

## Controlled non-gold discriminator

Controlled matched all-option baseline:
- all-option good loss **0.0006808108**

After perturbing exactly one **non-gold** query-option relation:
- all-option loss **0.4214525521**
- all-option increase **+0.4207717478**

Gold-only control on the same perturbation:
- gold-only good loss **0.0001362469**
- gold-only perturbed loss **0.0001362469**
- exact change **0**

Therefore the S32 treatment observes a relation that the S31 gold-only operator provably cannot observe.

## Controlled gradient discriminator

Non-gold gradients under S31 gold-only:
- canonical **0**
- paraphrase **0**

Non-gold gradients under S32 all-option:
- canonical L1 **6.7918653488**
- paraphrase L1 **6.5863475800**

Both treatment gradients are finite and nonzero.

## Symmetry/equivariance

- matched query+option permutation error **0**
- canonical/paraphrase view-swap error **0**

The treatment does not depend on arbitrary ordering or view direction.

## Real semantic gradient court

On fresh S32-A0 text rows:

Attention LoRA-B gradient L1:
- Q **0.2595716119**
- K **0.2727566063**
- V **2.6033835411**
- attention output **13.6473093033**

Projection gradient L1:
- **757.2583007812**

S17 primary/relation partition:
- primary norm **12.1072807312**
- relation norm **1.1059654951**
- normalized pre-dot **+0.0065013086**
- normalized post-dot **+0.0065013086**
- projection coefficient **0**
- combined norm **6.6066226959**

The treatment reaches the intended real semantic surfaces.

## Mechanics

- full-K PASS
- state encode calls **32**
- option-order flip **0**
- probability-mass error **1.1920929e-7**
- relation refinement disabled
- fusion expert-swap max abs **0**
- checkpoint roundtrip PASS

## A0 semantic diagnostics

Diagnostic only; not model-selection authority:

- fused canonical **18.75%**
- fused paraphrase **46.875%**
- fused agreement **56.25%**
- fused JS **0.0423501022**
- relation canonical **21.875%**
- relation paraphrase **31.25%**
- relation agreement **65.625%**
- signature cosine **0.5923287272**
- signature discrimination **-0.0201123264**
- real semantic all-option contrastive loss **4.1219387054**

These values MUST NOT tune:
- temperature
- outer coefficient
- batch size
- negative set
- loss mixture
- optimizer
- seed
- selector
- gates
- training duration

## Consequence

S32-A0 is **QUALIFIED**.

Fresh matched S32 TRAIN/DEV may open only after:
1. this receipt is frozen;
2. interpretation/contract remain unchanged;
3. operator/trainer/authority/checkpoint replay remain unchanged;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. a separate one-shot TRAIN/DEV authorization marker is committed.

No second S32 DEV run is authorized.
