# HIRA V1 S35 A0 receipt — Native A13 Relation Geometry

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #253
PR: #254

## Canonical authority

- run `37008578553`
- artifact `11227286171`
- artifact digest `sha256:a32475c6b9278ea5feada4f4c002188089b1a2add17ae732f3a4b6ae1fa16a47`
- authority head `72dca7b0a464d6083421d6a8c297740070a956ba`
- outcome `HIRA_V1_S35_A0_NATIVE_RELATION_GEOMETRY_READY`

No replacement A0 is authorized.

## Exact geometry isolation

Treatment:
- native relation dimension **256**
- role temperature **0.10**
- pair temperature **0.10**
- contrastive temperature **0.10**
- added learned params **0**

Physical surface:
- attention LoRA **16,384**
- shared primary projection **32,768**
- total **49,152**
- frozen A0 runtime trainable **0**
- original A13 trainable **0**
- HIRACore trainable **0**

Signatures:
- projected S13 control **128D**
- native treatment **256D**

## S13-equivalence court

Under an exact 256D identity projection:
- native vs S13 logit max abs **0**
- native vs S13 signature max abs **0**

Primary control/treatment identity:
- primary logit max abs **0**

## Projection-dependence discriminator

After a strong shared-projection perturbation:

Control relation:
- logit max abs change **2.7219405174**
- signature max abs change **0.2809848785**

Native treatment relation:
- logit max abs change **0**
- signature max abs change **0**

Primary path:
- logit max abs change **0.0016246404**

This proves the treatment relation path genuinely bypasses the shared projection while the primary path still depends on it.

## Gradient ownership

Treatment relation block -> shared projection:
- gradient L1 **0**

Treatment relation block -> A13 LoRA:
- gradient L1 **0.5091757309**

Primary block -> shared projection:
- gradient L1 **1024.1845703125**

This is the exact preregistered ownership split.

## Invariance / mechanics

- state-token permutation logit error **1.7881393e-7**
- state-token permutation signature error **7.4505806e-8**
- question-token permutation errors **0 / 0**
- option-token permutation logit error **2.2351742e-8**
- option-token permutation signature error **2.9802322e-8**
- logical-option permutation errors **0 / 0**
- masked-padding logit error **0**
- masked-padding signature error **2.9802322e-8**
- degenerate geometry finite
- probability-mass error **1.1920929e-7**
- full-K PASS
- expected state views **32**
- checkpoint key count **8**
- checkpoint roundtrip PASS

## A0 semantic diagnostics

Diagnostic only:
- native relation accuracy **31.25%**
- relation CE **1.3721621037**
- canonicalization **0.5093940496**
- relation block **0.2136253119**

These values MUST NOT tune:
- native dimension
- temperatures
- normalization
- projected/native mixing
- optimizer
- selector
- DEV gates
- seed / LR / epochs / batch

## Harness correction before DEV

After A0, the staged TRAIN/DEV script was found to contain a pre-DEV freshness-harness bug:
- self-import of `_assert_s35_fresh`
- self-recursive freshness call
- S35 rows mistakenly used as prior-S34 rows

No DEV run had been authorized or exposed.

Harness-only correction:
- use `_assert_s34_fresh`
- compare against `generate_s34_cases`
- retain all frozen S35 science settings unchanged

Correction commit:
`cc89718b37656ebc26ecec0ad99cac67bb5befba`

This does not alter the S35 hypothesis, operator, A0 evidence, selector, gates, optimizer, or fresh authority.

## Consequence

S35-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. trainer freshness correction is CI-qualified;
3. interpretation plan remains unchanged;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. a separate one-shot TRAIN/DEV marker is committed.

No second S35-A0 run is authorized.
