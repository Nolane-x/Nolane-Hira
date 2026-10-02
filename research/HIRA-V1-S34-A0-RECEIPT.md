# HIRA V1 S34 A0 receipt — Query-Conditioned Entropic Relation Transport

Status: **QUALIFIED / DIAGNOSTIC SEMANTICS ONLY**

Issue: #251
PR: #252

## Canonical authority

- run `36998495974`
- artifact `11222861913`
- artifact digest `sha256:239c2c5964a30d4ed8ca67a04e6b9ed5f4aea90efb80609368d126487fe044d9`
- authority head `de28e3f2cb51542970d07b761b85e8f36914e261`
- outcome `HIRA_V1_S34_A0_ENTROPIC_RELATION_TRANSPORT_READY`

No replacement A0 is authorized.

## Frozen transport

- state relevance temperature **0.10**
- option relevance temperature **0.10**
- state-option kernel temperature **0.10**
- relation-logit temperature **0.10**
- Sinkhorn iterations **12**
- epsilon **1e-12**
- added learned parameters **0**

Physical surface:
- attention LoRA **16,384**
- shared projection **32,768**
- total **49,152**
- A0 runtime trainable **0**
- original A13 trainable **0**
- HIRACore trainable **0**

## Controlled transport intervention

With state/options held fixed and only the semantic query changed:

- qA selected option **0**
- qB selected option **1**
- state marginal intervention max abs **0.9999091625**
- option marginal intervention max abs **0.4999546111**
- transport-plan intervention max abs **0.9998811483**

The question therefore changes both marginals and the actual many-to-many correspondence plan, and the controlled decision changes in the intended direction.

## Equivariance and masking

Errors:
- state-token permutation logits **0**
- state-token permutation signatures **0**
- option-token permutation logits **0**
- option-token permutation signatures **0**
- logical-option permutation logits **0**
- logical-option permutation signatures **0**
- masked-padding logits **0**
- masked-padding signatures **0**

Degenerate one-token / duplicate-token geometry:
- finite: PASS
- row residual **0**
- column residual **0**

## Sinkhorn marginal court

Controlled geometry:
- max row residual **5.9604645e-8**
- max column residual **5.9604645e-8**

Fresh real A13 semantic geometry:
- max row residual **4.3176115e-5**
- max column residual **1.1920929e-7**

Both pass the frozen **1e-4** tolerance.

## Real semantic gradient court

Relation block:
- relation CE **1.5186030865**
- local signature canonicalization **0.5360623598**
- total relation block **0.2322696745**

Attention LoRA-B gradient L1:
- Q **0.0422255099**
- K **0.0367128886**
- V **0.2506248951**
- attention output **1.2913545370**

Projection gradient L1:
- **46.1034240723**

All intended trainable surfaces receive finite nonzero gradients.

## Mechanics

- operator params **0**
- checkpoint LoRA keys **8**
- checkpoint roundtrip PASS
- full-K PASS
- state views expected **32**
- probability-mass error **1.7881393e-7**

## A0 semantic diagnostics

Diagnostic only:
- treatment relation accuracy **20.3125%**
- treatment same-option signature cosine **0.6724215746**

These values MUST NOT tune:
- any transport/marginal/logit temperature
- Sinkhorn iterations
- epsilon
- kernel form
- signature construction
- score aggregation
- optimizer
- selector
- DEV gates
- seed/LR/epochs/batch

## Consequence

S34-A0 is **QUALIFIED**.

Fresh matched TRAIN/DEV may open only after:
1. this receipt is frozen;
2. interpretation plan remains unchanged;
3. operator/authority/checkpoint/trainer remain unchanged;
4. TRAIN/DEV workflow is staged;
5. exact-head generic CI passes;
6. a separate one-shot TRAIN/DEV authorization marker is committed.

No second S34-A0 is authorized.
