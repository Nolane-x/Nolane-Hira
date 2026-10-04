# HIRA V1 S54 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #291  
PR: #292

## Parent

S53 merged main:
`6e4a61c7740e64f7f62d2fd0963fd48ad5913482`

S53 fresh scientific court:
- run `37205794717`
- artifact `11304553484`
- digest `sha256:3ab2f778eef687e3cf8c3be459ae75facb14f964b3a5345d5aa7ad1b8bb7c962`
- interpretation **Case C**

Key parent evidence:
- token-level query↔option context cross-view cosine ~0.978
- fused canonical delta 0.00 pp
- fused paraphrase +0.52 pp
- fused agreement -5.47 pp
- relation agreement -10.68 pp.

## Frozen S54 variable

Reference:
- exact S53 option-conditioned query↔option late interaction.

Treatment:
- zero-parameter joint state+query+option late interaction.

For query token t and option k:

`state_support_t = max_s <q_t,s_s>`

`option_support_t(k) = max_o <q_t,o_{k,o}>`

`weight_t(k) = softmax_t[(state_support_t + option_support_t(k))/0.10]`

`context_k = normalize(sum_t weight_t(k) q_t)`

No learned interaction parameter.

## Matched surface

Both arms:
- correction params **114,688**
- identity params **0**
- interaction trainable params **0**
- private trainable params **114,688**
- bit-identical correction initialization
- temperature **0.10**
- same cache/optimizer/loss/selector.

## Seed correction before exposure

S53 already used seed **74001**.

S54 fresh authority is therefore frozen at **75001** before any S54-A0/DEV exposure.

No scientific output existed before this correction.

## Parent native authority

Reuse exact sealed S51 artifact:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params **0**
- native retraining forbidden.

## A0 staged

- `src/nmd/v1_joint_state_query_option_interaction.py`
- `tests/test_v1_joint_state_query_option_interaction.py`
- `scripts/hira_v1_s54_a0_joint_state_query_option_interaction.py`
- `tests/test_v1_s54_a0_harness.py`
- `.github/workflows/hira-v1-s54-a0-joint-state-query-option-interaction.yml`

A0 courts:
- K=3/7/255
- full-K probability mass
- context normalization
- state/query/option mask + permutation invariants
- option permutation equivariance
- all-masked rejection
- deterministic replay
- cache detach / zero cache gradients
- exact correction capacity/init matching
- treatment bypasses inherited S53 query↔option-only context
- activated state/query/option perturbations each change context + logits.

A0 is mechanical only:
- no model selection
- no fresh S54 TRAIN/DEV
- no external Laya/Jev evaluation.

## Authorization rule

Marker:
`research/HIRA-V1-S54-ENABLE-A0`

The marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.
