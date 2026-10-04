# HIRA V1 S54 contract — Joint State–Query–Option Late Interaction

Status: **PREREGISTERED / NO S54-A0 EXPOSURE**

Issue: #291

Parent:
- S53 fresh run `37205794717`
- artifact `11304553484`
- Case **C**
- token-level query↔option context preserved correctness but did not recover stability.

## Scientific question

> With native evidence and private capacity fixed, does a direct joint state–query–option late interaction improve cross-view decision stability while preserving useful correctness versus S53 query↔option late interaction?

## Frozen backbone

S54 inherits:
- exact persisted S51 native authority;
- immutable shared native evidence;
- query-free state↔option identity;
- correction A/B/W surface;
- one encoder/state-once;
- full-K private readout.

Native retraining is forbidden.

## Reference

Exact S53 treatment operator:
`option_conditioned_token_level_query_context`

For option k:
`score_t(k) = <q_t, option_identity_k> / 0.10`

Query weights are option-conditioned only.

## Treatment

Parameter-free joint triadic late interaction.

For normalized cached tokens:

`a_t^state = max_s <q_t, s_s>`

`a_t^option(k) = max_o <q_t, o_{k,o}>`

`joint_t(k) = softmax_t[(a_t^state + a_t^option(k))/0.10]`

`c_k = normalize(sum_t joint_t(k) q_t)`

Only active masked state/query/option tokens participate.

The treatment therefore retains a query token only to the extent that it is jointly supported by:
- current state evidence; and
- the candidate option evidence.

## Matched private surface

Reference and treatment:
- interaction trainable params **0**
- correction params **114,688**
- query-free identity params **0**
- total private trainable **114,688**
- bit-identical correction initialization
- same native/cache bytes
- same TRAIN rows/order
- same AdamW/LR/weight decay/grad clip
- same correctness objective
- same checkpoint selector
- same private epochs
- temperature **0.10**
- one DEV only.

Controlled variable only:
- reference query context uses S53 query↔option late interaction;
- treatment query context uses joint state+query+option late interaction.

## Ownership

Joint interaction operates only on detached cached tensors.

It has:
- zero trainable parameters;
- zero native gradient path;
- zero cache gradient path;
- no second encoder.

Correction A/B/W remain the only private trainable parameters.

## Required S54-A0

Capacity:
- reference correction 114,688;
- treatment correction 114,688;
- interaction params 0;
- identity params 0;
- bit-identical correction initialization.

Invariants:
- K=3/7/255;
- option permutation equivariance;
- state-token permutation invariance;
- query-token permutation invariance;
- option-token permutation invariance;
- option-view permutation invariance;
- state/query/option padding-mask invariance;
- all-masked state rejection;
- all-masked query rejection;
- all-masked active option rejection;
- full-K probability mass <=1e-6;
- context norm error <=1e-6;
- deterministic replay;
- no pooled-query bypass;
- no S53 query↔option-only bypass in treatment.

Path sensitivity with activated correction B/W:
- informative state-token perturbation changes treatment context and logits;
- informative query-token perturbation changes treatment context and logits;
- informative option-token perturbation changes treatment context and logits.

Runtime:
- one encoder/state-once;
- native/cache gradients 0.

## Fresh S54 authority

Intended:
- seed **75001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S54 domains
- K=4
- private epochs 24
- batch 16
- one DEV only.

Reuse exact sealed S51 native authority:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`.

## Prohibited

No:
- temperature sweep
- max/mean/logsumexp sweep
- learned projection
- native retraining
- identity change
- correction capacity change
- selector change
- retry
- gate weakening
- second S54 DEV.

Scientific failure is valid.
