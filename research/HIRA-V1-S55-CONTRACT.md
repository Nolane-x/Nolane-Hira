# HIRA V1 S55 contract — Learned Joint Relation Interaction

Status: **PREREGISTERED / NO S55-A0 EXPOSURE**

Issue: #293

Parent:
- S54 fresh run `37209555118`
- artifact `11305929426`
- digest `sha256:1b9787fe775493e53d7f7f0108295914cd974bfd8da9f13a8f03eb02f7109ae2`
- Case **C**
- parameter-free triadic context improved correctness but did not improve selected-choice stability.

## Scientific question

> With native evidence, immutable cache, option identity, correction capacity and learned capacity matched, can a compact learned joint relation transform convert stable triadic evidence into more stable selected choices without sacrificing useful correctness?

## Frozen backbone

Reuse:
- sealed S51 persisted native authority;
- immutable shared cache;
- native trainability 0;
- S51 query-free option identity;
- S53 option-conditioned query↔option context;
- S54 parameter-free joint state-query-option context;
- one encoder/state-once;
- full-K.

No native retraining.

## Learned joint transform

Both arms contain exactly the same learned transform.

For each option k:

- `q_k`: normalized S53 option-conditioned query↔option context, 256D
- `j_k`: state-conditioned S54 joint state-query-option context, 256D for treatment; all-zero neutral channel for reference
- `o_k`: normalized query-free option identity, 256D

Concatenate:
`x_k = [q_k ; j_k ; o_k]` -> 768D

Residual bottleneck:
- A: `64 x 768` = **49,152**
- B: `256 x 64` = **16,384**
- no bias
- total learned joint params **65,536**
- seed **75555**
- A Gaussian std **0.02**
- B exactly zero initialized.

Learned relation code:
`r_k = normalize(q_k + B(gelu(A x_k)))`

At initialization, B=0 means `r_k == q_k` up to normalization tolerance.

## Matched private surface

Both arms:
- correction params **114,688**
- learned joint params **65,536**
- identity params **0**
- interaction params in S53/S54 **0**
- total private trainable **180,224**
- bit-identical correction initialization
- bit-identical learned-transform initialization
- same native/cache bytes
- same TRAIN order
- same optimizer/LR/weight decay/grad clip
- same CE + JS objective
- same checkpoint selector
- same epochs/batch.

## Controlled variable

Reference:
- explicit state-conditioned joint channel into the learned transform is **all zeros**
- S53 query↔option context remains live
- option identity remains live.

Treatment:
- explicit state-conditioned joint channel is the S54 joint state-query-option context.

The only controlled variable is whether real S54 joint evidence enters the matched learned transform.

## No hidden bypass

The S54 joint context MUST NOT bypass the learned transform into correction logits.

The learned relation code `r_k` is the only query/relation context passed into the correction shell in both arms.

Raw pooled query MUST NOT bypass S53/S54/learned transform.

Reference and treatment both remain state-dependent through the shared query-free option identity. Therefore controlled-variable probes must hold q_k and o_k fixed and perturb only the explicit joint channel when testing state-channel isolation.

## Required S55-A0

Architecture:
- learned joint params exactly 65,536 each;
- correction params exactly 114,688 each;
- total private trainable exactly 180,224 each;
- identity params 0;
- correction initialization bit-identical;
- learned transform initialization bit-identical;
- zero-init warm start preserves q_k.

Mechanics:
- K=3/7/255
- option permutation equivariance
- state-token permutation invariance in treatment
- query-token permutation invariance
- option token/view permutation invariance
- mask/padding invariance
- all-masked rejection
- full-K probability mass <=1e-6
- context/relation-code norm error <=1e-6
- no second encoder
- native/cache gradients 0
- learned transform gradients live
- correction gradients live.

Controlled-variable probes:
- with q_k/o_k held fixed, perturbing explicit joint channel changes treatment relation code/logits;
- with q_k/o_k held fixed, perturbing the would-be joint channel does not change reference relation code/logits because reference receives zeros;
- query evidence perturbation changes both arms;
- option evidence perturbation changes both arms;
- no direct S54 context bypass;
- deterministic replay.

## Fresh S55 authority

Intended:
- seed **76001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S55 domains
- K=4
- private epochs **24**
- one DEV only.

Reuse exact sealed S51 native authority:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`.

## Prohibited

No:
- hidden-dimension sweep
- transform-seed sweep
- neutral-channel variant
- state-channel scale sweep
- direct S54-context bypass
- native retraining
- identity/correction capacity change
- alternate selector
- retry
- gate weakening
- second S55 DEV
- external Laya/Jev evaluation.

Scientific failure is valid.
