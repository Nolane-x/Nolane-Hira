# HIRA V1 S59 contract — Explicit Learned Pairwise Decision Head

Status: **PREREGISTERED / NO S59-A0 OR DEV EXPOSURE**

Issue: #302

## Parent evidence

S58 closes as **Case B**.

Fresh S58 court:
- run `37276076841`
- artifact `11330721014`
- artifact digest `sha256:cea8203e98899feda6c1ca1558b9941a8900b987124dfb43958c90e5d0285e50`
- merged main `decc5a744f69e78e83bc57bf28c3283d230f9283`.

S58 treatment minus reference:
- fused agreement **+3.65 pp**
- fused JS **-0.010048**
- fused canonical **-6.25 pp**
- paired both-correct **-6.77 pp**
- question-swap **-14.06 pp**
- canonical relation **-11.46 pp**.

Conclusion: replacing self-anchors with a frozen teacher is not enough. S59 must learn pairwise decisions directly from gold-supervised TRAIN data.

## Scientific question

> Can a learned anti-symmetric pairwise decision surface improve stable full-K choice without inheriting a student/teacher boundary and without perturbing the existing correction trajectory?

## Frozen base

Retain:
- S51 persisted native authority;
- immutable encoded cache;
- one encoder/state-once;
- `JointStateQueryOptionPrivateCorrectionFork`;
- correction params **114,688 / arm**;
- existing base private objective;
- existing selector;
- no teacher;
- no pseudo-target;
- no self-anchor.

## Detached pairwise representation

Per option:

`r_i = normalize(concat(identity_i, joint_context_i))`

Dimensions:
- identity: 256
- joint context: 256
- pairwise representation: **512**.

Both sources are zero-parameter geometry over immutable cache tensors.
The representation is explicitly detached before entering S59 head.

Forbidden pairwise-head inputs:
- native logits
- triadic logits
- fused logits
- corrected relation logits
- S57/S58 teacher outputs.

## Explicit anti-symmetric head

Frozen:
- representation dimension **512**
- hidden width **64**
- no bias
- head seed **80059**.

Parameters:
- `A: [64,512]` = 32,768
- `u: [64]` = 64
- total **32,832**.

For i,j:

`d_ij = r_i-r_j`

`raw_ij = u^T tanh(A d_ij)`

Final matrix is explicitly antisymmetrized:

`p = 0.5 * (raw - raw^T)`

and diagonal is exactly zero.

Thus option swap flips pair direction.

## Gold-supervised TRAIN loss

Only gold-vs-distractor pairs are supervised.

For gold g and each j != g:

`L_gj = softplus(-p_gj)`

Aggregate mean over active gold pairs.

Distractor-vs-distractor pairs receive **no target and no loss**.

No teacher/pseudo-target is used.

## Full-K decision

For each option:

`s_i = sum_{j != i} p_ij / (K-1)`

Treatment final decision logits are `s`.

Reference final decision remains the existing fused decision shell.

Both arms carry identical correction and pairwise-head capacity. Pairwise training uses the same TRAIN-only gold labels in both arms. Pairwise inputs are detached, so head optimization cannot change correction/native trajectories.

## Tie rule

Raw pairwise decision logits remain the authoritative full-K scores.

If exact top-score ties occur, the implementation computes a frozen representation-derived key from the detached 512D option representation. The key is used only among exactly tied top options. No native/fused/corrected logit is allowed as a tiebreak.

If tied options are representation-identical as well, the operational fallback is the lowest current index; this degenerate symmetry is reported explicitly and is not counted as evidence of semantic distinguishability.

## Required S59-A0

Head:
- exactly **32,832** trainable params
- exactly two trainable tensors A/u
- no bias
- deterministic initialization
- exact checkpoint roundtrip
- anti-symmetry
- diagonal zero
- option permutation equivariance
- full-K aggregation
- K=3/7/255
- finite zero/degenerate inputs.

Supervision:
- gold-vs-distractor count = B*(K-1)
- distractor-vs-distractor supervised count = 0
- correct positive gold margin has lower loss than sign-flipped margin
- gradients reach both A and u.

Ownership:
- pairwise representation detached
- gradients to representation/native/cache/correction = 0
- no teacher dependency
- no raw score/logit input API
- correction params **114,688**
- pairwise params **32,832**
- native trainable params **0**
- same head initialization in both arms
- one encoder/state-once.

Decision:
- deterministic nondegenerate tie break
- no raw-score bypass
- softmax probability mass <= 1e-6.

## Fresh S59 authority

Intended:
- seed **80001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S59 domains
- exact S58 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Frozen interpretation

**A** — stability improves and correctness/discrimination remains or improves:
explicit pairwise decision solves the inherited-boundary failure.

**B** — correctness improves but stability remains weak:
learned pairwise boundary is useful; next family may add a separately preregistered stability mechanism.

**C** — stability improves but correctness materially falls:
pairwise-only aggregation is too lossy; next family is calibrated hybrid decision composition.

**D** — correctness and stability both materially regress:
reject this pairwise-head family.

**E** — full DEV_READY:
freeze and open separate confirmation before external Laya/Jev evaluation.

## Stop rule

After one S59 DEV:
- no width/activation sweep
- no loss variant
- no tie-rule variant
- no base-logit blend
- no joint-gradient variant
- no architecture/capacity change
- no native retraining
- no selector change
- no retry
- no gate weakening
- no second DEV
- no external Laya/Jev evaluation.
