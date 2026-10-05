# HIRA V1 S61 contract — Confidence-Adaptive Bounded Hybrid Gate

Status: **PREREGISTERED / NO S61-A0 OR DEV EXPOSURE**

Issue: #307

## Parent evidence

S60 is scientifically closed as **Case B**.

Fresh S60 court:
- run `37297784471`
- artifact `11340397535`
- digest `sha256:c55205e9b74aac6196346a0ce9b705900aaec7fe4c6b30380af8ee0093d1b18b`
- merged main `c795c6c448af942f597a8d33131557338def7fa9`.

Treatment minus reference:
- canonical **+0.52 pp**
- paraphrase **+0.78 pp**
- paired both-correct **-1.56 pp**
- question-swap **0.00 pp**
- selected-choice agreement **+0.26 pp**
- cross-view JS **+0.000412** worse
- selected alpha **0.1050317**.

Conclusion: bounded composition is safe, but one global alpha cannot decide when the stabilizing pairwise signal should matter.

## Scientific question

> Can a tiny per-query confidence-adaptive gate selectively increase the bounded pairwise residual only when detached fused/pairwise evidence supports it, improving cross-view stability without giving up the fused correctness frontier?

## Frozen base

Retain:
- S51 persisted native authority;
- immutable shared cache;
- one encoder/state-once;
- `JointStateQueryOptionPrivateCorrectionFork`;
- S59 explicit pairwise representation/head;
- S60 bounded residual direction;
- correction params **114,688**;
- pairwise params **32,832**;
- detached pairwise representation;
- shared correction/head/gate TRAIN trajectory;
- frozen S17 selector;
- no teacher;
- no pseudo-target;
- no self-anchor;
- pairwise-only final path forbidden.

## Four detached gate features

For each query derive exactly four K-agnostic, option-permutation-invariant scalars from detached fused logits `f` and detached pairwise aggregate `p`.

Centered RMS uses epsilon **1e-6**.

1. `x0 = tanh((top1(f)-top2(f))/rms(f))`
2. `x1 = tanh((top1(p)-top2(p))/rms(p))`
3. `x2 = +1` if `argmax(f)==argmax(p)`, otherwise `-1`
4. Standardize `zf`, `zp`; `x3 = clamp(mean(zf*zp), -1, 1)`

All feature tensors are detached.

## Adaptive gate

Trainable tensors:
- `w: [4]`
- `b: scalar`

Total gate trainable parameters: **5**.
No hidden layer.

`alpha_q = 0.35 * sigmoid(b + w·x_q)`

Frozen initialization:
- `w = 0`
- `b = -0.916290731874155`
- `alpha_max = 0.35`
- every initial query alpha = **0.10**.

## Bounded composition

Use the exact S60 residual direction:

`rp = tanh((p-mean(p))/rms(p))`

`sf = rms(f)`

Treatment:

`h = f + alpha_q * sf * rp`

Reference:
- exact fused logits `f`.

Contracts:
- `alpha_override=0` gives exact fused identity;
- residual coordinate magnitude <= `0.35 * sf`;
- pairwise-only replacement impossible;
- no K-specific state;
- option permutation equivariant;
- fused offset equivariant;
- positive fused scaling rescales output;
- pairwise offset and positive-scale invariant.

## TRAIN calibration

At every TRAIN batch:
1. update correction from existing private objective;
2. update pairwise head from gold pair supervision;
3. recompute current fused/pairwise surfaces;
4. detach both surfaces;
5. update only gate `w,b` using mean canonical+paraphrase gold CE.

Gate optimizer:
- AdamW
- LR = existing S35 LR
- weight decay **0.0**
- 24 epochs.

No gate gradient may enter correction, pairwise head, native runtime or immutable cache.

Controlled change from S60:
**global scalar alpha -> per-query 4-feature linear confidence gate.**

## Required S61-A0

Parameter surface:
- exactly **5** trainable params;
- tensors exactly `w`, `b`;
- w shape [4], b scalar;
- no hidden layer;
- zero w initialization;
- every initial alpha exactly 0.10.

Features:
- dimension exactly 4;
- finite flat-logit behavior;
- option permutation invariance;
- fused offset/positive-scale invariance;
- pairwise offset/positive-scale invariance;
- agreement exactly +/-1;
- detached.

Adaptation:
- controlled manual nonzero weights produce different alpha across query probes;
- alpha always in (0,0.35);
- override 0 exact fused identity;
- adversarial pairwise residual remains bounded.

Ownership:
- TRAIN gold CE gradients reach w and b;
- fused/pairwise input gradients zero after detach;
- correction/pairwise/native/cache gradients zero;
- no teacher/pseudo-target/self-anchor.

Full-K:
- K=3/7/255;
- probability mass <=1e-6;
- one encoder/state-once;
- deterministic checkpoint roundtrip.

## Fresh S61 authority

Intended:
- seed **82001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S61 domains
- exact S60 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Frozen interpretation

**A** — materially better stability while correctness/discrimination retained or improved:
confidence-adaptive bounded composition is a viable v1 decision architecture.

**B** — correctness remains but stability gain still weak:
pairwise signal needs a stronger TRAIN-only reliability target.

**C** — stability improves materially but correctness falls:
gate is too permissive; next family needs an explicit correctness-preserving veto/anchor.

**D** — correctness and stability both regress:
reject confidence-adaptive residual gating.

**E** — full DEV_READY:
freeze immediately and open a separate fresh confirmation court before external Laya/Jev evaluation.

## Stop rule

After one S61 DEV:
- no feature addition/removal;
- no hidden layer;
- no feature scaling sweep;
- no alpha_max/init sweep;
- no calibration objective variant;
- no optimizer/LR/weight-decay change;
- no gate regularizer;
- no gradient coupling;
- no architecture/capacity change;
- no native retraining;
- no selector change;
- no retry;
- no gate weakening;
- no second S61 DEV;
- no external Laya/Jev evaluation.
