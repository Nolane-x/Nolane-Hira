# HIRA V1 S60 contract — Train-Calibrated Bounded Hybrid Pairwise-Evidence Composition

Status: **PREREGISTERED / NO S60-A0 OR DEV EXPOSURE**

Issue: #305

## Parent evidence

S59 closes as **Case C**.

Fresh S59 court:
- run `37289522292`
- artifact `11335428630`
- artifact digest `sha256:72a348c27f95c1f7ea2dc0dbcba8db7afac56fa5cd4d284d0593f7321d603e4c`
- merged main `867056d999bab707a1d8f4b1ea9ca774ccaf514d`.

S59 proved:
- pairwise selected-choice agreement **+32.55 pp**
- pairwise cross-view JS **-0.128684**
- canonical accuracy **-15.36 pp**
- paired both-correct **-21.88 pp**
- question-swap **-66.15 pp**.

Conclusion: learned pairwise evidence is strongly stabilizing but pairwise-only replacement discards absolute fused evidence.

## Scientific question

> Can a bounded pairwise residual, globally calibrated from TRAIN-only gold labels, improve stability while preserving the exact fused evidence path that carries correctness and discrimination?

## Frozen base

Retain:
- S51 persisted native authority;
- immutable encoded cache;
- one encoder/state-once;
- S59 correction shell;
- S59 pairwise representation/head;
- correction params **114,688**;
- pairwise params **32,832**;
- pairwise representation detached;
- no teacher/pseudo-target/self-anchor;
- shared correction/pairwise trajectory;
- frozen S17 selector.

## Composer

Inputs:
- fused logits `f [B,K]`
- learned pairwise aggregate `p [B,K]`.

Frozen epsilon:
- **1e-6**.

Pairwise centered RMS:
`rms_p = sqrt(mean((p-mean(p))^2)).clamp_min(1e-6)`

Pairwise normalized:
`z_p=(p-mean(p))/rms_p`

Bounded direction:
`b_p=tanh(z_p)`

Fused centered RMS:
`s_f=sqrt(mean((f-mean(f))^2)).clamp_min(1e-6)`

One scalar calibration parameter:
`a`.

`alpha = 0.35 * sigmoid(a)`

Frozen:
- alpha max **0.35**
- alpha initial **0.10**
- a initial **-0.916290731874155**
- trainable composer params **1**.

Treatment:
`h = f + alpha * s_f * b_p`

Reference:
`f` exactly.

Explicit `alpha_override=0` MUST return the input fused tensor exactly.

## Bounded influence

Because `|tanh(z_p)|<=1`:

`|h_i-f_i| <= alpha*s_f < 0.35*s_f`

for every option i.

No pairwise magnitude can bypass this bound.

## Equivariance / scale contract

- option permutation equivariant;
- shared fused offset: output receives same shared offset;
- positive fused scale: output scales by same factor;
- shared pairwise offset: no change to pairwise residual;
- positive pairwise scale: no change to pairwise residual;
- K-independent.

## TRAIN calibration

Composer scalar is TRAIN-only.

Per batch:
1. correction base objective updates correction params;
2. S59 pairwise gold objective updates pairwise A/u;
3. fused logits and pairwise aggregate are recomputed/current then detached;
4. composer minimizes:
   `0.5 * (CE(h_c,gold)+CE(h_p,gold))`.

Composer optimizer:
- AdamW
- existing S35 LR
- weight decay **0.0**
- 24 epochs.

Composer gradients MUST NOT enter correction, pairwise head, native runtime, or cache.

## Shared trajectory / matched court

One shared training trajectory contains:
- correction state;
- pairwise head state;
- composer scalar state.

Reference selected decision:
- exact fused logits.

Treatment selected decision:
- calibrated hybrid logits.

Same frozen S17 selector is applied independently to reference/treatment metrics over the same shared epoch trajectory.

## Required S60-A0

Parameter surface:
- composer trainable params **1**
- one trainable tensor only
- alpha initial exactly **0.10** within tolerance 1e-7
- 0 < alpha < 0.35
- checkpoint replay exact.

Identity/bound:
- alpha override 0 gives exact fused logits
- adversarial pairwise magnitudes cannot exceed residual bound
- flat pairwise logits produce zero residual
- flat fused input remains finite.

Equivariance:
- option permutation
- fused shared offset
- fused positive scale
- pairwise shared offset
- pairwise positive scale
- K=3/7/255
- full-K probability mass.

Gradient ownership:
- TRAIN gold CE gives composer scalar gradient >0
- detached fused input gets no gradient
- detached pairwise input gets no gradient
- correction/pairwise/native/cache get no composer gradient.

No:
- teacher
- pseudo-target
- self-anchor
- pairwise-only final path
- K-specific parameters.

## Fresh S60 authority

Intended:
- seed **81001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S60 domains
- exact S59 state/question/option overlap **0**
- K=4
- epochs **24**
- one DEV only.

## Frozen interpretation

**A** — stability improves and correctness/discrimination remains or improves:
bounded global TRAIN calibration solves composition.

**B** — correctness retained but stability does not materially improve:
global alpha too conservative; next family may use preregistered confidence-adaptive gating.

**C** — stability improves but correctness materially falls:
global alpha insufficient; next family is confidence-adaptive hybrid composition.

**D** — both materially regress:
reject bounded global hybrid residual.

**E** — full DEV_READY:
freeze and open a separate fresh confirmation before external Laya/Jev evaluation.

## Stop rule

After one S60 DEV:
- no alpha max/init sweep
- no calibration objective change
- no optimizer/LR/weight-decay change
- no tanh/normalization change
- no adaptive gate
- no gradient coupling
- no architecture/capacity change
- no native retraining
- no selector change
- no retry
- no gate weakening
- no second DEV
- no external Laya/Jev evaluation.
