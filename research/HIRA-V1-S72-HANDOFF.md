# HIRA V1 S72 → S73 handoff

Parent verdict: **S72 Case B**

## What S72 established

The opponent-profile vector residual changes decision geometry materially while:
- raw pairwise evidence is exactly matched;
- composer alpha policy is exactly matched;
- no learned direction parameters are added.

The problem is **selectivity**: useful structural movement exists, but applying the residual to every case harms correctness on enough examples to erase the benefit.

## Required S73 question

> Can a small TRAIN-only counterfactual safety predictor hard-veto unsafe S72 residual applications, improving final correctness/stability without changing the frozen candidate residual itself?

## Frozen upstream

Both arms must share:
- exact S69 representation;
- exact S59 pairwise head and training trajectory;
- exact S71 80-param mean-only composer;
- exact S72 opponent-profile treatment direction;
- same correction trajectory;
- same alpha values;
- no gradient into native/correction/pairwise/composer from the veto;
- one encoder/state-once;
- no teacher or DEV-derived target.

## Candidate residual

Let:
`h_candidate = fused + alpha * fused_rms * d_s72`.

This candidate is frozen and identical in both arms.

## TRAIN-only counterfactual safety target

For each view independently, with the other view held at fused baseline:

Accept target `y=1` iff both:
1. own-view candidate cross-entropy is non-worse than fused baseline by tolerance `1e-8`;
2. paired cross-view JS is non-worse than the all-fused baseline by tolerance `1e-8`.

Else `y=0`.

Gold labels are used **only to construct TRAIN targets**. No gold/DEV signal exists at inference.

## Veto predictor

Use a tiny detached **8-feature linear logistic predictor** per view.

Frozen features:
1. fused top1-top2 margin;
2. fused normalized entropy;
3. alpha;
4. absolute candidate top1 logit change;
5. candidate-vs-fused top1 conflict indicator;
6. pairwise row-mean top1 margin;
7. S72 direction max absolute value;
8. cosine between centered fused logits and S72 direction.

Normalize continuous features with fixed deterministic formulas only.

Predictor:
- `w[8]` + bias = **9 trainable params**;
- initialized to zero bias/weights except bias chosen so initial accept probability is 0.5;
- BCEWithLogits on TRAIN-only safety labels;
- no class reweighting, smoothing or threshold tuning.

Hard threshold:
`accept = sigmoid(z) >= 0.5`.

## Matched arms

Both arms train and checkpoint the same 9-param safety predictor.

Reference:
- ignores veto at inference and always emits `h_candidate`.

Treatment:
- if accept, emits `h_candidate`;
- if veto, emits exact `fused`.

Thus trainable parameter count is exactly matched. The only scientific intervention is **using the hard safety veto**.

## Required A0

Must prove:
- predictor params **9 vs 9**
- bit-identical predictor initialization
- exact same predictor state can be loaded into both arms
- reference output exact S72 candidate
- treatment output is exactly either fused or S72 candidate, never interpolation
- hard threshold exactly 0.5
- target uses TRAIN gold only
- no DEV target path
- no gradient into fused/pairwise/composer/context/native/correction
- permutation-invariant/view-local feature extraction
- finite K=3/7/255
- deterministic checkpoint replay
- all-veto => exact fused
- all-accept => exact S72 candidate
- fresh S73 DEV exposed=false.

## Fresh S73 authority

- seed **94001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- exact S72 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

## Primary readouts

1. final canonical/paraphrase accuracy
2. paired both-correct
3. question-swap
4. agreement / JS
5. raw pairwise evidence remains matched
6. alpha remains matched
7. candidate geometry remains matched
8. predicted accept rate
9. TRAIN oracle accept rate
10. false-accept / false-veto diagnostics on TRAIN only
11. DEV output flip rate caused by veto.

## Frozen interpretation direction

**A** — hard veto materially recovers correctness/stability over always-apply S72 candidate:
selectivity bottleneck resolved; open independent fresh confirmation toward v1.0.

**B** — veto learns nontrivial policy but final gain remains weak:
bounded pairwise residual family is likely exhausted; return to decision-core architecture.

**C** — veto collapses to mostly accept or mostly reject:
available detached features cannot predict safety; stop extending this family.

**D** — veto materially worsens final correctness/stability:
reject safety-veto family.

**E** — full DEV_READY:
freeze and confirm immediately.

No threshold/feature/class-weight/target sweep after DEV.
