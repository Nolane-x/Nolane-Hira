# HIRA V1 S73 contract — Counterfactual Safety Veto

Status: **FROZEN / PRE-A0**

Issue: #333

Parent:
- S72 merged main `16eb7a583376cd824421d5e7ae9df948979b3953`
- fresh run `37470998139`
- artifact `11416962628`
- digest `sha256:9a653a681e0cb3182819634fe24eb345ce82a48f2d078d15e510d1d3ebfb0834`
- verdict **Case B**.

## Scientific question

Can a tiny TRAIN-only counterfactual safety predictor hard-veto unsafe uses of the frozen S72 candidate residual, improving final correctness/stability without changing the candidate itself?

## Frozen upstream

Both arms use exactly:
- S69 query-gated identity interaction representation;
- S59 pairwise head and objective;
- shared pairwise state/training trajectory;
- shared correction state/training trajectory;
- S71 mean-only **80-param composer**;
- S72 opponent-profile residual direction;
- S66 TRAIN-only responsibility objective for the composer;
- no teacher/pseudo-target/DEV target.

## Frozen candidate

`h_candidate=fused + alpha*fused_rms*d_s72`.

Candidate logits are identical in both arms.

## TRAIN-only safety target

For each view independently, while the opposite view remains fused baseline:

`y=1` iff:
1. candidate own-view CE <= fused own-view CE + **1e-8**;
2. candidate-vs-opposite-fused JS <= all-fused JS + **1e-8**.

Otherwise `y=0`.

Gold is used only to construct TRAIN labels.

## Predictor

Eight detached inference features:
1. normalized fused top1-top2 margin;
2. normalized fused entropy;
3. alpha / 0.35;
4. normalized absolute candidate change at fused top1;
5. candidate-vs-fused top1 conflict;
6. normalized pairwise-row-mean top1 margin;
7. S72 direction max abs;
8. cosine(centered fused, centered S72 direction).

Linear logistic predictor:
- 8 weights + 1 bias = **9 trainable params**;
- zero initialization;
- initial probability exactly **0.5**;
- BCEWithLogits;
- no class weighting/smoothing;
- hard accept threshold exactly **0.5**.

## Matched arms

Both arms train/checkpoint the same 9-param predictor.

Reference:
- predictor retained but ignored at inference;
- always emits exact S72 candidate.

Treatment:
- accept => exact S72 candidate;
- veto => exact fused baseline.

No interpolation is permitted.

## A0 requirements

Must prove:
- 9 vs 9 params
- bit-identical predictor initialization/state
- initial probability 0.5
- threshold exactly 0.5
- reference exact candidate
- treatment all-accept exact candidate
- treatment all-veto exact fused
- ordinary treatment output is endpoint-only
- features detached, finite and permutation invariant
- target detached and uses the frozen CE+JS rule
- predictor gradient live
- no predictor-loss gradient into fused/candidate/pair/direction/alpha/native/correction/composer
- K=3/7/255
- checkpoint replay exact
- fresh S73 TRAIN/DEV exposed=false.

## Fresh authority

Only after A0 and exact pre-DEV CI:
- seed **94001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S73 domains
- exact S72 state/question/option overlap **0**
- K=4
- 24 epochs
- one DEV only.

## Stop rule

After one S73 DEV:
- no threshold sweep
- no feature sweep
- no class-weight/smoothing sweep
- no target change
- no residual/representation/head/composer change
- no retry
- no second DEV
- no external Laya/Jev evaluation.

Scientific failure is valid.
