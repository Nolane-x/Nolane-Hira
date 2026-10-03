# HIRA V1 S45 handoff — to S46 Robust Three-Expert Evidence Consensus

S45 is frozen as **Case C**.

## What remains strong

The detached private correction branch still produces useful correctness:
- +20.83 pp canonical relation accuracy
- +7.29 pp paraphrase relation accuracy
- +7.03 pp fused canonical accuracy
- exact native trajectory preservation
- one encoder pass
- 163,840 total treatment trainables

So S44/S45 established a useful private correctness expert.

## What S45 disproved

A direct cross-view JS penalty is not enough.

Fresh treatment-reference:
- relation JS **+0.03762** worse
- fused JS **+0.04796** worse
- fused agreement **-8.07 pp**

At the exact same native runtime epoch 7:
- relation agreement **-16.93 pp**
- relation JS **+0.03762**
- fused agreement **-9.38 pp**

This means the next stage should not keep trying to make the private expert itself perfectly stable by scalar loss shaping.

## S46 scientific direction

Change the **decision/fusion family** while keeping:
- exact S45 private correction architecture
- exact S45 CE+JS training objective
- exact native ownership split
- exact parameter counts
- one encoder pass.

Expose three already-computed full-K experts:
1. primary triadic evidence;
2. native relation evidence;
3. corrected private relation evidence.

Replace the existing two-expert equal-mean fusion with a **zero-parameter robust three-expert consensus**.

Preferred preregistered operator:
- standardize each expert independently with the existing S14 centering/RMS primitive;
- stack the three standardized full-K vectors;
- take the **coordinate-wise median** across the expert axis.

Why median:
- zero trainable parameters;
- no fitted scalar;
- no temperature;
- permutation equivariant over K;
- robust to one arbitrarily extreme expert per option;
- exact identity when all experts agree;
- uses the private expert when it is supported rather than allowing one unstable view to dominate magnitude.

## S46-A0 must prove

- zero trainable fusion parameters;
- exact full-K option permutation equivariance;
- independent shift/positive-scale invariance inherited from standardization;
- identical-three-expert identity;
- one-expert-outlier containment: median score stays inside the interval of the other two standardized scores for every option;
- finite output under flat expert;
- arbitrary K=3/K=7/K=255;
- no gradient from fused primary objective into corrected relation/native relation if gradient isolation is retained;
- one encoder pass / state-once;
- no extra correction parameters;
- checkpoint compatibility.

## Fresh scientific rule

S46 must use wholly fresh A0 and TRAIN/DEV authority.

No exact S45 DEV row is reusable.

No fusion variant sweep:
- no mean vs median comparison on DEV;
- no trimmed mean;
- no learned gate;
- no confidence threshold;
- no entropy coefficient;
- no temperature.

If the median family fails after one fresh DEV, change family again.
