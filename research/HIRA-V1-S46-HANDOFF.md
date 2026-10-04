# HIRA V1 S46 handoff — to S47 Ordinal Pairwise Consensus

S46 is frozen as **Case C**.

## What S46 established

At the exact same selected checkpoint:
- legacy S45 shell agreement: **0.6901042**
- S46 median agreement: **0.6015625**
- legacy JS: **0.0195794**
- S46 median JS: **0.0483414**

The low-JS native relation expert:
- canonical **0.3541667**
- paraphrase **0.2864583**
- relation JS **0.0010528**

did not stabilize the final decision when fused by coordinate-wise median.

This suggests score magnitude itself is part of the problem: a weak expert can still distort option coordinates despite looking numerically smooth across views.

## S47 scientific direction

Change from **score aggregation** to **ordinal pairwise consensus**.

Keep exact S45 training mechanics and exact selected-checkpoint protocol.

At inference, use three existing one-pass experts:
1. primary triadic;
2. native relation;
3. corrected private relation.

For each expert, discard score magnitude and retain only ordering.

For every option pair `(i,j)`:
- each expert casts one vote for whichever option has the larger logit;
- majority vote over 3 experts determines the pairwise winner.

For each option:
- Copeland score = number of pairwise wins minus pairwise losses.

Final S47 full-K evidence:
- primary value = Copeland score;
- exact Copeland ties are broken lexicographically by the corrected-private expert's ordinal rank, because S44-S46 already established that the private expert is the dedicated correctness source;
- tie-break is rank-only, not magnitude;
- no learned/fitted scalar.

The implementation must encode lexicographic priority without a tunable coefficient, e.g. exact integer score construction based on K.

## Why this is a genuinely new family

Unlike S46 median:
- no expert magnitude enters;
- additive/scale differences become irrelevant;
- one extreme logit cannot dominate;
- pairwise majority lets two experts overrule one outlier;
- corrected-private evidence is used only as deterministic ordinal tie-break where pairwise consensus is insufficient.

## S47-A0 must prove

- zero trainable decision parameters;
- full-K output for K=3/7/255;
- exact option-permutation equivariance;
- independent monotonic positive affine invariance per expert;
- arbitrary positive monotonic rank-preserving transforms leave output unchanged;
- one-expert extreme-magnitude invariance;
- two-expert identical-order dominance over one adversarial expert;
- deterministic corrected-private ordinal tie-break;
- no use of raw score magnitude after ranking;
- finite softmax/probability mass if converted to probabilities;
- one encoder pass/state-once;
- correction/native ownership unchanged.

## Fresh authority

S47 must use wholly fresh A0 + TRAIN/DEV rows.

No exact S46 DEV rows may be reused.

One DEV only. No:
- Borda-vs-Copeland sweep
- majority threshold
- tie-break expert sweep
- rank temperature
- learned gate
- scalar weight
- second encoder
- post-DEV tuning.

If S47 fails, move family again.
