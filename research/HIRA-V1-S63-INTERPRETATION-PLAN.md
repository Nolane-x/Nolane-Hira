# HIRA V1 S63 frozen interpretation plan

Status: **FROZEN BEFORE A0 / BEFORE FRESH TRAIN-DEV**

## A

Treatment materially improves cross-view selected-choice agreement and/or JS over the exact S62 reliability-gate reference while retaining or improving correctness and question-swap discrimination.

Interpretation:
the learned permutation-invariant decision-surface representation solves the S62 feature bottleneck.

## B

Treatment learns materially different reliability behavior but stability improvement remains weak.

Interpretation:
the decision-surface geometry itself is insufficient. The next family must learn reliability from richer detached state/query/option representation rather than merely expanding surface features.

## C

Treatment materially improves stability but correctness falls.

Interpretation:
retain the learned reliability representation, but the next family must preregister a hard correctness-preserving inference veto/anchor.

## D

Correctness and stability both regress.

Interpretation:
reject the learned set reliability gate family.

## E

Full DEV_READY under frozen gates.

Interpretation:
freeze immediately and open a separate fresh confirmation court before any external Laya/Jev evaluation.

## Stop rule

After one S63 DEV:
- no feature change;
- no hidden-width sweep;
- no pooling change;
- no initialization change;
- no alpha-probe/tolerance/target change;
- no BCE weighting;
- no optimizer/LR/weight-decay change;
- no regularizer;
- no gradient coupling;
- no capacity change;
- no native retraining;
- no selector change;
- no retry for scientific weakness;
- no second S63 DEV;
- no external Laya/Jev evaluation.

Scientific failure is valid.
