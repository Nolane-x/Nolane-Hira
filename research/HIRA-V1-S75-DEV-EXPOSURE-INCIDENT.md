# HIRA V1 S75 fresh-DEV exposure incident

Status: **FROZEN / NO RERUN**

Scientific workflow:
- run `37579774298`
- head `a613cbdcb06d6c489aa0516dffd0b1034f15fd05`
- step 13: **Run one fresh S75 TRAIN DEV court**
- conclusion: **FAILURE**
- no artifact uploaded because the scientific step failed before receipt finalization.

## Exposure point

The failure occurred in:

`_materialize_rep_caches(dev_cache)`

after:
- fresh S75 TRAIN cache had already been materialized;
- fresh S75 DEV cache had already been materialized/encoded;
- TRAIN reference/treatment representation cache construction had passed its non-degeneracy guard.

The DEV representation cache then hit:

`RuntimeError: S75 TBER treatment representation degenerate`

This means the preregistered TBER controlled intervention failed the fresh-DEV non-degeneracy prerequisite.

## What is established

A0 mechanical authority had already shown:
- neutral collapse to S69 error **0**;
- nontrivial synthetic representation difference **0.1254134774**;
- 512D vs 512D;
- 0 representation params vs 0;
- exact 32,832-param S59 pairwise heads;
- upstream gradient isolation;
- K=3/7/255 PASS.

The one-shot fresh court establishes a stronger generalization failure:
- TRAIN cache passed the treatment-reference non-degeneracy guard;
- fresh DEV cache did **not**;
- therefore TBER did not preserve a distinct controlled representation on the fresh DEV authority.

## Scientific discipline

DEV was exposed before the exception. Therefore:
- **do not rerun S75**;
- do not weaken/remove the DEV non-degeneracy guard and rerun;
- do not change token-weight formula, temperature, pooling, normalization, interaction scale, head capacity, objective or target;
- do not open a second S75 DEV;
- do not claim pairwise or DEV_READY metrics that were never produced.

No scientific receipt artifact exists for run `37579774298`.

S75 is closed as a fresh-DEV control-collapse failure, not as a successful semantic intervention.
