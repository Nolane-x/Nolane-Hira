# HIRA V1 S73 matched scientific receipt — Counterfactual Safety Veto

Status: **FROZEN / CASE C**

Scientific run: `37480212536`  
Artifact: `11420533826`  
Artifact digest: `sha256:0f08d05bea219333495529c6ae2c54d65fd9a276dd5f783adff0b0ac3925982f`  
Scientific head: `0d011eb4ed60032dc0447894b67dd152996bb858`

Outcome:
`HIRA_V1_S73_COUNTERFACTUAL_SAFETY_VETO_DEV_COMPLETE`

A0:
- run `37476327384`
- artifact `11418978543`
- digest `sha256:bab338c55ae37bd322fcc20d838ed3b4250da959e2d7135525ec611da8f9c083`
- outcome `HIRA_V1_S73_A0_COUNTERFACTUAL_SAFETY_VETO_READY`.

## Controlled variable

Both arms share:
- exact S69 representation
- exact S59 pairwise head
- exact S71 mean-only composer
- exact S72 opponent-profile residual
- exact same 9-param S73 predictor
- same predictor state
- same selected epoch
- same raw pairwise evidence
- same alpha policy.

Reference:
- always emits S72 candidate.

Treatment:
- hard threshold 0.5
- emits exact candidate or exact fused baseline only.

Selected epoch: **24** both arms.

## Predictor policy

Reference predictor probability diagnostics:
- mean probability **0.3449220744**
- std **0.0287376898**.

Treatment uses the exact same predictor state.

Treatment DEV accept rate:
- **0.0000000000**

Therefore the hard policy collapses to **reject-all / fused-only** on fresh DEV.

## Matched upstream evidence

Both arms:
- pairwise gold-pair accuracy **0.6006944444**
- pairwise mean gold-pair margin **0.1757628093**
- composer mean alpha **0.0620717644**
- composer min alpha **0.0510277599**
- composer max alpha **0.0866026506**
- composer std alpha **0.0079451983**.

## Final decision

Reference — always S72 candidate:
- canonical accuracy **0.3411458333**
- paraphrase accuracy **0.4244791667**
- paired both-correct **0.0989583333**
- question-swap **0.3281250000**
- agreement **0.4062500000**
- JS **0.0858260392**.

Treatment — hard veto:
- canonical accuracy **0.3463541667**
- paraphrase accuracy **0.4166666667**
- paired both-correct **0.1145833333**
- question-swap **0.3437500000**
- agreement **0.4140625000**
- JS **0.0838671274**.

Treatment minus reference:
- canonical **+0.520833 pp**
- paraphrase **-0.781250 pp**
- paired both-correct **+1.562500 pp**
- question-swap **+1.562500 pp**
- agreement **+0.781250 pp**
- JS **-0.001958912** better
- canonical margin **+0.012439252**
- paraphrase margin **-0.008767169**.

These are mixed changes produced by a policy that collapsed to reject-all, not by selective safety prediction.

## Frozen interpretation

**Case C.**

The detached eight-feature predictor does not learn a nontrivial safety policy on fresh DEV. It collapses below the fixed 0.5 threshold for every item.

Per the preregistered stop rule:
- no threshold sweep
- no feature sweep
- no class weighting/smoothing
- no target rescue
- no residual/representation/head/composer rescue
- no second DEV.

The bounded pairwise-residual/selective-veto family is closed. The next intervention must return to the **decision-core architecture itself**.
