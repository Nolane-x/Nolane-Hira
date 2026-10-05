# HIRA V1 S62 matched scientific receipt — TRAIN-Only Reliability-Supervised Adaptive Gate

Status: **FROZEN / CASE B**

Scientific run: `37307658289`  
Artifact: `11344855774`  
Artifact digest: `sha256:7812c348d707ea8675fc9370af573a87d9e5b1a0b10e55a1f335bf890ddb3305`  
Scientific head: `f86e6b6296737d6e3274c93498217054eb7bc3a6`

Outcome:
`HIRA_V1_S62_TRAIN_ONLY_RELIABILITY_SUPERVISED_ADAPTIVE_GATE_DEV_COMPLETE`

## Controlled variable

Reference:
- exact S61 5-parameter adaptive gate;
- gate objective: canonical+paraphrase **gold CE**.

Treatment:
- identical 5-parameter adaptive gate;
- gate objective: **TRAIN-only counterfactual reliability BCE**.

Shared exactly:
- correction trajectory **114,688 params**
- pairwise-head trajectory **32,832 params**
- four S61 gate features
- gate initialization
- alpha max **0.35**
- initial alpha **0.10**
- optimizer/LR/weight decay
- frozen S17 selector
- immutable native/cache evidence.

Treatment adds **0 parameters**.

Reliability target:
- fixed probe alpha **0.35**
- tolerance **1e-8**
- positive iff probe CE is non-worse AND paired-view JS strictly improves
- no DEV-derived target.

## Selected checkpoints

Reference epoch: **23**  
Treatment epoch: **23**

## Reference — gold CE gate

- canonical accuracy **0.4739583333333333**
- paraphrase accuracy **0.3541666666666667**
- paired both-correct **0.21875**
- question-swap **0.65625**
- agreement **0.2578125**
- JS **0.13918909740944704**
- gate mean alpha **0.105502400547266**
- gate min alpha **0.09846032410860062**
- gate max alpha **0.11168564110994339**
- gate std alpha **0.0027201159353904794**

## Treatment — reliability BCE gate

- canonical accuracy **0.4739583333333333**
- paraphrase accuracy **0.3515625**
- paired both-correct **0.21875**
- question-swap **0.65625**
- agreement **0.2552083333333333**
- JS **0.13942998523513475**
- gate mean alpha **0.08251461200416088**
- gate min alpha **0.07619068026542664**
- gate max alpha **0.09289570152759552**
- gate std alpha **0.003323662042322327**
- pairwise gold-pair accuracy **0.6067708333333334**

## Treatment minus reference

- canonical accuracy **0.00 pp**
- paraphrase accuracy **-0.26 pp**
- paired both-correct **0.00 pp**
- question-swap discrimination **0.00 pp**
- selected-choice agreement **-0.26 pp**
- cross-view JS **0.00024088782568770783**
- canonical gold margin **-0.0043446479054788795**
- paraphrase gold margin **-0.0046374226609866565**

## Frozen interpretation

**Case B — reliability supervision changes gate behavior, but stability improvement remains weak.**

The reliability objective clearly changed the gate:
- mean alpha moved from **0.105502400547266** to **0.08251461200416088**;
- treatment alpha range became **0.07619068026542664–0.09289570152759552**;
- the gate therefore learned a materially more conservative residual policy.

But the preregistered stability question is not solved:
- selected-choice agreement falls by about **0.26 pp**;
- cross-view JS is slightly worse;
- canonical accuracy is unchanged;
- paraphrase accuracy falls about **0.26 pp**;
- correctness/discrimination do not gain enough to compensate.

Therefore S62 rejects the hypothesis that the four S61 scalar features are sufficient for TRAIN-only reliability supervision.

Next family: **S63 richer TRAIN-only reliability representation/head**. It may increase reliability-model expressivity while preserving the bounded residual, correctness veto, detached upstream ownership, equal matched capacity between reference/treatment where applicable, and one-shot fresh DEV discipline.

No S62 alpha-probe sweep, tolerance change, target-rule change, BCE weighting, feature retrofit, hidden-layer retrofit, retry, second DEV, selector change, native retraining or external Laya/Jev evaluation is authorized.
