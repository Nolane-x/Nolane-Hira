# HIRA V1 S66 matched scientific receipt — Per-View Counterfactual Responsibility

Status: **FROZEN / CASE B**

Scientific run: `37331468679`  
Artifact: `11354228532`  
Artifact digest: `sha256:f3c6322eec8cce02a698807037a4c05e48c32b2addc5709a116a4384338379b1`  
Scientific head: `e4b3a18b26288efda9128611444ef2de906c68bf`

Outcome:
`HIRA_V1_S66_PER_VIEW_RESPONSIBILITY_DEV_COMPLETE`

## Controlled variable

Reference:
- exact S64 contextual single-view gate
- **60 trainable params**
- shared S62 pair target applied independently to both views.

Treatment:
- exact same S64 contextual single-view gate
- **60 trainable params**
- explicit TRAIN-only per-view counterfactual responsibility targets.

Treatment parameter advantage: **0**.

Shared exactly:
- correction trajectory **114,688 params**
- S59 pairwise-head trajectory **32,832 params**
- context path/projection
- gate initialization
- alpha max **0.35**
- initial alpha **0.10**
- probe alpha **0.35**
- tolerance **1e-8**
- optimizer/LR/weight decay
- frozen S17 selector
- immutable native/cache evidence
- single-view inference.

## Selected checkpoints

Reference epoch: **19**  
Treatment epoch: **19**

## Treatment minus reference

- canonical accuracy **0.00 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap **0.00 pp**
- selected-choice agreement **0.00 pp**
- cross-view JS **-0.0006435302** better
- canonical gold margin **-0.0016827919**
- paraphrase gold margin **+0.0002006336**
- relation metrics **unchanged**.

## Gate behavior

Reference:
- mean alpha **0.0785311513**
- alpha range **0.0729355812–0.0859336555**
- alpha std **0.0028189884**.

Treatment:
- mean alpha **0.0668237867**
- alpha range **0.0530680753–0.0892903134**
- alpha std **0.0078778375**.

The treatment therefore learned a materially more selective policy, even though selected DEV outcome metrics moved only microscopically.

## TRAIN responsibility evidence

At epoch 24:
- shared pair-target positive fraction **0.173828125**
- canonical responsibility positive fraction **0.129557292**
- paraphrase responsibility positive fraction **0.1796875**
- canonical/paraphrase target disagreement fraction **0.305338542**.

This proves the S66 supervision carries substantial information unavailable to a shared pair label.

## Frozen interpretation

**Case B.**

Reason:
- per-view responsibility clearly changes the learned alpha distribution;
- cross-view JS improves in the intended direction;
- correctness/discrimination are retained;
- however the JS gain is only about **6.44e-4**, while agreement and correctness remain exactly unchanged.

That is evidence of useful signal, but not a material stability solution under the preregistered interpretation.

No S66 target smoothing, tolerance/probe sweep, CE/JS coefficient, architecture change, retry, second DEV, selector change, native retraining or external Laya/Jev evaluation is authorized.
