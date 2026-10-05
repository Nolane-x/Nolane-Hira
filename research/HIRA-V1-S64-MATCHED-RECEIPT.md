# HIRA V1 S64 matched scientific receipt — Parameter-Matched Context-Injected Reliability Gate

Status: **FROZEN / CASE B**

Scientific run: `37319541951`  
Artifact: `11348594503`  
Artifact digest: `sha256:f3d04d452a51b7358a23a9391579bca6ce1ab2c16fa107dc9f7f79740dba90ea`  
Scientific head: `eb38469d6139a46da3d2421135edff38b442c4f7`

Outcome:
`HIRA_V1_S64_CONTEXTUAL_RELIABILITY_GATE_DEV_COMPLETE`

## Controlled variable

Reference:
- **60 trainable params**;
- surface4 decision features;
- four context channels forced exactly to zero;
- exact S62 TRAIN-only reliability BCE target.

Treatment:
- **60 trainable params**;
- same surface4 features;
- four context channels from fixed projection of detached S59 state/query/option representation;
- exact same S62 TRAIN-only reliability BCE target.

Treatment parameter advantage: **0**.

Shared exactly:
- correction trajectory **114,688 params**
- pairwise-head trajectory **32,832 params**
- parameter initialization
- context projection buffer
- alpha max **0.35**
- initial alpha **0.10**
- alpha probe **0.35**
- target tolerance **1e-8**
- optimizer/LR/weight decay
- frozen S17 selector
- immutable native/cache evidence.

## Selected checkpoints

Reference epoch: **15**  
Treatment epoch: **15**

## Reference — surface-only matched gate

- canonical accuracy **0.5052083333333334**
- paraphrase accuracy **0.3776041666666667**
- paired both-correct **0.22916666666666666**
- question-swap **0.6510416666666666**
- agreement **0.3541666666666667**
- JS **0.12892126571387053**
- canonical gold margin **-0.048827563102046646**
- paraphrase gold margin **-0.6419397195180258**
- mean alpha **0.0779476851845781**
- alpha range **0.07057363539934158–0.08947118371725082**
- alpha std **0.0033808407900329006**
- pairwise gold-pair accuracy **0.5755208333333334**

## Treatment — contextual matched gate

- canonical accuracy **0.5026041666666666**
- paraphrase accuracy **0.3776041666666667**
- paired both-correct **0.22395833333333334**
- question-swap **0.6510416666666666**
- agreement **0.3541666666666667**
- JS **0.12890253774821758**
- canonical gold margin **-0.04877773548165957**
- paraphrase gold margin **-0.6418593352039655**
- mean alpha **0.07728730980306864**
- alpha range **0.06981301307678223–0.08701418340206146**
- alpha std **0.003218232824752714**
- pairwise gold-pair accuracy **0.5755208333333334**

## Treatment minus reference

- canonical accuracy **-0.26 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **-0.52 pp**
- question-swap discrimination **0.00 pp**
- selected-choice agreement **0.00 pp**
- cross-view JS **-0.000018727965652942657** — microscopically better
- canonical gold margin **+0.00004982762038707733**
- paraphrase gold margin **+0.00008038431406021118**
- relation-binding metrics **unchanged**.

## Frozen interpretation

**Case B — per-view contextual information changes the gate policy only slightly and does not materially recover the S59 stability signal.**

The contextual path is mechanically real:
- A0 proved non-degenerate contextual channels and direct context sensitivity;
- treatment and reference had equal capacity and bit-identical initialization;
- treatment mean alpha moved from **0.07795** to **0.07729**;
- cross-view JS improved by about **1.87e-5**.

But the preregistered stability question is not solved:
- selected-choice agreement is exactly unchanged;
- the JS gain is microscopic;
- canonical accuracy falls about **0.26 pp**;
- paired both-correct falls about **0.52 pp**;
- paraphrase accuracy and question-swap discrimination are unchanged.

Therefore S64 rejects the hypothesis that independent per-view S59 context, injected through a fixed compact projection, is sufficient to predict pairwise usefulness.

Next family: **S65 explicit cross-view/context interaction reliability**, while retaining the S62 correctness veto, bounded residual, detached ownership, matched-capacity discipline where possible, and one-shot fresh DEV.

No S64 context-projection sweep, width sweep, feature retrofit, pooling change, initialization sweep, alpha-probe/tolerance/target change, BCE weighting, retry, second DEV, selector change, native retraining or external Laya/Jev evaluation is authorized.
