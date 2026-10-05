# HIRA V1 S63 matched scientific receipt — Learned Permutation-Invariant Set Reliability Gate

Status: **FROZEN / CASE B**

Scientific run: `37314015283`  
Artifact: `11347706059`  
Artifact digest: `sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e`  
Scientific head: `ddfa724edc8665518489dc82a7b22267081d82ed`

Outcome:
`HIRA_V1_S63_LEARNED_SET_RELIABILITY_GATE_DEV_COMPLETE`

## Controlled variable

Reference:
- exact S62 four-scalar reliability gate;
- **5 trainable params**;
- exact S62 TRAIN-only reliability BCE target.

Treatment:
- learned permutation-invariant set reliability gate;
- **61 trainable params**;
- **+56 params** vs reference;
- per-option normalized fused/pairwise geometry;
- shared `4 -> 8` tanh encoder;
- mean+max pooling;
- exact four S61 scalar features concatenated;
- final reliability representation dim **20**;
- exact same S62 TRAIN-only reliability BCE target.

Shared exactly:
- correction trajectory **114,688 params**
- pairwise-head trajectory **32,832 params**
- alpha max **0.35**
- initial alpha **0.10**
- alpha probe **0.35**
- target tolerance **1e-8**
- optimizer/LR/weight decay
- frozen S17 selector
- immutable native/cache evidence.

## Selected checkpoints

Reference epoch: **4**  
Treatment epoch: **4**

## Reference — S62 four-scalar reliability gate

- canonical accuracy **0.4505208333333333**
- paraphrase accuracy **0.3385416666666667**
- paired both-correct **0.203125**
- question-swap **0.5989583333333334**
- agreement **0.2994791666666667**
- JS **0.12400765375544627**
- mean alpha **0.09659098802755277**
- alpha range **0.09518265724182129–0.0985473170876503**
- alpha std **0.000732101939028997**

## Treatment — learned set reliability gate

- canonical accuracy **0.453125**
- paraphrase accuracy **0.3385416666666667**
- paired both-correct **0.203125**
- question-swap **0.6041666666666666**
- agreement **0.2994791666666667**
- JS **0.12401692351947229**
- mean alpha **0.0889854443569978**
- alpha range **0.08658302575349808–0.09229219704866409**
- alpha std **0.001186576162018632**
- pairwise gold-pair accuracy **0.5625**

## Treatment minus reference

- canonical accuracy **+0.26 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap discrimination **+0.52 pp**
- selected-choice agreement **0.00 pp**
- cross-view JS **+0.000009269764026015315** worse
- canonical gold margin **+0.00005223788321018219**
- paraphrase gold margin **-0.001138768158853054**

## Frozen interpretation

**Case B — the richer decision-surface representation changes reliability behavior, but stability remains weak.**

The treatment is mechanically active:
- mean alpha moves from **0.09659** to **0.08899**;
- alpha variance increases;
- canonical correctness improves slightly;
- question-swap discrimination improves slightly.

But the preregistered stability question is not solved:
- selected-choice agreement is unchanged;
- cross-view JS is microscopically worse;
- paraphrase correctness is unchanged;
- the +56 parameter set encoder does not recover the S59 stability signal.

Therefore S63 rejects the hypothesis that richer geometry of the detached fused/pairwise decision surfaces alone is enough.

Next family: **S64 richer state/query/option-conditioned TRAIN-only reliability representation**, while retaining the S62 correctness-veto target, bounded residual, detached upstream ownership, and one-shot fresh DEV discipline.

No S63 feature retrofit, hidden-width sweep, pooling change, initialization sweep, alpha-probe/tolerance/target change, BCE weighting, retry, second DEV, selector change, native retraining or external Laya/Jev evaluation is authorized.
