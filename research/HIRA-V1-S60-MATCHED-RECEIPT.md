# HIRA V1 S60 matched scientific receipt — Train-Calibrated Bounded Hybrid

Status: **FROZEN / CASE B**

Scientific run: `37297784471`  
Artifact: `11340397535`  
Artifact digest: `sha256:c55205e9b74aac6196346a0ce9b705900aaec7fe4c6b30380af8ee0093d1b18b`  
Scientific head: `7352d5bd5d52233f0d6b815682102ce226a98209`

Outcome:
`HIRA_V1_S60_TRAIN_CALIBRATED_BOUNDED_HYBRID_DEV_COMPLETE`

## Controlled variable

Reference:
- existing fused decision.

Treatment:
- `fused + alpha * fused_rms * tanh(standardized_pairwise)`
- alpha max **0.35**
- initial alpha **0.10**
- selected alpha **0.10503172129392624**
- composer params **1**
- pairwise-only path **false**
- no teacher / pseudo-target / self-anchor.

Both arms share the same correction, pairwise and composer training trajectory.

## Selected checkpoints

Reference epoch: **24**  
Treatment epoch: **24**

## Reference

- canonical accuracy **0.5416666666666666**
- paraphrase accuracy **0.3307291666666667**
- paired both-correct **0.2760416666666667**
- question-swap **0.75**
- agreement **0.328125**
- JS **0.1310040783137083**

## Treatment

- canonical accuracy **0.546875**
- paraphrase accuracy **0.3385416666666667**
- paired both-correct **0.2604166666666667**
- question-swap **0.75**
- agreement **0.3307291666666667**
- JS **0.13141609293719134**
- composer alpha **0.10503172129392624**
- pairwise gold-pair accuracy **0.5837673611111112**

## Treatment minus reference

- canonical accuracy **0.52 pp**
- paraphrase accuracy **0.78 pp**
- paired both-correct **-1.56 pp**
- question-swap **0.00 pp**
- selected-choice agreement **0.26 pp**
- cross-view JS **0.000412**
- canonical gold margin **0.008635**
- paraphrase gold margin **0.014180**

## Frozen interpretation

**Case B — correctness retained/improved, stability does not materially improve.**

The bounded residual successfully prevents the S59 pairwise-only correctness collapse:
- canonical correctness improves slightly;
- paraphrase correctness improves slightly;
- question-swap discrimination is retained;
- margins improve.

But the global scalar is too conservative to transfer the strong S59 stability signal:
- selected-choice agreement rises only ~0.26 pp;
- cross-view JS is slightly worse.

Therefore S60 validates bounded hybrid composition as safe, but rejects one global alpha as sufficient.

Next family: **S61 confidence-adaptive bounded hybrid composition**, where residual influence varies per query using preregistered confidence/disagreement evidence while remaining bounded and TRAIN-only calibrated.

No S60 alpha sweep, retry, second DEV, alternate normalization/nonlinearity, per-query gate retrofit, selector change, native retraining or external Laya/Jev evaluation is authorized.
