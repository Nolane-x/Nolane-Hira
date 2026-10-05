# HIRA V1 S61 matched scientific receipt — Confidence-Adaptive Bounded Hybrid Gate

Status: **FROZEN / CASE B**

Scientific run: `37302829277`  
Artifact: `11343045582`  
Artifact digest: `sha256:6e99ba1a5fab92cb6031c41eea54636721b78e690236dfb39788066803785f74`  
Scientific head: `d731fe8c33e97279681116327bc0500cd340bb83`

Outcome:
`HIRA_V1_S61_CONFIDENCE_ADAPTIVE_BOUNDED_HYBRID_DEV_COMPLETE`

## Controlled variable

Reference:
- existing fused decision shell.

Treatment:
- confidence-adaptive bounded hybrid:
  `fused + alpha(q) * fused_rms * tanh(standardized_pairwise)`
- gate features **4**
- gate params **5**
- no hidden layer
- alpha max **0.35**
- initial alpha **0.10**
- pairwise-only path **false**
- no teacher / pseudo-target / self-anchor.

Both arms share the same correction, pairwise and gate training trajectory.

## Selected checkpoints

Reference epoch: **17**  
Treatment epoch: **21**

## Reference

- canonical accuracy **0.5**
- paraphrase accuracy **0.3489583333333333**
- paired both-correct **0.23958333333333334**
- question-swap **0.6822916666666666**
- agreement **0.3203125**
- JS **0.14236914676924547**

## Treatment

- canonical accuracy **0.5026041666666666**
- paraphrase accuracy **0.375**
- paired both-correct **0.234375**
- question-swap **0.6979166666666666**
- agreement **0.3203125**
- JS **0.1456414001683394**
- gate mean alpha **0.1037044757977128**
- gate min alpha **0.09383107721805573**
- gate max alpha **0.11369072645902634**
- gate std alpha **0.005762574538066296**
- pairwise gold-pair accuracy **0.6063368055555556**

## Treatment minus reference

- canonical accuracy **0.26 pp**
- paraphrase accuracy **+2.60 pp**
- paired both-correct **-0.52 pp**
- question-swap discrimination **+1.56 pp**
- selected-choice agreement **0.00 pp**
- cross-view JS **+0.003272**
- relation canonical accuracy **-1.30 pp**
- relation paraphrase accuracy **-1.82 pp**
- relation agreement **-1.82 pp**
- relation JS **+0.071393**

## Frozen interpretation

**Case B — correctness remains/improves, but stability gain is still weak.**

The per-query gate is mechanically real:
- alpha varies across DEV queries;
- selected alpha range is approximately **0.0938–0.1137**;
- paraphrase correctness improves;
- canonical correctness is retained/slightly improved;
- question-swap discrimination improves.

But the preregistered stability question is not solved:
- selected-choice agreement does not improve;
- fused JS becomes slightly worse;
- relation agreement/JS regress.

Therefore a gate trained only by gold CE learns local correctness calibration, but does not learn the causal condition “pairwise residual is reliability-improving across views.”

Next family: **S62 TRAIN-only reliability-supervised adaptive gate**, where the gate receives an explicit TRAIN-only target derived from whether bounded pairwise intervention improves paired-view correctness/stability without damaging correctness.

No S61 feature sweep, hidden layer, alpha sweep, objective retry, gate regularizer retrofit, gradient coupling, second DEV, selector change, native retraining or external Laya/Jev evaluation is authorized.
