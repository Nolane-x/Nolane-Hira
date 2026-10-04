# HIRA V1 S56 contract — Explicit Cross-View Decision Consistency

Status: **PREREGISTERED / NO S56-A0 EXPOSURE**

Issue: #295

Parent:
- S55 fresh run `37212962160`
- artifact `11307227880`
- digest `sha256:7059cf8477eb4bd775e89badb0c734a2d351afe2e1a1d23ceaf4f7189c7744e4`
- Case **C**
- S51–S55 correction/factorization family exhausted.

## Scientific question

> With architecture and private capacity fixed, can explicit consistency on final full-K decisions improve cross-view selected-choice stability without collapsing useful correctness?

## Frozen backbone

Both arms use the exact same S54 joint state-query-option private correction architecture:
- `JointStateQueryOptionPrivateCorrectionFork`
- correction params **114,688**
- interaction trainable params **0**
- identity params **0**
- total private trainable **114,688**
- one encoder/state-once
- full-K
- sealed S51 native authority
- immutable shared cache
- native trainability 0.

No S55 learned transform is retained. S55 showed that added relation-transform capacity did not solve stability. S56 isolates the **objective** rather than architecture.

## Controlled variable

Reference:
- existing private correctness objective only
- decision consistency coefficient **0.0**
- ordering consistency coefficient **0.0**

Treatment:
- exact same architecture, initialization, rows, optimizer, selector
- existing private correctness objective
- standardized full-K decision consistency coefficient **0.10**
- pairwise ordering consistency coefficient **0.05**

No added trainable parameters.

## Standardized full-K decision consistency

For logits `z` over K options:

`std(z) = (z - mean(z)) / sqrt(mean((z-mean(z))^2) + 1e-6)`

For canonical/paraphrase paired views:
- `p_c = softmax(std(z_c))`
- `p_p = softmax(std(z_p))`
- `L_dec = JS(p_c || p_p)`

Frozen epsilon: **1e-6**.

Properties required:
- zero for identical logits;
- invariant to shared offset;
- invariant to positive scale within tolerance;
- finite for flat logits;
- option-permutation invariant under matched permutation.

## Pairwise ordering consistency

For each unordered pair `i < j`:
- `m_c = std(z_c)_i - std(z_c)_j`
- `m_p = std(z_p)_i - std(z_p)_j`

Frozen active threshold:
`tau_order = 0.25`

Pair active iff:
`max(|m_c|, |m_p|) >= 0.25`

Frozen margin floor:
`m_floor = 0.05`

Signed shared strength:
`s = sign(m_c * m_p) * min(|m_c|, |m_p|)`

Active-pair penalty:
`relu(0.05 - s)`

Mean over active pairs; if no active pairs, ordering loss = 0.

Properties required:
- zero for sufficiently separated matching ordering;
- positive on controlled sign flip;
- inactive below threshold;
- matched-permutation invariant.

## Treatment auxiliary

`L_aux = 0.10 * L_dec + 0.05 * L_order`

Reference auxiliary:
exactly **0**.

## Anti-collapse

Consistency alone can be minimized by flat/uniform predictions.

Therefore S56 reports:
- mean fused entropy;
- mean top1-top2 probability gap;
- mean standardized logit RMS;
- existing correctness, margin and agreement metrics.

A flat/uniform decision may have zero consistency loss but:
- top1-top2 gap = 0;
- standardized RMS = 0;
- correctness/margin gates remain authoritative.

No entropy auxiliary is added.

## Ownership

Auxiliary gradients may update only S54 private correction parameters.

Must be zero/absent for:
- persisted native runtime
- immutable cache
- query-free identity (zero params)
- interaction operator (zero params).

## Required S56-A0

Architecture:
- reference/treatment private trainable **114,688** each
- added params **0**
- bit-identical initialization
- native trainable params 0
- second encoder false.

Decision consistency:
- identical logits => exact/nearly-zero loss
- controlled disagreement => positive loss
- shared offset invariance
- positive scale invariance
- flat logits finite
- matched option-permutation invariance.

Ordering:
- matching confident ordering => zero
- sign flip => positive
- below-threshold pair => inactive
- matched option-permutation invariance.

Gradient ownership:
- reference auxiliary exactly zero
- treatment auxiliary gradient reaches correction params on controlled probe
- no native/cache gradient.

Anti-collapse:
- uniform probe reports top1-top2 gap 0
- uniform standardized RMS 0
- not interpreted as success.

Full-K:
- K=3/7/255
- probability mass <=1e-6
- one encoder/state-once.

## Fresh S56 authority

Intended:
- seed **77001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh domains
- K=4
- private epochs **24**
- batch **16**
- one DEV only.

Parent native:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`.

## Prohibited

No:
- coefficient sweep
- threshold sweep
- margin-floor sweep
- entropy auxiliary
- architecture/capacity change
- native retraining
- selector change
- retry
- gate weakening
- second S56 DEV
- external Laya/Jev evaluation.

Scientific failure is valid.
