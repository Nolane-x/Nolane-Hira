# HIRA V1 S30 contract — Matched Attention-vs-FFN Adaptation Surface Court

Status: **OPEN / PREREGISTERED BEFORE S30-A0 EXPOSURE**

Issue: #243

Parent:
- S29 issue #241 / PR #242
- S29 outcome `HIRA_V1_S29_FULL_BLOCK_LORA_DEV_FAIL`
- merged main `c2b8c4afa3a99303e914dee484075d21e54255e9`

## 1. Scientific question

S29 jointly adapted the final A13 attention and FFN sublayers. The added FFN LoRA received real gradients and trained, but the combined surface failed fresh DEV and exhibited cross-view drift.

S30 isolates the adaptation surface with a matched two-arm court on the exact same fresh data.

> Under the same S17 semantic shell, is final-block FFN-only adaptation stronger or weaker than final-block attention-only adaptation?

S30 is a mechanism-localization court. It is not a rank search, learning-rate search, layer search or loss search.

## 2. Frozen shared shell

Both arms keep:
- W28 T0 projection initialization
- original A13 weights frozen
- HIRACore frozen
- S13 cross-view relation expert/canonicalizer
- S14 equal standardized full-K fusion, epsilon **1e-6**
- S15 fused-primary relation-logit detach
- S17 primary/relation loss partition
- S17 relation-priority norm-balanced gradient rule
- S17 loss coefficients and temperatures
- one shared bias-free 256→128 projection per arm
- state-once
- full-K
- opaque option IDs
- no learned downstream head/router/gate/calibrator

The arms differ only in which final A13 sublayer receives LoRA.

## 3. Arm A — attention-only

Exact S17 encoder adaptation surface:
- `attention.self.query`, 256→256: 4,096
- `attention.self.key`, 256→256: 4,096
- `attention.self.value`, 256→256: 4,096
- `attention.output.dense`, 256→256: 4,096
- attention LoRA subtotal: **16,384**
- projection: **32,768**
- exact trainable total: **49,152**

LoRA:
- rank **8**
- alpha **8.0**
- dropout **0.0**

## 4. Arm B — FFN-only

No attention LoRA.

Final block:
- `intermediate.dense`, 256→1024: **10,240**
- `output.dense`, 1024→256: **10,240**
- FFN LoRA subtotal: **20,480**
- projection: **32,768**
- exact trainable total: **53,248**

LoRA:
- rank **8**
- alpha **8.0**
- dropout **0.0**

## 5. Matched conditions

Both arms use:
- identical fresh TRAIN rows
- identical fresh DEV rows
- identical row order per epoch
- identical seed
- identical optimizer hyperparameters
- independent model/runtime instances
- independent LoRA tensors
- independent projection tensors initialized from the same W28 T0 projection
- identical selector
- identical semantic gates

No arm may inspect the other arm's DEV result before training completes.

## 6. S30-A0

A0 uses fresh S30-A0 rows and is diagnostic only.

Required zero-init identity:
- token outputs exact between arms
- pooled outputs exact between arms
- raw primary logits exact
- relation logits exact
- fused logits exact
- signatures exact
- selected choices exact

Required capacity:
- Arm A LoRA **16,384**
- Arm A projection **32,768**
- Arm A total **49,152**
- Arm B LoRA **20,480**
- Arm B projection **32,768**
- Arm B total **53,248**
- original A13 trainable **0** for both
- HIRACore trainable **0** for both
- frozen A0 runtime trainable **0** for both

Required real gradient court:
- Arm A at least one attention-LoRA B gradient nonzero
- Arm A projection gradient nonzero
- Arm B FFN intermediate B gradient nonzero
- Arm B FFN output B gradient nonzero
- Arm B projection gradient nonzero
- primary and relation gradient blocks finite/nonzero in both
- norm-balanced update finite in both

Required mechanics:
- full-K
- state-once
- option permutation
- probability mass <= **1e-6**
- relation delta **0**
- attention checkpoint roundtrip
- FFN checkpoint roundtrip

A0 semantic accuracy cannot tune S30.

## 7. Fresh matched TRAIN/DEV

Only after qualified A0 + frozen interpretation + exact-head CI + one-shot authorization.

Frozen authority:
- seed **51001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S30 domains
- K=4
- two state views
- two question views per semantic query
- two option semantic views
- epochs **24**
- batch **16**
- AdamW lr **2e-4**
- weight decay **0.01**
- grad clip **1.0**

Freshness:
- no exact S0-S29 exposed rows
- no S30-A0 rows
- no M5 final/confirmatory rows
- no W29-W34 sealed rows

## 8. Independent selection

Each arm uses the same frozen S17 lexicographic selector independently:
1. paired both-correct
2. fused canonical
3. relation canonical
4. relation canonical margin
5. fused canonical margin
6. question-swap
7. fused cross-view agreement
8. same-option signature cosine
9. signature discrimination margin
10. lower canonical decision loss
11. earlier epoch

No joint selector may mix epochs across arms.

## 9. Semantic gates

Each arm is evaluated independently against:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused JS <= **0.05**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination >= **0.15**
- option flip <= **0.02**
- probability mass <= **1e-6**
- full-K
- state-once
- exact arm capacity
- original A13 frozen
- HIRACore frozen
- fusion params zero
- relation delta zero

## 10. Matched interpretation rule

The primary comparison is paired on the same DEV rows.

Report raw metrics for both arms and deltas:
`FFN-only - attention-only`.

Do not create a scalar winner score.

Interpretation:
- if FFN-only materially improves the primary semantic endpoints coherently, FFN-only remains a viable encoder-adaptation family for a later fresh confirmation;
- if attention-only matches or exceeds FFN-only on the main semantic endpoints, close final-block FFN adaptation as a useful v1 direction;
- if results split by endpoint, close S30 as unresolved and do not tune either arm on the exposed DEV.

S30 itself does not authorize production readiness or Laya/Jev parity.

## 11. Stop rule

After S30 matched DEV exposure:
- no arm-specific rank/alpha/dropout change
- no arm-specific LR or schedule
- no extra epochs
- no seed retry
- no loss/fusion/selector change
- no gate weakening
- no second DEV run
- no earlier-layer adaptation
- no third full-block arm

Scientific FAIL or unresolved evidence is valid.
