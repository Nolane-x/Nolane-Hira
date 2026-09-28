# HIRA V1 S12 closure — Relation-Structured Role/Value Binding

Status: **CLOSED — DEV FAIL / SCIENTIFIC NEGATIVE RESULT**

Issue: #198  
PR: #199  
Branch: `feat/hira-v1-s12-relation-binding`

## 1. Authority

A0:
- run: `36497191159`
- artifact: `11004450349`
- artifact digest: `sha256:62340525b36377e08d5ec481d852dd9422264a43579a0347c7d126943180bbf5`
- outcome: `HIRA_V1_S12_A0_IDENTITY_READY`

Fresh TRAIN/DEV:
- run: `36498016327`
- artifact: `11004453797`
- artifact digest: `sha256:f1939bac3e772a969e7e43d0dc78c163333b7cce1baa16922e929877c04950c9`
- outcome: `HIRA_V1_S12_RELATION_BINDING_DEV_FAIL`
- selected epoch: **18**
- selected checkpoint SHA256: `60bddcd73e21220ea4131bc3f1ca5f55ba8f18c5277152902a2ca94f7b661869`

No post-DEV tuning was performed.
No sealed confirmation was opened.
No multilingual probe was opened.
No Laya/Jev benchmark was reopened.

## 2. Frozen model surface

S12 kept the exact S8-S11 optimization surface:
- A13 final-layer attention LoRA: **16,384 params**
- shared bias-free 256->128 projection: **32,768 params**
- total trainable: **49,152 params**
- original A13 trainable: **0**
- HIRACore trainable: **0**
- learned downstream scorer params: **0**
- relation-binding added params: **0**

The structural change was parameter-free:
- no fixed token-distance window;
- query-conditioned state and option role anchors;
- role-relative residual vectors;
- explicit state-token <-> option-token direct + relation pair scoring.

## 3. Selected DEV result

Primary decision:
- canonical accuracy: **0.28125**
- paraphrase accuracy: **0.3020833333**
- canonical paired both-correct: **0.109375**
- canonical question-swap choice-change: **0.6302083333**
- cross-view selected-choice agreement: **0.3802083333**
- cross-view mean JS: **9.2741076e-09**

Relation binding:
- canonical relation-binding accuracy: **0.4010416667**
- paraphrase relation-binding accuracy: **0.5390625**
- canonical signed gold-vs-max-wrong margin: **-0.7074106541**
- paraphrase signed margin: **-0.1640277983**
- relation-binding cross-view selected-choice agreement: **0.4036458333**
- canonical mean relation-binding loss: **1.6258039623**
- paraphrase mean relation-binding loss: **1.1904561420**

Role localization:
- canonical normalized role entropy: **0.7658492724**
- canonical role max weight: **0.1778150188**
- paraphrase normalized role entropy: **0.6679870685**
- paraphrase role max weight: **0.3067472676**

Mechanical invariants:
- option-order flip: **0.0**
- max probability mass error: **1.7881393433e-07**
- full-K: PASS
- state-once: PASS
- relation delta: **0.0**

## 4. Training dynamics

The relation-binding auxiliary was highly learnable on TRAIN:
- TRAIN mean binding loss epoch 1: **1.364938**
- TRAIN mean binding loss epoch 24: **0.367593**

Fresh DEV did not follow.

Observed fresh-DEV extrema:
- best canonical relation-binding accuracy: **0.4322916667** at epoch 2;
- best canonical signed relation margin: **-0.094713** at epoch 1;
- best canonical primary accuracy: **0.3072916667** at epochs 17 and 20;
- best canonical paired both-correct: **0.109375** at epoch 18;
- best question-swap change: **0.6510416667** at epoch 22.

As TRAIN relation loss continued to improve, the fresh canonical signed margin became substantially more negative. The selected checkpoint also shows a large wording-view asymmetry:
- canonical relation accuracy **0.4010** vs paraphrase **0.5391**;
- canonical margin **-0.7074** vs paraphrase **-0.1640**.

This is evidence that relation geometry remains strongly wording/template dependent.

## 5. Frozen gate result

Passed:
- cross-view mean JS <= 0.05;
- option-order flip <= 0.02;
- probability-mass error <= 1e-6;
- full-K;
- relation delta = 0;
- state-once TRAIN/DEV;
- exact 49,152 trainable surface;
- A13 frozen;
- HIRACore frozen.

Failed:
- canonical accuracy >= 0.85;
- paired both-correct >= 0.75;
- question-swap change >= 0.80;
- cross-view choice agreement >= 0.95;
- canonical relation-binding accuracy >= 0.80;
- canonical relation-binding signed margin >= 0.15.

Therefore `HIRA_V1_S12_RELATION_BINDING_DEV_READY` is not authorized.

## 6. Preregistered interpretation

S12 matches primarily **Outcome C + Outcome E**.

### Outcome C — role anchors improve but relation pairs stay wrong

Compared with the fresh A0 baseline, role concentration becomes stronger, but correct signed semantic separation is not established. Relation accuracy remains weak and the canonical signed margin is strongly negative.

### Outcome E — TRAIN relation binding improves but fresh DEV collapses

TRAIN relation-binding loss falls by roughly 73%, while fresh DEV relation accuracy/margin stagnate or regress and wording-view agreement remains weak.

The large canonical/paraphrase asymmetry strengthens the conclusion that the learned relation geometry is not yet invariant to semantically equivalent wording.

## 7. Scientific conclusion

S12 rejects the hypothesis that role-relative residual geometry plus best explicit token-pair matching is sufficient, by itself, to generalize role/value binding across fresh lexical/template variation under the current 49,152-parameter surface.

The evidence does **not** justify:
- sharper role temperature;
- a different direct/relation mixing weight;
- a different seed/LR;
- more parameters;
- reopening Laya/Jev.

The strongest next hypothesis is explicit **cross-view relation canonicalization/invariance**: semantically equivalent state/question wording views must map to the same role/value relation evidence rather than merely receiving independent CE supervision.

Production-ready remains false.
Laya/Jev parity remains unestablished.
