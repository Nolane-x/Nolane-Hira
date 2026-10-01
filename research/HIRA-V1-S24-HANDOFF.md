# HIRA V1 S24 handoff — to S25 Decoupled Expert Projection Surfaces

S24 is frozen as:

`HIRA_V1_S24_RELIABILITY_FUSION_DEV_FAIL`

Canonical authority:
- A0 run `36793018851`, artifact `11133405764`
- A0 digest `sha256:4eed0771f2e2037a9cccbee547ffc7831140e53781256ff2b4dd9ce1c67810b3`
- TRAIN/DEV run `36819995046`
- artifact `11143228027`
- artifact digest `sha256:e6437365d3b1f94bffa390f676d6516a8839cb843f4e7fca54f926ac89cf1775`
- selected epoch **22**
- checkpoint `b4fc574c2b1058172642bd10b41c4e18b8404952a1e08fa55d12b8c7a872eb08`

Selected DEV:
- fused canonical **0.390625**
- fused paraphrase **0.3151041667**
- paired **0.0572916667**
- question-swap **0.2239583333**
- fused agreement **0.4713541667**
- fused JS **0.0962312045**
- fused canonical margin **-0.4352511764**
- primary canonical/paraphrase **0.3619791667 / 0.2890625**
- relation canonical/paraphrase **0.4010416667 / 0.296875**
- relation canonical margin **-0.1124879544**
- relation paraphrase margin **-0.2615829383**
- signature cosine **0.8084561278**
- signature margin **0.0858259757**

Mechanics:
- exact physical surface **49,152**
- fusion added params **0**
- full-K PASS
- state-once PASS
- relation isolation PASS
- option permutation PASS
- probability mass PASS
- mean optimizer conflict **0.2951388889**

## Stop rule applied

Do **not** retry S24 with a modified top2-gap formula, epsilon, temperature, clipping, seed, LR, selector, or gate.

The S24 top2-gap reliability rule is closed.

## Why the next track changes representation

S24 mechanics are healthy but both experts are weak on the fresh authority:
- primary canonical **36.20%**
- relation canonical **40.10%**
- fused canonical **39.06%**

So another post-hoc blend is not the highest-value next experiment.

Prior S21-S23 evidence also showed persistent competition on the shared trainable surface. Optimizer priority did not solve it.

## S25 target

**Decoupled Expert Projection Surfaces**

Freeze before A0:
- shared A13 encoder and LoRA;
- shared LoRA trainable surface **16,384**;
- private primary projection **32,768**;
- private relation projection **32,768**;
- total physical trainable surface **81,920**;
- original A13/HIRACore frozen;
- no learned router/gate/calibrator;
- revert fusion to S14 equal standardized **0.5 / 0.5** to isolate representation;
- relation expert retains its canonicalization objective;
- primary expert retains role-gated content objective;
- neutral-bisector balancing applies only to shared LoRA gradients;
- private primary projection receives only primary block gradient;
- private relation projection receives only relation block gradient;
- exact gradient ownership must be proved in A0.

Required A0 proofs:
- exact parameter ownership/counts;
- private-surface initialization identity;
- zero cross-private gradients;
- shared-LoRA gradients from both expert blocks;
- equal S14 fusion mechanics;
- option permutation/full-K/state-once;
- checkpoint roundtrip/frozen replay;
- no hidden learned downstream state.

Use wholly fresh S25 A0/TRAIN/DEV authority.

No S24 DEV rows may enter S25 TRAIN/DEV.
No Laya/Jev until DEV_READY.
