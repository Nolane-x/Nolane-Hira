# HIRA V1 S29 closure — Final-Block Attention+FFN LoRA Semantic Adaptation

Status: **CLOSED — DEV FAIL / FULL-BLOCK ADAPTATION DOES NOT BEAT THE SEMANTIC BOTTLENECK**

Issue: #241  
PR: #242

## Authority

Canonical A0:
- run `36883766054`
- artifact `11172806806`
- digest `sha256:1f05cab39d88b481968d27c7260b9a7192f184bf650863c5fc7874c2d2f53f9f`
- authority head `ed3d7b9c64e6a3bf75fa62f577901c41b51dfcdd`
- outcome `HIRA_V1_S29_A0_FULL_BLOCK_LORA_READY`

Fresh TRAIN/DEV:
- run `36884997207`
- artifact `11175525933`
- artifact digest `sha256:a0047da70d0708c51aa9a23f5fe1338cfb1f434f88e5d80e2e322bfd9eaa51ec`
- scientific head `6a479d20c727031011c734d7f4ba34498f526cf9`
- outcome `HIRA_V1_S29_FULL_BLOCK_LORA_DEV_FAIL`
- selected epoch **12**
- checkpoint SHA256 `7abfe0209d1c0a8166a365c58c0e1193dafe5ceb33a0ad8a4e961d34b3eba426`

No second DEV run.  
No post-DEV tuning.  
No sealed confirmation.  
No multilingual probe.  
No matched Laya/Jev evaluation.

## Frozen intervention

S29 resets to the S17 semantic shell and changes only the final A13 adaptation surface.

Kept:
- S13 relation expert / canonicalizer
- S14 equal standardized full-K fusion
- S15 primary-loss relation-logit detach
- S17 relation-priority norm-balanced gradient handling
- S17 loss partition and coefficients
- shared bias-free 256→128 projection
- state-once / full-K / opaque option IDs
- original A13 weights frozen
- HIRACore frozen

Changed:
- existing final attention LoRA: **16,384**
- added final FFN intermediate LoRA: **10,240**
- added final FFN output LoRA: **10,240**
- total A13 LoRA: **36,864**
- shared projection: **32,768**
- exact trainable total: **69,632**

Rank **8**, alpha **8.0**, dropout **0.0**.

## A0 result

A0 proves the new surface is real and initially identity-preserving:

- A13 token output max abs vs S17: **0**
- A13 pooled output max abs: **0**
- raw / relation / fused / signature max abs: **0**
- exact decision-logit identity rate: **1.0**
- exact selected-choice identity rate: **1.0**
- six-module checkpoint roundtrip: PASS

Real fresh gradients:
- attention LoRA B max L1: **8.3233375549**
- FFN intermediate B L1: **2.3468730450**
- FFN output B L1: **0.9416253567**
- projection L1: **358.0661621094**

Mechanics:
- full-K PASS
- state-once PASS
- option-order flip **0**
- probability-mass error **1.1920929e-7**

S29 therefore did not fail because the FFN adapters were dead or incorrectly wired.

## Selected DEV — epoch 12

Fused:
- canonical accuracy **0.5807291667**
- paraphrase accuracy **0.3385416667**
- paired both-correct **0.2916666667**
- question-swap change **0.5104166667**
- cross-view selected-choice agreement **0.4661458333**
- cross-view mean JS **0.0421490973**
- canonical signed margin **+0.0490893635**
- paraphrase signed margin **-0.2587409501**

Primary:
- canonical accuracy **0.3880208333**
- paraphrase accuracy **0.3203125**
- cross-view agreement **0.59375**

Relation:
- canonical accuracy **0.4348958333**
- paraphrase accuracy **0.3385416667**
- canonical signed margin **-0.0679718368**
- paraphrase signed margin **-0.6121830083**
- cross-view agreement **0.5963541667**

Relation signature:
- same-option cosine **0.8419482509**
- same-vs-strongest-wrong margin **0.1245272141**

Mechanics:
- full-K PASS
- option-order flip **0**
- probability-mass error **1.7881393e-7**
- relation delta **0**
- state-once PASS
- exact trainable surface **69,632**
- original A13 frozen
- HIRACore frozen

## Gate result

PASS **13 / 22**.

FAIL **9 / 22**:
- fused canonical >= **0.85**
- paired >= **0.75**
- question-swap >= **0.80**
- fused agreement >= **0.95**
- fused canonical margin >= **0.15**
- relation canonical >= **0.80**
- relation canonical margin >= **0.15**
- same-option signature cosine >= **0.90**
- signature discrimination margin >= **0.15**

The fused-JS gate and all mechanical/capacity/freeze gates pass.

DEV_READY is not authorized.

## Best observed DEV values across 24 epochs

- fused canonical **0.59375**, epoch 7
- fused paraphrase **0.4270833333**, epoch 4
- paired both-correct **0.2916666667**, epoch 12
- question-swap **0.609375**, epoch 24
- fused agreement **0.6171875**, epoch 4
- minimum fused JS **0.0159848178**, epoch 1
- fused canonical margin **+0.1792202912**, epoch 7
- fused paraphrase margin best **-0.0951350583**, epoch 22
- relation canonical **0.4609375**, epoch 10
- relation paraphrase **0.4348958333**, epoch 4
- relation canonical margin best **-0.0433199493**, epoch 9
- relation paraphrase margin best **-0.0495568141**, epoch 1
- relation agreement **0.7734375**, epoch 5
- same-option signature cosine **0.9519436409**, epoch 5
- signature discrimination margin **0.1455361911**, epoch 6
- raw primary canonical **0.4479166667**, epoch 24
- raw primary paraphrase **0.3515625**, epoch 17

No epoch reaches the semantic gate set.

Most importantly:
- relation canonical margin never becomes positive;
- relation paraphrase margin never becomes positive;
- paraphrase fused margin remains negative at every epoch.

## Optimization dynamics

Epoch 1 → 24:
- total TRAIN loss **1.7388281996 → 0.8126386789**
- decision loss **1.4866003816 → 0.6539563586**
- relation binding loss **1.3859392181 → 0.7604993954**
- canonicalization loss **0.2805088044 → 0.0865243498**
- fused consistency JS **0.0147167225 → 0.0082274604**
- LoRA-B norm **0.6023294926 → 3.3483345509**

Mean shared-gradient conflict rate:
- **0.3020833333**

This is not an optimizer crash.

However, DEV transport quality peaks early and then degrades:
- same-option signature cosine peaks **0.9519436409** at epoch 5;
- by epoch 24 it falls to **0.5464356343**.

At the same time TRAIN objectives continue improving.

This is direct evidence of representation over-specialization / cross-view drift under the joint attention+FFN adaptation surface.

## Scientific interpretation

S29 falsifies:

**“adding final-block FFN LoRA on top of the S17 attention-LoRA surface is sufficient to solve the remaining semantic bottleneck.”**

The additional nonlinear capacity is active, but the combined surface does not produce stable fresh semantic discrimination.

The result also prevents a simplistic conclusion that the FFN itself is useless. S29 changes two adaptation families jointly:
- attention LoRA continues adapting;
- FFN LoRA is added and adapts simultaneously.

Therefore S29 cannot separate:
1. FFN adaptation is intrinsically weak;
2. FFN adaptation is useful but conflicts with simultaneous attention adaptation;
3. attention-only remains the better adaptation surface.

The next court should isolate this variable directly on the same fresh authority rather than stack another heuristic loss.

## Closed directions

Do not create S29b by:
- changing LoRA rank or alpha
- adding earlier encoder layers
- lowering FFN LR after seeing S29 DEV
- freezing FFN at an exposed epoch
- retrying seed / LR / epochs / batch
- changing S17 loss coefficients
- changing fusion or selector
- weakening gates
- adding frozen-base drift loss as a nominally new idea; v0 M3 already tested an English frozen-A13 anchor term

## Next controlled direction

**S30 — Matched Attention-vs-FFN Adaptation Surface Court**

Use one wholly fresh S30 authority and train two arms under the exact same S17 semantic shell.

Arm A — attention-only:
- final Q/K/V/attention-output LoRA
- rank 8 / alpha 8 / dropout 0
- LoRA **16,384**
- projection **32,768**
- total **49,152**

Arm B — FFN-only:
- final intermediate.dense LoRA **10,240**
- final output.dense LoRA **10,240**
- no attention LoRA
- rank 8 / alpha 8 / dropout 0
- LoRA **20,480**
- projection **32,768**
- total **53,248**

Keep identical:
- frozen original A13
- frozen HIRACore
- S13 relation expert
- S14 fusion
- S15 detach
- S17 losses
- S17 norm-balanced rule
- optimizer / epochs / batch
- TRAIN rows
- DEV rows
- batch order
- selector
- semantic gates

The court is paired by construction. It is intended to answer whether FFN adaptation has independent value or whether the S29 failure is explained by joint sublayer adaptation interference.

No third full-block arm is needed in S30; S29 already closes that candidate family on fresh authority.

Production-ready remains false.  
Laya/Jev parity remains unestablished.
