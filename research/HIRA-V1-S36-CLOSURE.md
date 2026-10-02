# HIRA V1 S36 closure — Native Signature Linear Correctness Readout

Status: **CLOSED — MATCHED DEV COMPLETE / PREREGISTERED CASE C / SINGLE GLOBAL LINEAR READOUT INSUFFICIENT**

Issue: #255
PR: #256

## Canonical authority

A0:
- run `37025445955`
- artifact `11235297229`
- digest `sha256:33c0961945b0f4cb3c56022107fa0fe726481e6e5f6d5602445e78220da78070`
- authority head `4ca77a3659dfa1b3f2676a6bff41cb4afd78ed0b`
- outcome `HIRA_V1_S36_A0_NATIVE_SIGNATURE_READOUT_READY`

Fresh matched TRAIN/DEV:
- run `37027342390`
- artifact `11237776679`
- artifact digest `sha256:2f3c51e409cdcc94774c2b55dd3a0ca63d2f480a040cafa7a5baede2eadb0044`
- scientific head `28a28324ccc9941139f8983243ef741894f26658`
- outcome `HIRA_V1_S36_MATCHED_READOUT_DEV_COMPLETE`

No second DEV.
No post-DEV readout tuning.
No scale/bias/nonlinearity/rank retry.
No projected/native mixing.
No sealed confirmation.
No multilingual probe.
No Laya/Jev evaluation.

## Frozen matched setup

TRAIN 768 / DEV 192 across 12 wholly fresh S36 domains.
Seed 57001, 24 epochs, batch 16, AdamW 2e-4, weight decay .01, grad clip 1.0.

Both arms:
- exact S35 native 256D relation representation
- exact S17 primary path/fusion/optimizer shell
- A13 attention LoRA 16,384
- shared primary projection 32,768
- original A13 frozen
- HIRACore frozen
- identical rows and per-epoch order
- independent runtime and optimizer state

Control:
- exact S35 native relation logits/signatures
- trainable surface 49,152

Treatment:
- native logits + `signature @ w`
- one shared zero-init `w in R^256`
- no bias/nonlinearity/learned scale
- added params 256
- trainable surface 49,408

## A0 mechanism result

S36-A0 passed the intended isolation:
- zero-init relation/signature/primary/fused differences: exactly 0
- selected-choice identity rate: 1.0
- relation -> readout gradient L1: 0.0340911485
- primary -> readout gradient L1: 0
- native relation -> shared projection direct gradient L1: 0
- primary -> projection gradient L1: 1318.4007568359
- relation -> LoRA gradient L1: 0.2824096456
- native path remains exactly projection-independent
- logical-option permutation equivariance PASS
- arbitrary K=3 and K=7 PASS
- checkpoint/probability/full-K mechanics PASS

The DEV result is therefore not a wiring failure.

## Selected DEV — control

Selected epoch: 20
Checkpoint: `a97c47967a6ae9e70f50f524932faa2af7bb65603aa291f53d265daa706f168c`

- fused canonical/paraphrase: 0.4557292 / 0.4270833
- paired both-correct: 0.2239583
- question-swap: 0.5729167
- fused agreement / JS: 0.6484375 / 0.0215314
- fused canonical/paraphrase margin: -0.0786602 / -0.2148795
- raw primary canonical/paraphrase: 0.4479167 / 0.4114583
- relation canonical/paraphrase: 0.3177083 / 0.2864583
- relation canonical/paraphrase margin: -0.0668640 / -0.0842782
- relation agreement: 0.5052083
- signature cosine / discrimination margin: 0.9126326 / 0.1927358
- gates: 16/23 PASS
- DEV_READY: false

## Selected DEV — treatment

Selected epoch: 23
Checkpoint: `6c13e92407fcbb09ea7f364d2d352e1221931056596cfeedce614b5042fd0f7f`

- fused canonical/paraphrase: 0.5208333 / 0.4218750
- paired both-correct: 0.2604167
- question-swap: 0.5989583
- fused agreement / JS: 0.6770833 / 0.0176693
- fused canonical/paraphrase margin: +0.0731877 / -0.0607783
- raw primary canonical/paraphrase: 0.4557292 / 0.4453125
- relation canonical/paraphrase: 0.3385417 / 0.2916667
- relation canonical/paraphrase margin: -0.0660571 / -0.0808898
- relation agreement: 0.6875000
- signature cosine / discrimination margin: 0.9218181 / 0.1975873
- gates: 16/23 PASS
- DEV_READY: false

## Treatment minus control

Positive:
- fused canonical +0.0651042
- paired +0.0364583
- question-swap +0.0260417
- fused agreement +0.0286458
- fused JS -0.0038621
- fused canonical margin +0.1518480
- fused paraphrase margin +0.1541012
- raw primary canonical/paraphrase +0.0078125 / +0.0338542
- relation agreement +0.1822917
- signature cosine +0.0091854
- signature discrimination +0.0048515

Weak correctness movement:
- relation canonical accuracy +0.0208333
- relation paraphrase accuracy +0.0052083
- relation canonical margin +0.0008069
- relation paraphrase margin +0.0033884

Regression:
- fused paraphrase accuracy -0.0052083

## Preregistered interpretation

S36 is **Case C — little/no transferable correctness gain from one shared linear direction**.

The readout changes several fused and consistency endpoints, but the direct relation correctness quantities that S36 was designed to localize barely move. The only fused accuracy gain is canonical-side; paraphrase accuracy does not improve.

Therefore the evidence does not support a coherent global correctness vector in native signature space.

This does not show that the native signature lacks correctness information. It shows that a single query-independent direction is insufficient.

## S36 closure

Close the global one-vector readout family on exposed S36 DEV.

Do not:
- tune residual scale
- add bias or nonlinearity on S36 rows
- retry rank/MLP
- change readout LR
- mix projected/native paths
- rerun seed/LR/epoch/batch
- weaken gates
- run a second DEV

## Next controlled direction

**S37 — Query-Gated Native Signature Correctness Readout**

Question:
> Is the missing correctness direction query-dependent rather than globally shared?

Keep the same native 256D relation representation.

Control:
- exact S35 native relation logits.

Treatment:
- build a fixed normalized native query summary from masked question tokens;
- form per-option feature `signature_k * query_summary`;
- apply one shared zero-init vector `w in R^256`;
- residual `(signature_k * query_summary) @ w`;
- no bias, MLP, rank expansion, option-ID/domain/K-specific parameters;
- added trainable params exactly 256.

This preserves arbitrary-K and option-permutation equivariance while testing query-conditioned correctness with the same parameter budget as S36.

Production-ready remains false.
Laya/Jev parity remains unestablished.
