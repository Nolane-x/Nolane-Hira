# HIRA V1 S36 handoff — to S37 Query-Gated Native Signature Readout

S36 is frozen as:

`HIRA_V1_S36_MATCHED_READOUT_DEV_COMPLETE`

Preregistered interpretation:
**Case C — one query-independent 256D correctness vector is insufficient.**

## Canonical evidence

A0:
- run `37025445955`
- artifact `11235297229`

Fresh matched DEV:
- run `37027342390`
- artifact `11237776679`
- digest `sha256:2f3c51e409cdcc94774c2b55dd3a0ca63d2f480a040cafa7a5baede2eadb0044`
- scientific head `28a28324ccc9941139f8983243ef741894f26658`

Control selected epoch 20:
- fused canonical/paraphrase 0.4557292 / 0.4270833
- relation canonical/paraphrase 0.3177083 / 0.2864583
- signature cosine/margin 0.9126326 / 0.1927358
- 16/23 gates

Treatment selected epoch 23:
- fused canonical/paraphrase 0.5208333 / 0.4218750
- relation canonical/paraphrase 0.3385417 / 0.2916667
- signature cosine/margin 0.9218181 / 0.1975873
- 16/23 gates

## Residual

The global readout substantially improves canonical fused margin, relation agreement and several stability metrics, but direct relation correctness barely changes and fused paraphrase accuracy does not improve.

The next hypothesis must change the **conditioning of the readout**, not merely its scale.

## S37 hypothesis

**Query-Gated Native Signature Correctness Readout**

Shared:
- exact S35 native 256D relation operator/signature
- exact S17 primary/fusion/optimizer shell
- same LoRA/projection surface
- no projected relation path

Control:
- exact native relation logits.

Treatment:
1. masked-mean the native A13 question tokens;
2. L2-normalize that 256D query summary with fixed epsilon;
3. compute `feature_k = signature_k * query_summary`;
4. compute residual `feature_k @ w`;
5. add residual to the native relation logit.

Frozen:
- `w in R^256`
- zero initialization
- residual scale 1.0
- no bias
- no MLP/nonlinearity
- no option-specific/domain-specific/K-specific parameters
- added parameters exactly 256
- treatment total trainable surface 49,408

## Required S37-A0

Must prove:
- exact zero-init identity to native control
- exact 256 added params / 49,408 total
- relation -> readout gradient nonzero
- primary -> readout gradient zero
- native relation -> shared projection direct gradient zero
- query intervention changes the nonzero-readout residual
- option permutation equivariance
- question-token permutation invariance under masked mean
- masked-padding invariance
- arbitrary K
- projection independence
- checkpoint/probability/full-K/state-once mechanics

Use wholly fresh S37 authority.
No S36 rows.
No S37-A0 rows in TRAIN/DEV.
One DEV only.
No post-DEV scale/bias/normalization/rank/MLP retry.
No Laya/Jev benchmark before a confirmed DEV_READY candidate.
