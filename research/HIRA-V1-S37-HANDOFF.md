# HIRA V1 S37 handoff — to S38 Full Bilinear Native Query-Signature Readout

S37 is frozen as:

`HIRA_V1_S37_MATCHED_QUERY_GATED_DEV_COMPLETE`

Preregistered interpretation:
**Case C — diagonal query-signature interaction is insufficient.**

## Canonical evidence

A0:
- run `37078578256`
- artifact `11257573101`
- digest `sha256:adbff1f2af80978615f437d2350f4dd132905a42aeb18b39225c7d50fc0c31e6`

Fresh matched DEV:
- run `37079989756`
- artifact `11259139433`
- digest `sha256:1f6fbd32881f3ac01fae93250a1407ee519adab6a8f516cdf6564174203661af`
- scientific head `fc88dc9f8d8a4e94a8ef04c1b9c974b18f1054f7`

Control selected epoch 21:
- fused canonical/paraphrase **0.5208333 / 0.4348958**
- relation canonical/paraphrase **0.3411458 / 0.28125**
- signature cosine/margin **0.8558237 / 0.1971761**
- gates **15/23**

Treatment selected epoch 20:
- fused canonical/paraphrase **0.5182292 / 0.4244792**
- relation canonical/paraphrase **0.34375 / 0.2916667**
- signature cosine/margin **0.8502296 / 0.1887375**
- gates **15/23**

## Residual

S36 ruled out one global linear correctness direction.
S37 ruled out one query-conditioned **diagonal** bilinear direction.

The direct relation gains in S37 are only:
- canonical +0.0026042
- paraphrase +0.0104167
- canonical margin +0.0030691
- paraphrase margin +0.0022843

while both fused wording accuracies regress.

The next test should therefore add **cross-coordinate query-signature interactions**, not another diagonal vector or arbitrary MLP.

## S38 hypothesis

**Full Bilinear Native Query-Signature Correctness Readout**

Shared base:
- exact S35 native 256D relation representation/signature
- exact S17 primary/fusion/optimizer shell
- same native question-summary definition as S37
- no projected relation path

Control:
- exact native relation logits.

Treatment:
- `q_hat = L2(masked_mean(question_tokens), eps=1e-12)`
- one shared `W in R^(256x256)`
- `residual_k = signature_k^T W q_hat`
- `logit_k = native_logit_k + residual_k`
- W initialized exactly zero
- no bias
- no MLP/nonlinearity
- no option/domain/K-specific parameters
- residual scale 1.0

Added trainable params:
**65,536**.

Treatment total trainable surface:
**114,688**.

## Required S38-A0

Before fresh DEV:
- exact zero-init relation/signature/primary/fused identity
- selected-choice identity 1.0
- exact added params 65,536 / treatment total 114,688
- full-matrix relation gradient finite and nonzero at zero init
- primary -> bilinear matrix gradient exactly zero
- native relation -> shared projection direct gradient zero
- primary -> projection gradient nonzero
- relation -> LoRA gradient nonzero
- off-diagonal entries receive nonzero gradient
- query intervention changes nonzero-W residual
- signature intervention changes nonzero-W residual
- logical-option permutation equivariance
- question-token permutation and masked-padding invariance
- arbitrary K
- projection independence
- checkpoint/probability/full-K/state-once mechanics

Use wholly fresh S38 rows.
No S37 rows.
One DEV only.
No rank/factorization/regularization/LR retry after exposure.
No Laya/Jev benchmark before confirmed DEV_READY.
