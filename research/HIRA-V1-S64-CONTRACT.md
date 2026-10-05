# HIRA V1 S64 contract — Matched-Capacity Context-Projected Reliability Gate

Status: **FROZEN BEFORE A0**

Issue: #313  
PR: #314

Parent:
- S63 merged main `4fcb58f0c35e2feca841b08f6f560c092ae600fd`
- scientific run `37314015283`
- artifact `11347706059`
- digest `sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e`
- verdict **Case B**.

## Scientific variable

Reference:
- exact S63 learned decision-surface set reliability gate
- 61 trainable params.

Treatment:
- matched 61-param contextual reliability gate
- option source = detached S59 512D identity+state-query-option representation
- fixed parameter-free Rademacher projection 512→4.

Both:
- exact same learned tensor shapes
- bit-identical learned initialization
- 4→8 tanh option encoder
- mean+max pooling
- exact four S61 scalar global surface features
- 20D final reliability representation
- exact S62 TRAIN-only reliability BCE target
- alpha probe 0.35
- target tolerance 1e-8
- bounded residual alpha <=0.35
- same optimizer/LR/weight decay
- same correction/head trajectory
- same S17 selector.

Treatment minus reference trainable params: **0**.

## Fixed contextual projection

- shape [4,512]
- seed 64064
- CPU generator
- entries +/-1/sqrt(512)
- non-trainable persistent buffer
- no data fit
- no TRAIN/DEV statistics.

Context option normalization:
- center over 512 features
- RMS normalize with epsilon 1e-6
- fixed projection.

## Materiality

Stability success requires both:
- agreement delta >= +1.00 pp
- JS delta <= -0.002.

Correctness retained requires all:
- canonical >= -0.50 pp
- paraphrase >= -0.50 pp
- paired both-correct >= -0.50 pp
- question-swap >= -0.50 pp.

No threshold may change after DEV.

## Fresh authority intent

- seed 85001
- TRAIN 768
- DEV 192
- 12 fresh domains
- exact S63 overlap 0
- K=4
- 24 epochs
- one DEV only.

External Laya/Jev evaluation remains closed.
