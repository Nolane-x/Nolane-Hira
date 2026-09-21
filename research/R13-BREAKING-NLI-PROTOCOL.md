# R13 Breaking NLI protocol

Status: **frozen before evaluation**.

Model:
- exact R12-selected HIRA head SHA-256:
  `0ca95572399d15717e1069545439165e46083c58afea8707cbbadeca67ad3b86`
- A13 revision:
  `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- no weight update, calibration or threshold tuning.

Dataset:
- repository: `BIU-NLP/Breaking_NLI`
- revision: `8a7658c1ce6b732f4e8af3b06560f1a13b8b18b0`
- zip blob SHA: `9fcd602891e4f844d53d611924915ca921d881ce`
- examples: 8,193
- license: CC BY-SA 4.0

The dataset is strongly imbalanced, so raw accuracy alone is not a valid success criterion.

Pre-registered diagnostic gate:
- overall accuracy >= 0.50
- macro recall >= 0.45
- entailment recall >= 0.30
- contradiction recall >= 0.50

All conditions must pass.

Also report:
- majority-class accuracy baseline;
- macro-F1;
- neutral recall even though neutral has only 47 examples;
- full confusion matrix;
- per-category accuracy.

No result from this dataset may alter R12 model selection.
