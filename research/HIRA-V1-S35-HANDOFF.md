# HIRA V1 S35 handoff — to S36 Native Signature Linear Correctness Readout

S35 is frozen as:

`HIRA_V1_S35_MATCHED_NATIVE_DEV_COMPLETE`

Preregistered interpretation:
**Case B — native transport improves but semantic decision quality regresses.**

## Canonical authority

A0:
- run `37008578553`
- artifact `11227286171`
- digest `sha256:a32475c6b9278ea5feada4f4c002188089b1a2add17ae732f3a4b6ae1fa16a47`

Fresh matched TRAIN/DEV:
- run `37014291731`
- artifact `11231711883`
- digest `sha256:570486a4460b420f628db155edda626a8a05aa0cee708003b45bb055c27171c7`
- scientific head `40a64d519636411e7c18bb762d0d717cc07bce95`

Projected control selected:
- epoch **22**
- fused canonical/paraphrase **0.53125 / 0.5390625**
- relation canonical/paraphrase **0.4921875 / 0.6354167**
- signature cosine/margin **0.8087542 / 0.0595524**
- gates **13/22 PASS**

Native treatment selected:
- epoch **23**
- fused canonical/paraphrase **0.4479167 / 0.4791667**
- relation canonical/paraphrase **0.3619792 / 0.328125**
- signature cosine/margin **0.9196730 / 0.1631556**
- gates **15/22 PASS**

## Residual

Native 256D relation geometry:
- strongly improves cross-wording option identity;
- passes selected signature cosine and signature discrimination gates;
- improves fused JS and agreement;
- but substantially worsens relation correctness.

The stable native signature is therefore not itself a correctness score.

## S36 hypothesis

**Native Signature Linear Correctness Readout**

Both arms use:
- exact S35 native relation operator and 256D signatures;
- exact S17 primary path/fusion/loss shell;
- final-attention LoRA 16,384;
- shared primary projection 32,768;
- original A13 frozen;
- HIRACore frozen.

Control:
- exact S35 native relation logits.

Treatment:
- native logits + `signature @ w`;
- `w` is one shared zero-initialized 256D trainable vector;
- no bias;
- no MLP;
- no option-specific or domain-specific parameters;
- added trainable params **256**;
- exact zero-init identity to control.

Treatment trainable total:
**49,408**.

## Required S36-A0

Before fresh DEV:
- exact zero-init relation-logit identity
- exact signature identity
- exact primary/fused identity
- exact added params 256
- treatment total 49,408
- original A13/HIRACore frozen
- readout relation-gradient nonzero
- readout primary-gradient exactly zero
- LoRA/projection ownership remains finite and expected
- logical-option permutation equivariance
- arbitrary-K support
- native relation remains independent of shared projection
- full-K/state-once/probability/checkpoint mechanics

A0 semantic scores diagnostic only.

## Fresh matched authority

Use wholly fresh S36 A0/TRAIN/DEV.
No S35 rows.
No M5 final/confirmatory rows.
No W29-W34 sealed rows.

Freeze before exposure:
- readout dimension 256
- zero initialization
- no bias
- residual scale **1.0**
- AdamW treatment includes readout under the same LR/weight decay
- no readout-only LR
- no nonlinearity

No second DEV.
No scale/bias/MLP/rank retry.
No projected/native mixing.
No Laya/Jev benchmark before DEV_READY.
