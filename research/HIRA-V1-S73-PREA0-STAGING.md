# HIRA V1 S73 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #333  
PR: #334

Parent S72:
- merged main `16eb7a583376cd824421d5e7ae9df948979b3953`
- scientific run `37470998139`
- artifact `11416962628`
- digest `sha256:9a653a681e0cb3182819634fe24eb345ce82a48f2d078d15e510d1d3ebfb0834`
- verdict **Case B**.

Frozen S73 mechanics:
- exact S72 candidate in both arms;
- 8 detached safety features;
- 9-param linear logistic predictor;
- zero initialization / initial probability 0.5;
- fixed threshold 0.5;
- TRAIN-only CE+JS counterfactual safety target;
- reference always applies candidate;
- treatment emits exactly candidate or fused;
- no upstream gradient;
- fresh S73 TRAIN/DEV exposed=false.

A0 seed: **94001**.

A0 may verify mechanics only. It may not expose fresh S73 TRAIN/DEV.

Authorization marker:
`research/HIRA-V1-S73-ENABLE-A0`

Marker MUST remain absent until this pre-A0 staging head passes generic CI on Python 3.10 and 3.12.
