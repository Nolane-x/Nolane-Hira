# HIRA V1 S53 pre-A0 staging receipt

Status: **STAGED / A0 NOT AUTHORIZED**

Issue: #289

## Parent

S52 merged main:
`4b9ac7e3613c86bbb897421d6d00d9b800b5ae50`

S52 closure:
**CASE D**

Scientific run:
- `37204041655`
- artifact `11303604910`
- digest `sha256:71c36c502bf60b7ed4ae38e7ab610d7f9e28269166076123f78d4bc3b2b1487d`

Key evidence:
- pooled query canonicalization increased same-relation query-code cosine;
- cross-relation centroid cosine also increased;
- fused canonical/paraphrase accuracy -3.39 pp;
- paired both-correct -6.25 pp;
- fused agreement -5.73 pp;
- fused JS worsened.

## S53 controlled family

Reference:
- exact S51 pooled-query correction.

Treatment:
- exact same query-free option identity;
- zero-parameter token↔option cosine interaction;
- temperature 0.10;
- masked softmax over query tokens;
- per-option normalized query context;
- no pooled global query bypass.

Matched per-arm trainable surface:
- correction adapter A 32,768
- adapter B 16,384
- bilinear W 65,536
- total correction/private trainable **114,688**
- identity params **0**
- treatment late-interaction params **0**
- bit-identical correction initialization.

## Frozen backbone

Reuse exact S51 persisted native authority:
- run `37192490832`
- artifact `11299783210`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`

Native retraining forbidden.

## A0 staged

Files:
- `src/nmd/v1_token_query_option_late_interaction.py`
- `tests/test_v1_token_query_option_late_interaction.py`
- `scripts/hira_v1_s53_a0_token_query_option_late_interaction.py`
- `tests/test_v1_s53_a0_harness.py`
- `.github/workflows/hira-v1-s53-a0-token-query-option-late-interaction.yml`

A0 checks:
- matched 114,688 correction params
- zero treatment interaction params
- K=3/7/255
- probability mass
- context unit norm
- attention mass
- query padding/mask invariance
- token permutation invariance
- option permutation equivariance
- all-masked rejection
- per-option context diversity
- runtime proof of no pooled `query_summary` bypass
- informative-token context/logit sensitivity
- native/cache ownership isolation
- no second encoder.

## Authorization rule

Marker:
`research/HIRA-V1-S53-ENABLE-A0`

The marker MUST remain absent until the exact final pre-A0 staging head passes generic CI on Python 3.10 and Python 3.12.

A0 is mechanical/diagnostic only:
- no model selection
- no S53 TRAIN/DEV exposure
- no external Laya/Jev evaluation.
