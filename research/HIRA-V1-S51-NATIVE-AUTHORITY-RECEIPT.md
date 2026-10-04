# HIRA V1 S51 Phase-A native authority receipt

Status: **QUALIFIED / SEALED**

Run: `37192490832`  
Job: `111407437243`  
Artifact: `11299783210` / `hira-v1-s51-native-authority`  
Artifact digest: `sha256:d2c2ec63ef6d7c197051ff357ed3ea21c93e8a903b4c5bb7ab8c01e216d91c59`  
Authorization head: `a5f0240459eae04d2d1d7dd4e24e735d58ca8d83`

Outcome:
`HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY`

## Frozen native authority

- seed: **72001**
- TRAIN semantic cases: **768**
- domains: **12**
- fixed native epoch: **24**
- native trainable surface during Phase A: **49,152**
- LoRA: **16,384**
- projection: **32,768 / 128×256**

Exact authority hashes:

- runtime state SHA-256:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- logical native tensor digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- persisted checkpoint file SHA-256:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- TRAIN manifest SHA-256:
  `590aa9464a5d7925eb1028130f0f66957955437f2b62c6338fcd252b7de56c51`
- semantic revision:
  `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- T0 initialization SHA:
  `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

## Isolation evidence

Phase A reported:

- DEV generated: **false**
- DEV encoded: **false**
- DEV scored: **false**
- private correction constructed: **false**
- model selection across native epochs: **false**
- production-ready claim: **false**

The epoch-24 checkpoint is therefore the single sealed S51 native authority.

## Governance consequence

No second Phase-A authority is authorized.

Phase B must:
1. download exactly run `37192490832`, artifact `hira-v1-s51-native-authority`;
2. verify `INTEGRITY.sha256`;
3. verify checkpoint file SHA, logical tensor digest, runtime hash, TRAIN manifest SHA, semantic revision and T0 binding;
4. load that exact authority and freeze all native parameters;
5. only then generate/encode S51 DEV;
6. run exactly one private DEV court.

Native retraining in Phase B is forbidden.
