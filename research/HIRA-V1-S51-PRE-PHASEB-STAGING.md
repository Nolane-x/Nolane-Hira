# HIRA V1 S51 pre-Phase-B staging receipt

Status: **STAGED / PRIVATE DEV COURT NOT AUTHORIZED**

Issue: #285  
PR: #286

## Qualified parent chain

S51-A0:
- run `37192065019`
- artifact `11299312438`
- digest `sha256:b54596990a444e1ac3ffba71a9bfc3a910fced67f482af43879550278145fd68`
- outcome `HIRA_V1_S51_A0_PERSISTED_NATIVE_AUTHORITY_READY`

S51 Phase-A native authority:
- run `37192490832`
- job `111407437243`
- artifact `11299783210`
- artifact name `hira-v1-s51-native-authority`
- artifact digest `sha256:d2c2ec63ef6d7c197051ff357ed3ea21c93e8a903b4c5bb7ab8c01e216d91c59`
- outcome `HIRA_V1_S51_PERSISTED_NATIVE_AUTHORITY_READY`

## Sealed native authority

- seed **72001**
- fixed epoch **24**
- runtime state SHA:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- logical tensor digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint file SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- TRAIN manifest SHA:
  `590aa9464a5d7925eb1028130f0f66957955437f2b62c6338fcd252b7de56c51`
- semantic revision:
  `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- T0 SHA:
  `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

Phase-A exposure:
- DEV generated **false**
- DEV encoded **false**
- DEV scored **false**
- private correction constructed **false**
- model selection **false**

No second Phase-A authority is authorized.

## Phase-B workflow

Workflow:
`.github/workflows/hira-v1-s51-artifact-pinned-private-court.yml`

Marker:
`research/HIRA-V1-S51-ENABLE-PRIVATE-COURT`

Pinned inputs:
- M4 runtime run `36357825580`
- S51-A0 run `37192065019`
- S51 native authority run `37192490832`
- exact artifact name `hira-v1-s51-native-authority`
- frozen artifact/checkpoint/runtime/TRAIN-manifest hashes above.

Phase B must:
1. verify A0;
2. verify authority artifact `INTEGRITY.sha256`;
3. verify frozen receipt and exact native hashes;
4. load persisted native checkpoint without retraining;
5. verify loaded runtime-state hash exact;
6. freeze native to zero trainability;
7. generate fresh S51 TRAIN/DEV;
8. materialize exactly one shared immutable cache;
9. destroy live native runtime;
10. train reference/treatment private branches using identical cache bytes;
11. expose exactly one S51 private DEV authority.

Private matched surface:
- reference correction **114,688**
- treatment correction **114,688**
- identity params **0**
- LR `s35.LR = 2e-4`
- weight decay `s35.WEIGHT_DECAY = 0.01`
- same loss / selector / 24 private epochs
- raw query remains only at final readout.

## Stop rule after Phase-B DEV begins

No:
- native retraining
- native authority regeneration
- cache regeneration after DEV
- identity variant
- temperature/mixing sweep
- query leak/blend/projector
- private loss/capacity change
- selector change
- retry for scientific weakness
- second S51 DEV
- external Laya/Jev evaluation.

## Authorization rule

`HIRA-V1-S51-ENABLE-PRIVATE-COURT` MUST remain absent until the exact final Phase-B staging head passes generic CI on Python 3.10 and Python 3.12.
