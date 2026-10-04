# HIRA V1 S51 pre-Phase-A staging receipt

Status: **STAGED / NATIVE AUTHORITY NOT AUTHORIZED**

Issue: #285  
PR: #286

## Qualified S51-A0

Run: `37192065019`  
Artifact: `11299312438`  
Digest: `sha256:b54596990a444e1ac3ffba71a9bfc3a910fced67f482af43879550278145fd68`  
Authorization head: `a4369e92b11ab724801dd69418798f5f8b8f51d7`

Outcome:
`HIRA_V1_S51_A0_PERSISTED_NATIVE_AUTHORITY_READY`

Qualified mechanics:
- save/load/second-load runtime hash exact
- file tamper rejected
- loaded native trainable params 0
- loaded native optimizer params 0
- cache normal non-inference tensors
- branch-order replay error 0
- correction initialization bit-identical
- reference/treatment correction params 114,688 each
- identity params 0
- Phase-A DEV/private isolation PASS
- Phase-B native-training isolation PASS.

## Phase-A authority workflow

Workflow:
`.github/workflows/hira-v1-s51-native-authority.yml`

Marker:
`research/HIRA-V1-S51-ENABLE-NATIVE-AUTHORITY`

Pinned A0 input:
- run `37192065019`
- artifact `hira-v1-s51-a0-persisted-native-authority`

Phase A:
- uses M4 runtime bundle run `36357825580`
- uses S51 TRAIN only
- seed **72001**
- TRAIN **768**
- 12 wholly fresh domains
- fixed native epoch **24**
- native surface **49,152**
- projection surface **32,768 / 128x256**
- does not generate/encode/score S51 DEV
- does not construct private correction
- does not model-select native epochs
- persists exact epoch-24 authority.

Output artifact:
`hira-v1-s51-native-authority`

Required files:
- `native-authority.pt`
- `train-manifest.json`
- `authority-receipt.json`
- `INTEGRITY.sha256`

Receipt binds:
- file SHA-256
- logical native tensor digest
- runtime-state SHA-256
- semantic revision
- T0 initialization SHA
- TRAIN manifest SHA.

## Authorization rule

The Phase-A marker MUST remain absent until the exact final staging head passes generic CI on Python 3.10 and 3.12.

Once Phase A begins:
- exactly one native authority artifact;
- no S51 DEV exposure;
- no alternate native authority;
- no authority regeneration for scientific weakness;
- Phase B remains separately gated after artifact qualification.
