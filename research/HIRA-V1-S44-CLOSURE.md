# HIRA V1 S44 closure — Private Correction Representation Fork

Status: **SCIENTIFICALLY CLOSED**

Issue: #271  
PR: #272

## Qualified mechanism

S44-A0:
- run `37120084360`
- artifact `11272917607`
- outcome `HIRA_V1_S44_A0_PRIVATE_CORRECTION_REPRESENTATION_READY`

A0 established:
- one encoder pass / state-once
- private adapter A/B + full W only receive correction gradients
- correction -> native LoRA/projection gradient exactly zero
- native objective -> private A/B/W gradient exactly zero
- B/W zero-init identity
- deterministic W -> B -> A warm-start
- nonlinear private representation differs from W-only S39
- arbitrary-K, invariance, checkpoint and full-K mechanics

## Mechanical abort audit

The first authorized run `37122580868` aborted before training due to a nonexistent import.

No `TRAIN_BEGIN`, epoch line, DEV metric, selected epoch or artifact was exposed.

The mechanical repair was frozen and re-authorized at:
`18f94cb05d48fa7b114ca3db08af8e3a10d5dd93`.

## Fresh matched scientific court

Run: `37123003224`  
Artifact: `11274687323`  
Digest: `sha256:a3b5c1c2ec16211c0f9f3123fccd7e93b53d934bd37371af94741616eed748cc`

All workflow steps passed.

Outcome:
`HIRA_V1_S44_MATCHED_PRIVATE_CORRECTION_REPRESENTATION_DEV_COMPLETE`

Reference and treatment both selected epoch **17**.

### Reference

- gates **15/23**
- DEV_READY false
- fused canonical/paraphrase **0.5286458 / 0.4817708**
- relation canonical/paraphrase **0.3255208 / 0.3020833**
- fused agreement **0.5182292**
- relation agreement **0.5703125**
- fused JS **0.0360378**
- native signature cosine/margin **0.8880017 / 0.1563350**

### Treatment

- gates **15/24**
- DEV_READY false
- fused canonical/paraphrase **0.5833333 / 0.4895833**
- corrected relation canonical/paraphrase **0.5026042 / 0.4244792**
- fused agreement **0.5546875**
- corrected relation agreement **0.4244792**
- fused JS **0.0670551**
- native signature cosine/margin **0.8880017 / 0.1563350**

### Treatment minus reference

Correctness:
- fused canonical **+0.0546875**
- fused paraphrase **+0.0078125**
- paired both-correct **+0.0625000**
- question-swap **+0.0520833**
- relation canonical **+0.1770833**
- relation paraphrase **+0.1223958**

Native path:
- raw primary canonical **0**
- raw primary paraphrase **0**
- signature cosine delta **0**
- signature margin delta **0**
- LoRA max state delta **0**
- projection max state delta **0**
- native pre-W logit max delta **0**
- native signature max delta **0**
- every epoch runtime-state fingerprint identical

Remaining stability cost:
- relation agreement **-0.1458333**
- fused JS **+0.0310173** worse

## Frozen interpretation

**Case A — the private correction representation resolves the representation-ownership conflict at the native-path boundary.**

This is the first stage in the S38-S44 chain where:
1. useful correctness rises materially,
2. native transport/primary representation is provably unchanged,
3. the correction branch remains one-pass and state-once.

The result is not a final candidate:
- treatment is not DEV_READY;
- paraphrase fused improvement remains small;
- corrected relation cross-view agreement is poor;
- fusion JS worsens.

Therefore S44 establishes the architectural split but not the final decision shell.

## Stop-rule compliance

No:
- hidden-width or activation sweep
- bias/gate/layernorm addition
- residual-scale tuning
- native-gradient leakage
- second encoder
- native/private mixing coefficient
- W-only scheduler retry
- seed/LR/epoch/batch retry
- gate weakening
- second DEV
- external Laya/Jev evaluation

S44 is frozen and closed.

Exact receipts:
- `research/HIRA-V1-S44-MATCHED-RECEIPT.json`
- `research/HIRA-V1-S44-MATCHED-RECEIPT.md`

Next:
**S45 — Cross-View Consistent Private Correction**.
