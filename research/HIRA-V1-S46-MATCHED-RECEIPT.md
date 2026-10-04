# HIRA V1 S46 matched scientific receipt

Status: **FROZEN**

Scientific run: `37163095749`  
Artifact: `11289416980`  
Artifact digest: `sha256:2d3c6ef18a3082fcc4515f5977ad3997c625bd07a239c3ba4a4aa9ccf0735789`  
Scientific head: `e14e4813da41828d685929ebbf18eb19d93fc4c2`

Outcome:
`HIRA_V1_S46_ROBUST_THREE_EXPERT_CONSENSUS_DEV_COMPLETE`

## Authority

- seed **67001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S46 domains
- K=4
- 24 epochs
- batch 16
- one DEV only
- no post-DEV tuning
- no second DEV
- no external Laya/Jev evaluation

Selected S45-mechanics treatment checkpoint:
- epoch **23**
- total trainable **163,840**
- correction trainable **114,688**

## Same-checkpoint legacy S45 shell

- canonical accuracy **0.4583333**
- paraphrase accuracy **0.4088542**
- paired both-correct **0.1979167**
- question-swap **0.5989583**
- cross-view agreement **0.6901042**
- cross-view JS **0.0195794**
- canonical margin **-0.1148622**
- paraphrase margin **-0.2281751**
- option-order flip **0**
- probability-mass max error **1.192e-7**

## Same-checkpoint S46 robust median shell

- canonical accuracy **0.4921875**
- paraphrase accuracy **0.4010417**
- paired both-correct **0.2083333**
- question-swap **0.5468750**
- cross-view agreement **0.6015625**
- cross-view JS **0.0483414**
- canonical margin **-0.1071954**
- paraphrase margin **-0.2852077**
- option-order flip **0**
- probability-mass max error **1.788e-7**

## S46 minus legacy S45 at identical checkpoint

Correctness:
- canonical **+0.0338542**
- paraphrase **-0.0078125**
- paired **+0.0104167**
- question-swap **-0.0520833**

Stability:
- cross-view agreement **-0.0885417**
- cross-view JS **+0.0287620** worse
- canonical margin **+0.0076668**
- paraphrase margin **-0.0570325**

## Expert evidence at selected checkpoint

Primary:
- canonical/paraphrase **0.3541667 / 0.3541667**

Native relation:
- canonical/paraphrase **0.3541667 / 0.2864583**
- cross-view agreement **0.5078125**
- cross-view JS **0.0010528**

Corrected private relation:
- canonical/paraphrase **0.5078125 / 0.3906250**
- cross-view agreement **0.6093750**
- cross-view JS **0.0287791**

Native signature:
- same-option cosine **0.9133995**
- same-vs-wrong margin **0.1491522**

## Native trajectory identity

All **24** epoch runtime fingerprints are exactly equal between reference and treatment.

## Frozen interpretation

**Case C — useful correctness remains, but score-level robust consensus does not recover cross-view stability.**

The S46 median shell improves canonical accuracy modestly, but:
- paraphrase accuracy does not improve;
- final selected-choice agreement drops by **8.85 pp**;
- final JS worsens by **0.02876**;
- therefore adding a low-JS but weak native relation expert through magnitude-level median fusion does not solve the instability.

This closes the score-magnitude robust-aggregation family for S46.

No second S46 DEV is authorized.
