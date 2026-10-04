# HIRA V1 S47 matched scientific receipt

Status: **FROZEN**

Scientific run: `37168305687`  
Artifact: `11291161358`  
Artifact digest: `sha256:9887903bcdc7e59907d0bde425389b7642bff2f91f2a7fe5e12555ad89c3db76`  
Scientific head: `63b7521a4ed60ad616e0ab8675bfa076cc29e81a`

Outcome:
`HIRA_V1_S47_ORDINAL_PAIRWISE_CONSENSUS_DEV_COMPLETE`

## Authority

- seed **68001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S47 domains
- K=4
- 24 epochs
- batch 16
- one DEV only
- no post-DEV tuning
- no second DEV
- no external Laya/Jev evaluation

Selected S45-mechanics treatment checkpoint:
- epoch **24**
- total trainable **163,840**
- correction trainable **114,688**

## Same-checkpoint legacy S45 shell

- canonical accuracy **0.5520833**
- paraphrase accuracy **0.5494792**
- paired both-correct **0.3020833**
- question-swap **0.6458333**
- cross-view agreement **0.6432292**
- cross-view JS **0.0441499**
- canonical margin **0.0147362**
- paraphrase margin **-0.0159475**
- option-order flip **0**
- probability-mass max error **1.788e-7**

## Same-checkpoint S47 ordinal shell

- canonical accuracy **0.4921875**
- paraphrase accuracy **0.5338542**
- paired both-correct **0.2239583**
- question-swap **0.5677083**
- cross-view agreement **0.5078125**
- cross-view JS **0.3377110**
- canonical margin **-4.78125**
- paraphrase margin **-1.71875**
- option-order flip **0**
- probability-mass max error **5.96e-8**

## S47 minus legacy S45 at identical checkpoint

Correctness:
- canonical **-0.0598958** (**-5.99 pp**)
- paraphrase **-0.0156250** (**-1.56 pp**)
- paired **-0.0781250**
- question-swap **-0.0781250**

Stability:
- selected-choice agreement **-0.1354167** (**-13.54 pp**)
- cross-view JS **+0.2935611** worse
- canonical margin **-4.7959862**
- paraphrase margin **-1.7028025**

## Expert evidence at selected checkpoint

Primary:
- canonical/paraphrase **0.5182292 / 0.4583333**

Native relation:
- canonical/paraphrase **0.2994792 / 0.3203125**
- cross-view agreement **0.3932292**
- cross-view JS **0.0018580**

Corrected private relation:
- canonical/paraphrase **0.3593750 / 0.5182292**
- cross-view agreement **0.5312500**
- cross-view JS **0.0860871**

Native signature:
- same-option cosine **0.8321675**
- same-vs-wrong margin **0.2224311**

## Ordinal diagnostics

Canonical:
- unanimous pair fraction **0.3732639**
- 2/3 majority fraction **0.6267361**
- exact tied pair fraction **0**
- private top-set tie-break fraction **0.03125**

Paraphrase:
- unanimous pair fraction **0.3237847**
- 2/3 majority fraction **0.6762153**
- exact tied pair fraction **0**
- private top-set tie-break fraction **0.0442708**

## Native trajectory identity

All **24** epoch runtime fingerprints are exactly equal between reference and treatment.

## Frozen interpretation

**Case C — correctness remains nontrivial, but cross-view stability fails more strongly under rank-only consensus.**

S47 eliminates score magnitude entirely yet final selected-choice agreement falls by **13.54 pp** and cross-view JS worsens by **0.29356**.

Therefore the dominant remaining failure is not cross-expert scale or magnitude. It is **view-conditioned expert ordering itself**: paraphrased queries cause the experts to reorder options differently.

This closes the downstream score/rank aggregation family.

No second S47 DEV is authorized.
