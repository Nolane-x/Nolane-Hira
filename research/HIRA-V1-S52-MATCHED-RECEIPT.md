# HIRA V1 S52 matched scientific receipt — Paired-View Query Relation Canonicalization

Status: **FROZEN / CASE D**

Scientific run: `37204041655`  
Artifact: `11303604910`  
Artifact digest: `sha256:71c36c502bf60b7ed4ae38e7ab610d7f9e28269166076123f78d4bc3b2b1487d`  
Scientific head: `d444266075ddab4e35b6b1d933a346858fe1f230`

Outcome:
`HIRA_V1_S52_QUERY_RELATION_CANONICALIZATION_DEV_COMPLETE`

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- native trainable params **0**
- native optimizer **absent**
- native retraining **false**
- S52-A0 run `37199183308`
- S52-A0 artifact `11302402275`
- shared TRAIN cache digest `7bc1bc8d7fef4ea2d6251a174aa4e300c4843be4ab99e1ee27d0bdb52a77430b`
- shared DEV cache digest `9c42fe536dea4a4ff5c0fbecba9ae86883048bb616db28328e83a85195eff0a6`
- reference/treatment same cache bytes **true**
- private state-view encodes **0**

Private matched surface per arm:
- correction **114,688**
- query canonicalizer **32,768**
- total private trainable **147,456**
- bit-identical initialization **true**

Controlled variable:
- reference relation auxiliary coefficient **0.0**
- treatment relation auxiliary coefficient **0.10**
- separation ceiling **0.25**

## Selected checkpoints

Reference selected epoch: **12**  
Treatment selected epoch: **18**

## Reference

- fused canonical accuracy **0.5208333333333334**
- fused paraphrase accuracy **0.46875**
- paired both-correct **0.265625**
- fused agreement **0.5390625**
- fused JS **0.057475363137200475**
- question-swap **0.8229166666666666**
- canonical relation accuracy **0.515625**
- paraphrase relation accuracy **0.4739583333333333**
- relation agreement **0.5234375**
- relation JS **0.013750451461722454**

Query code:
- same-relation cosine **0.49614306327809266**
- relation-centroid cross cosine **0.6917581222951412**
- separation margin **-0.19561505901704856**
- raw-vs-canonicalized cosine **0.35118719438711804**
- residual norm **1.1375782589117687**

## Treatment

- fused canonical accuracy **0.4869791666666667**
- fused paraphrase accuracy **0.4348958333333333**
- paired both-correct **0.203125**
- fused agreement **0.4817708333333333**
- fused JS **0.07113030459731817**
- question-swap **0.8541666666666666**
- canonical relation accuracy **0.5234375**
- paraphrase relation accuracy **0.46875**
- relation agreement **0.5598958333333334**
- relation JS **0.04104015836492181**

Query code:
- same-relation cosine **0.6501543795069059**
- relation-centroid cross cosine **0.8285731474558512**
- separation margin **-0.1784187679489453**
- raw-vs-canonicalized cosine **0.6455001632372538**
- residual norm **0.834097887078921**

## Treatment minus reference

Correctness:
- fused canonical **-3.39 pp**
- fused paraphrase **-3.39 pp**
- paired both-correct **-6.25 pp**
- canonical relation accuracy **0.78 pp**
- paraphrase relation accuracy **-0.52 pp**

Stability:
- fused agreement **-5.73 pp**
- fused JS **+0.013655** worse
- relation agreement **+3.65 pp**
- relation JS **+0.027290** worse

Query-code geometry:
- same-relation cosine **+0.154011**
- relation-centroid cross cosine **+0.136815**
- separation margin **+0.017196**
- raw-vs-canonicalized cosine **+0.294313**
- residual norm **-0.303480**

## Frozen interpretation

**Case D — correctness and fused stability materially regress.**

The paired-view auxiliary did what it was designed to do locally: same-relation query-code cosine increases materially.

However, it also raises the cosine between the A/B relation centroids almost as strongly. The separation margin improves only slightly and remains negative.

At the decision surface:
- canonical and paraphrase fused accuracy each fall by ~**3.39 pp**;
- paired both-correct falls **6.25 pp**;
- fused selected-choice agreement falls **5.73 pp**;
- fused JS worsens by **0.013655**.

Relation agreement rises **3.65 pp**, but relation JS worsens sharply and this does not translate into fused stability or correctness.

Therefore vector-level query relation canonicalization with this paired alignment/separation family is rejected.

No S52 retry or coefficient/margin/hidden-dimension sweep is authorized.
