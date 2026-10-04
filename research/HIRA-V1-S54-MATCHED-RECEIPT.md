# HIRA V1 S54 matched scientific receipt — Joint State–Query–Option Interaction

Status: **FROZEN / CASE C**

Scientific run: `37209555118`  
Artifact: `11305929426`  
Artifact digest: `sha256:1b9787fe775493e53d7f7f0108295914cd974bfd8da9f13a8f03eb02f7109ae2`  
Scientific head: `3384d86655664f9c28a7f6850ae47fc63838da87`

Outcome:
`HIRA_V1_S54_JOINT_STATE_QUERY_OPTION_INTERACTION_DEV_COMPLETE`

## Matched authority

- parent S51 native run `37192490832`
- parent native artifact `11299783210`
- runtime/native digest `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- native optimizer absent
- native trainable params **0**
- native retraining **false**
- S54-A0 run `37208186642`
- S54-A0 artifact `11305528035`
- shared TRAIN cache digest `d2f26e8a4cdd2c6e1c0de64b7aa5511e3d01891f3e05cfa278618be86d36a1f6`
- shared DEV cache digest `6627c9b922795991ea4dba40cc75ab7380ebd55b7645b6a7f39bc3de8c62666c`
- reference/treatment same cache bytes **true**
- private state-view encodes **0**

Matched private surface:
- reference correction **114,688**
- treatment correction **114,688**
- joint interaction params **0**
- identity params **0**
- bit-identical initialization **true**
- temperature **0.10**

## Selected checkpoints

Reference selected epoch: **18**  
Treatment selected epoch: **22**

## Reference — S53 query↔option context

- fused canonical accuracy **0.5078125**
- fused paraphrase accuracy **0.4609375**
- paired both-correct **0.22395833333333334**
- fused agreement **0.5677083333333334**
- fused JS **0.04582352594782909**
- relation canonical accuracy **0.5078125**
- relation paraphrase accuracy **0.4947916666666667**
- relation agreement **0.7447916666666666**
- relation JS **0.02400724298786372**

## Treatment — joint state-query-option context

- fused canonical accuracy **0.5182291666666666**
- fused paraphrase accuracy **0.5**
- paired both-correct **0.234375**
- fused agreement **0.5625**
- fused JS **0.04649155167862773**
- relation canonical accuracy **0.5208333333333334**
- relation paraphrase accuracy **0.5286458333333334**
- relation agreement **0.7447916666666666**
- relation JS **0.025834825006313622**

## Treatment minus reference

Correctness:
- fused canonical **1.04 pp**
- fused paraphrase **3.91 pp**
- paired both-correct **1.04 pp**
- canonical relation accuracy **1.30 pp**
- paraphrase relation accuracy **3.39 pp**

Stability:
- fused selected-choice agreement **-0.52 pp**
- fused JS **+0.000668**
- relation selected-choice agreement **0.00 pp**
- relation JS **+0.001828**

Margins:
- fused canonical **-0.030909**
- fused paraphrase **+0.061081**
- relation canonical **+0.019818**
- relation paraphrase **+0.088296**

## Treatment diagnostics

- same-option context cross-view cosine **0.981315885980924**
- mean joint attention entropy **2.9256500204404197**
- mean max joint token weight **0.06129637531315287**
- mean state support **0.9582775747022664**
- mean option support **0.9565168118388236**
- context norm max error **1.7881393432617188e-7**

## Frozen interpretation

**Case C — useful correctness improves, but selected-choice stability does not materially improve.**

Treatment improves:
- fused canonical **+1.04 pp**
- fused paraphrase **+3.91 pp**
- paired both-correct **+1.04 pp**
- canonical relation accuracy **+1.30 pp**
- paraphrase relation accuracy **+3.39 pp**.

But:
- fused agreement **-0.52 pp**
- fused JS worsens **+0.000668**
- relation agreement **0.00 pp**
- relation JS worsens **+0.001828**.

The triadic context itself is highly cross-view stable (same-option cosine ~**0.9813**) and receives strong state/option support, yet that does not translate into more stable selected choices.

Therefore direct parameter-free state–query–option factorization is useful for correctness but is **not sufficient** to solve the remaining decision instability.

S54 closes the parameter-free joint-interaction family. The next experiment should test a **small learned joint interaction** under matched capacity, while retaining the exact persisted-native/shared-cache discipline.

No S54 retry, temperature sweep, aggregation sweep, or second DEV is authorized.
