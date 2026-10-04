# HIRA V1 S49 closure — matched court invalidated by native-trajectory divergence

Status: **SCIENTIFICALLY CLOSED / INVALID MATCHED COURT / INCONCLUSIVE**

Issue: #281  
PR: #282

## Qualified A0

S49-A0:
- run `37178164136`
- artifact `11293039754`
- digest `sha256:ce07ed474c21185d5a4316550e116606d7c09b90c808aea028354f2f046678e0`
- outcome `HIRA_V1_S49_A0_PRIVATE_QUERY_FREE_OPTION_IDENTITY_READY`

A0 established:
- query-free identity API has no question inputs
- identity trainable parameters **0**
- correction parameters **114,688**
- total treatment trainable **163,840**
- K=3/7/255
- question substitution identity error **0**
- option/state/token/view permutation invariants
- ownership isolation
- one encoder pass/state-once.

## Initial scientific run

Run: `37178974852`  
Scientific head: `7ece9b16fd5ccfd5b93d9ff9eb1cb73ff15105ab`

Reference arm:
- completed **24/24 epochs**
- DEV exposed for all 24 epochs
- exact frozen selector chose epoch **6**

Treatment:
- did not reach TRAIN_BEGIN
- mechanical constructor failure before treatment exposure.

Frozen recovery:
- `research/HIRA-V1-S49-RECOVERED-REFERENCE.json`
- `research/HIRA-V1-S49-PARTIAL-POSTREFERENCE-MECHANICAL-FAILURE-37178974852.md`

## Mechanical continuation

Constructor compatibility was repaired without changing:
- identity equations
- pair temperature
- A/B/W shapes
- seed
- rows
- optimizer
- loss
- selector.

Final continuation staging CI:
- head `14c0d63de7dbd345c251bcf72cb59be83f336d35`
- CI `37180944335`
- Python 3.10 PASS
- Python 3.12 PASS.

Treatment-only continuation:
- authorization head `f6119ef02063af7201dbc8f608e541d1b6410a53`
- run `37181138393`

Treatment completed **24/24 epochs** and exposed DEV.

## Frozen selected DEV observations

Reference selected epoch: **6**
Treatment selected epoch: **5**

Reference:
- fused canonical **0.5182292**
- fused paraphrase **0.4505208**
- paired both-correct **0.1510417**
- fused agreement **0.7265625**
- fused JS **0.0236701**
- canonical margin **-0.0461578**
- paraphrase margin **-0.1718225**
- relation canonical **0.4609375**
- relation paraphrase **0.4244792**
- relation agreement **0.7447917**
- relation JS **0.00623785**

Treatment:
- fused canonical **0.5156250**
- fused paraphrase **0.4427083**
- paired both-correct **0.1510417**
- fused agreement **0.6536458**
- fused JS **0.0372158**
- canonical margin **0.1046890**
- paraphrase margin **-0.2480520**
- relation canonical **0.4791667**
- relation paraphrase **0.4140625**
- relation agreement **0.7526042**
- relation JS **0.00337507**

Naive selected deltas (treatment minus reference), reported for audit only:
- fused canonical **-0.26 pp**
- fused paraphrase **-0.78 pp**
- paired both-correct **0**
- fused agreement **-7.29 pp**
- fused JS **+0.01355** worse
- canonical margin **+0.15085**
- paraphrase margin **-0.07623**
- relation canonical **+1.82 pp**
- relation paraphrase **-1.04 pp**
- relation agreement **+0.78 pp**
- relation JS **-0.00286**

These deltas are **NOT causal evidence**.

## Court invalidation

The preregistered interpretation plan ranks:

1. matched identical rows
2. **native trajectory identity**
3. corrected relation selected-choice behavior
4. fused correctness/stability
5. A0 invariants.

Native trajectory identity failed at **every epoch 1–24**.

Treatment continuation terminated with:

`RuntimeError: S49 continuation native trajectory differs from frozen reference at epochs [1, ..., 24]`

Therefore the matched controlled-variable claim is invalid.

The observed metric deltas confound:
- private identity treatment
with
- a different native runtime trajectory.

S49 cannot be classified as A, B, C, D, or E.

## Frozen conclusion

**S49 scientific question remains unresolved.**

The query-free identity family is neither accepted nor rejected by S49 because the required matched-native condition failed.

The next experiment must remove the possibility of arm-specific native divergence **by construction**, not by checking hashes after two independent native training trajectories.

## Stop-rule compliance

No:
- identity variant
- temperature sweep
- query leak
- identity blend
- learned projector
- loss/capacity change
- alternate selector
- scientific retry
- gate weakening
- second S49 DEV.

S49 is closed.
