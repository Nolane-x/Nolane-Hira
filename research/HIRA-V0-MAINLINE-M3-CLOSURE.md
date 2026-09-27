# HIRA V0 MAINLINE M3 closure — multilingual EN/VI

Status: **CLOSED — MULTILINGUAL REMAINS PROVISIONAL**

Issue: #165  
PR: #166  
Branch: `feat/hira-v0-mainline-m3-multilingual`  
Base main: `aed5ec3ad0651e18e27a9fe77cd3ed6c33dceac4`

## 1. Purpose

M3 tested whether the existing Hira semantic path could support English/Vietnamese paired semantics, then allowed exactly one blocking multilingual alignment rescue.

M3 did not reopen M2 high-K evidence and did not use MASSIVE/XNLI direct benchmark rows for architecture selection.

## 2. M3-A zero-training paired baseline

Authoritative run:

`36320497057`

Artifact:

- name: `hira-v0-mainline-m3-paired-dev`
- ID: `10932028604`
- digest: `sha256:d9837f68ed0f9a7d216c42a95e3cc9977fa5f3d218aa792493ecfcf63a47d6bd`

Outcome:

`HIRA_V0_M3_PAIRED_DEV_FAIL`

Fresh DEV authority:
- MVA + MVB
- 72 latent EN/VI pairs
- 144 language cases
- zero gradient updates

### English

- top-1: 0.4305555555555556
- MRR: 0.6782407407407408
- mean gold rank: 1.7916666666666667
- choice top-1: 0.375
- score top-1: 0.4166666666666667
- noul top-1: 0.5

### Vietnamese

- top-1: 0.3888888888888889
- MRR: 0.6643518518518519
- mean gold rank: 1.7916666666666667
- choice top-1: 0.3333333333333333
- score top-1: 0.3333333333333333
- noul top-1: 0.5

### Cross-lingual

- VI/EN top-1 ratio: 0.9032258064516129
- VI/EN MRR ratio: 0.9795221843003412
- paired prediction agreement: 0.16666666666666666
- paired both-correct rate: 0.05555555555555555

Runtime integrity:
- state-once: PASS
- full-K: PASS
- finite: PASS
- probability mass max error: 1.1920928955078125e-07
- relation delta: 0
- gradient updates: false

Scientific localization:

The baseline failure was not a simple Vietnamese-only collapse. VI retained roughly 90%+ of EN marginal quality, while both languages were weak in absolute semantic discrimination and paired EN/VI predictions agreed only 16.67%.

MVC remained sealed after M3-A.

## 3. M3-R1 shared alignment rescue

R1 used one shared language-agnostic alignment adapter:

`A(x) = x + U(gelu(D(layer_norm(x))))`

Architecture:
- rank 16
- 256 -> 16 -> 256
- bias-free
- non-affine layer norm
- zero-initialized up projection
- no language-specific branch
- no language-ID feature
- exactly **8,192 trainable parameters**

Frozen:
- A13
- W28 T0
- W34 co-evidence scorer
- HIRACore
- reliability components

Fresh R1 evidence:
- TRAIN MVD/MVE/MVF/MVG: 144 latent pairs = 288 language cases
- DEV MVH/MVI: 72 latent pairs = 144 language cases

Loss:
- task CE
- + 0.35 paired alignment
- + 0.10 English anchor

Optimization:
- 8 epochs
- AdamW
- lr 2e-4
- weight decay 0.01
- gradient clip 1.0

Seeds:
- primary 23031
- replica 23037

## 4. M3-R1 authoritative result

Run:

`36322772094`

Artifact:

- name: `hira-v0-mainline-m3-r1-train-dev`
- ID: `10933185512`
- digest: `sha256:29b0ae348ddc4dfbdfcb37f8a276861b540725cc47cd3aaa85ef509f3b521e03`

Outcome:

`HIRA_V0_M3_R1_DEV_FAIL`

Joint qualification:

`pass = false`

### Primary — seed 23031

Selected epoch: **8**

English:
- top-1: **0.6388888888888888**
- MRR: **0.7916666666666666**
- choice: 0.3333333333333333
- score: 0.6666666666666666
- noul: 0.9166666666666666

Vietnamese:
- top-1: **0.5416666666666666**
- MRR: **0.7453703703703703**
- choice: 0.2916666666666667
- score: 0.5833333333333334
- noul: 0.75

Cross-lingual:
- VI/EN top-1 ratio: **0.8478260869565217**
- VI/EN MRR ratio: **0.9415204678362573**
- paired prediction agreement: **0.5277777777777778**
- paired both-correct rate: 0.3888888888888889

Failed frozen gates:
- EN top-1 >= 0.65
- VI top-1 >= 0.65
- VI/EN top-1 ratio >= 0.90
- paired prediction agreement >= 0.85

Passed:
- VI/EN MRR ratio
- every runtime integrity gate

### Replica — seed 23037

Selected epoch: **6**

English:
- top-1: **0.5972222222222222**
- MRR: **0.7708333333333334**
- choice: 0.375
- score: 0.5833333333333334
- noul: 0.8333333333333334

Vietnamese:
- top-1: **0.5416666666666666**
- MRR: **0.7430555555555556**
- choice: 0.375
- score: 0.5833333333333334
- noul: 0.6666666666666666

Cross-lingual:
- VI/EN top-1 ratio: **0.9069767441860465**
- VI/EN MRR ratio: **0.963963963963964**
- paired prediction agreement: **0.5277777777777778**
- paired both-correct rate: 0.3888888888888889

Failed frozen gates:
- EN top-1 >= 0.65
- VI top-1 >= 0.65
- paired prediction agreement >= 0.85

Passed:
- VI/EN top-1 ratio
- VI/EN MRR ratio
- every runtime integrity gate

## 5. Scientific conclusion

The shared 8,192-parameter alignment adapter materially improved M3-A absolute quality and cross-lingual consistency:

Baseline:
- EN top-1: 0.4306
- VI top-1: 0.3889
- paired agreement: 0.1667

R1 primary:
- EN top-1: 0.6389
- VI top-1: 0.5417
- paired agreement: 0.5278

However, the improvement is insufficient for the preregistered multilingual promotion gate.

The remaining problem is not runtime integrity. The remaining problem is semantic representation/discrimination quality, especially:
- absolute VI accuracy;
- EN retention above 0.65;
- much stronger paired semantic agreement.

Therefore multilingual capability is **not promoted**.

## 6. Evidence boundary

Now permanently exposed:
- MVA / MVB
- MVD / MVE / MVF / MVG
- MVH / MVI

Forbidden for future:
- fitting;
- selection;
- threshold tuning;
- architecture ranking.

Still sealed and unexposed:
- **MVC**

MVC must remain sealed for future independent multilingual research.

Public MASSIVE/XNLI direct benchmark lanes remain outside M3 architecture selection.

## 7. Closure decision

There is **no M3-R2 blocking rescue**.

Mainline status after M3:

- typed runtime: available
- state-once: available
- high-K mechanics: available
- high-K semantic quality: provisional
- reliability/OOD/abstention: provisional/fail-closed
- multilingual: **provisional**
- production-ready: false

M3 closes without promoting multilingual.

Future multilingual semantic research becomes a parallel replaceable-module track and must not block the remaining Hira v0 mainline.

## 8. Next mainline phase

Next phase:

# HIRA V0 MAINLINE M4 — runtime / latency / RAM / packaging

M4 must optimize deployment characteristics without silently changing semantic maturity claims.

M4 may improve:
- schema compilation cost;
- caching;
- latency;
- memory;
- serialization;
- local runtime packaging;
- CPU execution path.

M4 must preserve all fail-closed maturity labels from M1-M3.
