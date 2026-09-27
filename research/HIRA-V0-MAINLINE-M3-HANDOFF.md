# HIRA V0 MAINLINE M3 handoff — multilingual EN/VI

Status: **M3-A DEV FAIL / M3-R1 PREREGISTERED**

Issue: #165  
Branch: `feat/hira-v0-mainline-m3-multilingual`  
Base main: `aed5ec3ad0651e18e27a9fe77cd3ed6c33dceac4`

## 1. Mainline state entering M3

M0:
- integrated typed model shell;
- exact W28/W34 provenance;
- state-once dynamic typed runtime.

M1:
- reliability/OOD/abstention mechanism integrated;
- empirical production authority remains provisional/fail-closed.

M2:
- high-K mechanics through K=255 are available;
- high-K semantic quality remains provisional;
- HKG K255 semantic confirm remains unexposed.

M3 does not reopen M2 high-K evidence.

## 2. M3 objective

Establish whether the current Hira semantic path transfers between English and Vietnamese while preserving:
- choice / score / noul semantics;
- option IDs and typed values;
- state-once;
- dynamic schemas;
- full-K;
- probability integrity;
- relation-delta-zero.

M3-A measures the current frozen model before any multilingual rescue.

## 3. Exact M3-A model

Semantic front-end:
- A13
- `microsoft/xtremedistil-l6-h256-uncased`
- revision `4226d9e4d2c08703e5cb0491b479bfc6a1607181`
- weight SHA256 `5b0593e0bb4620631320d2b4d5604cc39ca53348a9a240392c6c0b830dd8d880`

Projection:
- W28 T0
- SHA256 `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`

Transfer:
- W34 provisional co-evidence
- SHA256 `d69fa11805291e6a06631d5bda941065f209ea96f5c46091187e984ff083834c`

High-K mechanics:
- authority `run:36310118240;artifact:10928771833;digest:sha256:99e8c02aeb48fa32910a591d71b92e3e6f96258c0fa8501f38682ea4a2b7f453`

M3-A trainable parameter count:
- **0**

No semantic front-end replacement or adaptation is allowed before the M3-A receipt freezes.

## 4. Fresh paired authority

DEV:
- MVA: 36 latent pairs
- MVB: 36 latent pairs
- total: 72 EN/VI pairs = 144 language cases

Sealed CONFIRM:
- MVC: 36 latent pairs = 72 language cases

Each domain contains:
- 12 choice
- 12 score
- 12 noul

Each EN/VI pair preserves:
- pair ID;
- primitive;
- gold index;
- option IDs;
- typed option values.

But all of the following are language-specific:
- state text;
- question text;
- option criterion text;
- aliases.

This prevents a state-only translation shortcut.

## 5. M3-A frozen DEV gates

Absolute:
- EN top-1 >= 0.65
- VI top-1 >= 0.65

Relative:
- VI/EN top-1 ratio >= 0.90
- VI/EN MRR ratio >= 0.90
- paired prediction agreement >= 0.85

Runtime:
- one state encode per language case;
- full-K;
- finite;
- probability mass error <= 1e-6;
- relation delta = 0;
- no gradient updates;
- no candidate pruning;
- no relation refinement;
- no adaptive budget.

DEV pass authorizes MVC exposure exactly once.

DEV fail leaves MVC sealed and triggers only the preregistered M3-R1 design step.

## 6. Evidence firewall

Forbidden for M3-A fitting/selection:
- MASSIVE direct benchmark rows;
- XNLI direct benchmark rows;
- W34 confirm rows;
- all exposed M2 semantic rows HKA-HKF/HKH/HKI;
- M1 sealed reliability rows.

M3-A uses zero fitting anyway; these restrictions remain explicit for future rescue design.

## 7. Sealed evaluator

`scripts/hira_v0_m3_paired_confirm.py` is implemented before DEV exposure.

It:
- refuses a non-qualified DEV receipt;
- refuses prior MVC exposure;
- reloads exact A13/W28/W34 provenance;
- runs MVC exactly once;
- uses the same frozen qualification thresholds;
- performs no training or selection.

Possible outcomes:
1. `HIRA_V0_M3_MULTILINGUAL_READY`
2. `HIRA_V0_M3_MULTILINGUAL_FAIL`

Only outcome 1 may promote multilingual capability.

## 8. M3-R1 rule

No M3-R1 encoder/front-end candidate is selected before M3-A DEV freezes.

If M3-A fails because VI representation is materially weaker than EN, M3-R1 may compare compact multilingual front-end mechanisms using wholly fresh TRAIN/DEV evidence.

MASSIVE/XNLI direct lanes remain final/public benchmark evidence, not architecture-selection data.

M3 has at most one blocking rescue before mainline proceeds.

## 9. Current boundary

No M3 empirical authority has been exposed.

Next:
1. pass dedicated M3 contracts;
2. pass full repository Python 3.10/3.12 CI;
3. create `research/HIRA-V0-MAINLINE-M3-ENABLE-DEV`;
4. expose only MVA/MVB;
5. freeze DEV receipt;
6. create confirm marker only if DEV qualification passes.


## 10. M3-A authoritative DEV result — FAIL

Run:

`36320497057`

Artifact:
- `hira-v0-mainline-m3-paired-dev`
- ID: `10932028604`
- digest: `sha256:d9837f68ed0f9a7d216c42a95e3cc9977fa5f3d218aa792493ecfcf63a47d6bd`

Outcome:

`HIRA_V0_M3_PAIRED_DEV_FAIL`

Exact zero-training metrics:

English:
- top-1: **0.4305555555555556**
- MRR: **0.6782407407407408**
- mean gold rank: 1.7916666666666667
- choice top-1: 0.375
- score top-1: 0.4166666666666667
- noul top-1: 0.5

Vietnamese:
- top-1: **0.3888888888888889**
- MRR: **0.6643518518518519**
- mean gold rank: 1.7916666666666667
- choice top-1: 0.3333333333333333
- score top-1: 0.3333333333333333
- noul top-1: 0.5

Cross-lingual:
- VI/EN top-1 ratio: **0.9032258064516129** — PASS relative ratio
- VI/EN MRR ratio: **0.9795221843003412** — PASS relative ratio
- paired prediction agreement: **0.16666666666666666** — FAIL
- paired both-correct rate: 0.05555555555555555

Runtime:
- state-once: PASS
- full-K: PASS
- finite: PASS
- probability mass max error: 1.1920928955078125e-07
- relation delta: 0
- gradient updates: false

Frozen gate failures:
- EN absolute top-1
- VI absolute top-1
- paired prediction agreement

Frozen gate passes:
- VI/EN top-1 ratio
- VI/EN MRR ratio
- every runtime integrity gate

Scientific localization:

The failure is not simply "Vietnamese is much weaker than English". Both languages have weak absolute semantic discrimination while VI retains roughly 90%+ of EN marginal quality. At the same time, EN/VI paired predictions agree only 16.67%.

Therefore M3-R1 targets **shared cross-lingual semantic geometry and absolute typed discrimination together**, not a Vietnamese-only patch.

MVC remains fully sealed.

## 11. M3-R1 — single blocking alignment rescue

Only one blocking M3 rescue is authorized.

### Mechanism

Insert a shared identity-initialized rank-16 residual semantic alignment adapter around the frozen A13 output:

```
A(x) = x + U(gelu(D(layer_norm(x))))
```

where:
- D: 256 -> 16, bias-free
- U: 16 -> 256, bias-free
- U starts at exact zero
- layer norm is non-affine
- total trainable parameters: **8,192**

The same adapter is used for:
- state tokens;
- state pooled embedding;
- question/schema text;
- option/schema text.

There is no language-specific branch and no language-ID feature.

### Frozen base

Must remain frozen:
- A13 parameters;
- W28 projection;
- W34 co-evidence scorer;
- HIRACore;
- reliability components.

Only the 8,192 alignment-adapter parameters may receive gradients.

### Fresh R1 authority

TRAIN:
- MVD
- MVE
- MVF
- MVG
- 4 fresh domains
- 144 EN/VI latent pairs = 288 language cases

Fresh DEV:
- MVH
- MVI
- 72 EN/VI latent pairs = 144 language cases

Forbidden:
- MVA/MVB fitting or selection
- MVC exposure
- MASSIVE/XNLI direct rows
- M2 semantic rows
- W34 confirm rows

### Loss

Frozen training objective:

```
L = task_ce + 0.35 * paired_alignment + 0.10 * english_anchor
```

- task CE: typed gold CE on both EN and VI
- paired alignment: cosine alignment for paired EN/VI state/question/option embeddings
- English anchor: squared drift from frozen A13 English representation

### Optimization

- epochs: 8
- AdamW
- lr: 2e-4
- weight decay: 0.01
- gradient clip: 1.0

Two fixed seeds:
- primary: 23031
- replica: 23037

### Epoch selection

Select within each seed using fresh R1 DEV only:

1. larger min(EN top1, VI top1)
2. larger paired prediction agreement
3. larger min(VI/EN top1 ratio, VI/EN MRR ratio)
4. larger min(EN MRR, VI MRR)
5. earlier epoch

### R1 DEV qualification

Both primary and replica must independently pass the original M3 gates:

- EN top-1 >= 0.65
- VI top-1 >= 0.65
- VI/EN top-1 ratio >= 0.90
- VI/EN MRR ratio >= 0.90
- paired prediction agreement >= 0.85
- all runtime integrity gates PASS

Only then may MVC be exposed once using the primary checkpoint.

If either primary or replica fails, M3 closes with multilingual provisional and MVC remains sealed. No M3-R2 blocking rescue is authorized.
