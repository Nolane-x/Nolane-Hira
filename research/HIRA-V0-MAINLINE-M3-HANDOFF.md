# HIRA V0 MAINLINE M3 handoff — multilingual EN/VI

Status: **M3-A ZERO-TRAINING PRE-EXPOSURE**

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
