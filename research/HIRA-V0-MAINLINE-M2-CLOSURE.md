# HIRA V0 MAINLINE M2 closure — dynamic high-K

Status: **CLOSED — HIGH-K MECHANICS AVAILABLE / HIGH-K SEMANTICS PROVISIONAL**

Issue: #163  
PR: #164  
Branch: `feat/hira-v0-mainline-m2-high-k`  
Base main: `2920dc025c840245b77468f0b13626ed6dc57fac`

## 1. Purpose

M2 moved dynamic high-cardinality schemas into the actual Hira v0 mainline and separated two questions:

1. can the runtime execute full-K schemas through K=255 cleanly?
2. does the current semantic core rank the correct option well enough at high K?

The first question passed strongly. The second did not.

## 2. M2-A mechanics — READY

Authoritative integration run:

`36310118240`

Outcome:

`HIRA_V0_M2_MECHANICS_READY`

Artifact:
- `hira-v0-mainline-m2-mechanics-integration`
- ID: `10928771833`
- digest: `sha256:99e8c02aeb48fa32910a591d71b92e3e6f96258c0fa8501f38682ea4a2b7f453`

Exact runtime:
- resident parameters: 13,213,199
- trainable parameters: 0
- K ladder: 4 / 8 / 16 / 32 / 64 / 128 / 255
- state encode delta for whole ladder: 1
- dynamic queries: 14
- full-K: PASS at every K
- finite outputs: PASS at every K
- relation delta: 0
- max probability-mass error: 1.1920928955078125e-07
- max permutation probability error: 4.656612873077393e-10
- selected-option permutation invariance: PASS at every K

At K=255 on the exact CPU integration:
- schema compile: 1482.21 ms
- typed decision: 2.66 ms
- schema tensor surface: 4,477,609 bytes

Conclusion:

`high_k_mechanics = available`

The dominant scaling cost is semantic schema compilation, not the typed decision core.

## 3. Zero-training high-K semantic baseline — FAIL

Authoritative run:

`36311890201`

Artifact:
- `hira-v0-mainline-m2-semantic-dev`
- ID: `10928764745`
- digest: `sha256:18903d770c1917db94e395c49feb72c0e2fea1c01ef70e976c54ad40ed223148`

Outcome:

`HIRA_V0_M2_SEMANTIC_DEV_FAIL`

No new training occurred.

### HKE / K=64

- top-1: 0.625
- top-5: 0.75
- MRR: 0.6941468253968254
- mean gold rank: 5.375
- mean gold probability: 0.016648271412122995
- mean confidence: 0.016815155744552612

Frozen gate:
- top-1 >= 0.70
- top-5 >= 0.90
- MRR >= 0.75

Semantic: FAIL  
Mechanics: PASS

### HKF / K=128

- top-1: 0.3125
- top-5: 0.4375
- MRR: 0.37779609918896906
- mean gold rank: 28.9375
- mean gold probability: 0.008089849870884791
- mean confidence: 0.008322596666403115

Frozen gate:
- top-1 >= 0.60
- top-5 >= 0.85
- MRR >= 0.65

Semantic: FAIL  
Mechanics: PASS

HKG K=255 therefore remained sealed.

## 4. M2-R1 single semantic rescue

M2-R1 was preregistered as the only blocking rescue.

Frozen mechanism:
- exact W34 co-evidence base
- only W34 candidate modules trainable
- exactly 8,192 trainable parameters
- A13 frozen
- W28 projection frozen
- HIRACore frozen
- relation refinement OFF
- adaptive budget OFF
- candidate pruning OFF

TRAIN:
- HKA K=4
- HKB K=8
- HKC K=16
- HKD K=32
- 64 cases total

Fresh DEV:
- HKH K=64
- HKI K=128
- 48 cases total

Candidate factorial:
- CE primary / replica
- CE+margin primary / replica
- 8 epochs
- fixed seeds and fixed selection rule

HKE/HKF were forbidden for R1 fitting and selection.

## 5. M2-R1 authoritative result — FAIL

Authoritative run:

`36314952944`

Artifact:
- `hira-v0-mainline-m2-r1-train-dev`
- ID: `10929884694`
- digest: `sha256:6441bfacbabc7a184887fa3d2b6ad5b9f906373b34d18704bd21d2db5a21d202`
- size: 323,903,843 bytes

Outcome:

`HIRA_V0_M2_R1_DEV_FAIL`

Selected family:

`ce-margin`

Selected primary:

`ce-margin-primary`

Required corresponding replica:

`ce-margin-replica`

### Selected primary fresh DEV

K=64:
- top-1: 0.2916666666666667
- top-5: 0.5416666666666666
- MRR: 0.4203540513708857
- mean gold rank: 8.916666666666666
- full-K: 1.0
- state-once: 1.0
- finite: 1.0
- permutation max error: 1.862645149230957e-09
- probability-mass max error: 1.1920928955078125e-07
- relation delta: 0

K=128:
- top-1: 0.20833333333333334
- top-5: 0.625
- MRR: 0.3916255120996501
- mean gold rank: 9.791666666666666
- full-K: 1.0
- state-once: 1.0
- finite: 1.0
- permutation max error: 9.313225746154785e-10
- probability-mass max error: 1.1920928955078125e-07
- relation delta: 0

Primary gate: **FAIL**.

### Selected-family replica fresh DEV

K=64:
- top-1: 0.2916666666666667
- top-5: 0.625
- MRR: 0.4455692415358113
- mean gold rank: 9.5

K=128:
- top-1: 0.20833333333333334
- top-5: 0.5833333333333334
- MRR: 0.40606438088055735
- mean gold rank: 10.375

All runtime mechanics remained valid.

Replica gate: **FAIL**.

Therefore:

`sealed_exposure_authorized = false`

## 6. Scientific interpretation

M2 establishes a clean localization.

### Established

1. Full-K Hira mechanics scale correctly through K=255.
2. State-once behavior survives high K.
3. Numerical stability and permutation invariance are not the bottleneck.
4. The typed decision stage is cheap even at K=255.
5. The frozen W34 semantic geometry has useful high-K signal in the zero-training baseline.
6. Simply retuning the 8,192-parameter W34 co-evidence surface on K<=32 does not transfer to fresh K64/K128.
7. The R1 CE+margin specialization actually reduced fresh high-K top-1 relative to the zero-training baseline.

### Not established

M2 does not establish production-qualified semantic ranking at K=64/128/255.

Therefore:

`high_k = provisional`

must remain unchanged.

## 7. HKG evidence boundary

HKG K=255 was **never exposed**.

Because the preregistered R1 DEV gate failed:
- no M2-R1 confirm marker is authorized;
- no K=255 semantic claim is made;
- no threshold or architecture may be tuned against HKG inside M2.

HKG may remain as untouched sealed evidence for a future parallel high-K research track, but it is not part of the blocking mainline path.

## 8. Exposed evidence firewall

Now permanently exposed:
- HKE K=64
- HKF K=128
- HKA K=4
- HKB K=8
- HKC K=16
- HKD K=32
- HKH K=64
- HKI K=128

These rows are forbidden for future fitting, candidate selection, seed selection, epoch selection, margin tuning or architecture ranking.

HKG remains unexposed.

## 9. Product decision

M2 does not open another blocking rescue.

The following code is retained:
- high-K mainline mechanics;
- K<=255 fail-closed contract;
- state-once/full-K execution;
- mechanics authority and manifest support;
- high-K diagnostics/evaluation infrastructure.

The failed semantic candidates are research evidence only and are not promoted.

M2 closes with:
- `high_k_mechanics = available`
- `high_k = provisional`
- `production_ready = false`

Future high-K semantic research becomes a parallel replaceable semantic-core track.

## 10. Mainline transition

The next blocking phase is:

# HIRA V0 MAINLINE M3 — MULTILINGUAL

M3 must not reopen HKE/HKF/HKA-HKD/HKH/HKI.

Initial M3 objectives:
- preserve exact M0/M1/M2 runtime contracts;
- establish multilingual input semantics, beginning with English + Vietnamese;
- preserve typed choice / score / noul;
- preserve state-once;
- preserve dynamic schema mechanics;
- quantify cross-lingual semantic transfer separately from English quality;
- keep reliability provisional and fail-closed;
- keep high-K semantic quality provisional.

M2 is complete as a mainline phase.
