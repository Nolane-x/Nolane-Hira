# R8-W30 handoff — low-rank semantic-transfer bridge

Status: **PRE-EXPOSURE / IMPLEMENTATION**

Issue: #149  
Branch: `feat/r8-w30-semantic-transfer-bridge`  
Base main: `bc8827d3a86a7ade940e6d66ea72287aa6d624f9`

## 0. Why W30 exists

W29 closed with `W29_RUNTIME_INTEGRATION_FAIL`.

The important localization is asymmetric:

- reference semantics passed on EW/EX/EY/EZ;
- runtime integrity passed globally and per-domain;
- state-once, full-K, option-order invariance and primitive agreement passed;
- HIRA semantic transfer failed, especially F1/F2.

Therefore W30 does not reopen typed wrappers, caching, candidate selection, relation refinement, or probability normalization.

It tests one narrow hypothesis:

> the frozen W28 T0 geometry is useful but too wording-regime-specific; a tiny shared residual map in the 128-d rescued space may improve state/schema paraphrase transfer without destroying the compact architecture.

W29 EW/EX/EY/EZ are permanently exposed and are not training/dev/confirm evidence for W30.

## 1. Frozen base

- A13 encoder: frozen;
- exact W28 T0 projection: frozen;
- T0 checkpoint SHA256:
  `1ed6c94d179fddffa2859a67ee3f9f383e677d456365d7e87bdcd844cc49010f`;
- original W29 `SymmetricSemanticScorer`: unchanged baseline;
- HIRACore relation refinement: OFF for W30;
- no primitive-specific or factor-specific learned head.

## 2. Candidate: shared rank-8 residual bridge

After the frozen T0 projection and before normalization:

```
z = T0(x)
z' = z + up(down(z))
y = normalize(z')
```

Frozen dimensions:
- T0: 256 -> 128, bias-free, frozen;
- down: 128 -> 8, bias-free;
- up: 8 -> 128, bias-free;
- bridge parameters: 2,048;
- same bridge for state tokens and option-view tokens.

Initialization:
- down: normal Kaiming-compatible PyTorch linear initialization;
- up: exact zeros;
- therefore the initial bridged scorer is functionally identical to W29 T0.

The bridge is the only trainable component in W30 training.

## 3. Runtime boundary

Add a separate coarse mode:

`bridged_symmetric_semantic`

Do not mutate `symmetric_semantic`.

Both modes:
- consume state content tokens;
- consume positive multi-view option token artifacts;
- emit full-K logits;
- bypass relation refinement by default;
- bypass reliability calibration until a later gate.

W30 runtime must preserve state-once typed fan-out.

## 4. Fresh authority

Partitions:
- TRAIN: 4 fresh domains;
- DEV: 1 fresh domain;
- CONFIRM: 2 sealed fresh domains.

Each domain:
- 96 cases;
- 24 per severity S0/S1/S2/S3;
- F0/F1/U/C are deterministic;
- F2 = U AND C;
- severity vectors are 000, 100, 110, 111.

Text policy:
- no exact W29 or older authority sentence reuse;
- TRAIN, DEV and CONFIRM use separate paraphrase/style banks;
- schema views in CONFIRM are held out from TRAIN/DEV;
- exact-text overlap checks are mandatory before any empirical run.

## 5. Training contract

Only 2,048 bridge parameters train.

- seed: 3011
- optimizer: AdamW
- lr: 3e-4
- weight decay: 0.01
- epochs: 12
- logical batch: 32
- grad clip: 1.0
- epoch selection: DEV only

Loss:
- F0 + F1 + F2 cross-entropy;
- exact bridged symmetric multi-view scorer;
- no W29 row/prediction/output in loss or selection.

The T0 projection must have `requires_grad=False` for the entire run.

## 6. Baseline

On every DEV/CONFIRM authority, evaluate both:

- B0 = frozen unbridged W29 T0 scorer;
- B1 = frozen T0 + selected bridge.

No baseline tuning is allowed.

## 7. Sealed CONFIRM gate

Every confirm domain must meet for B1:
- F0 top1 >= .90;
- F1 top1 >= .90;
- F2 top1 >= .90;
- each factor balanced accuracy >= .88;
- factor-vector top1 >= .82;
- composed severity top1 >= .82;
- invalid vector rate <= .05;
- option-order invariance = 1.0;
- full-K rate = 1.0;
- probability mass max error <= 1e-6.

Transfer delta:
- composed severity B1 - B0 >= +.20 absolute pooled;
- no factor top1 may regress by more than .03 absolute pooled.

Runtime:
- one state encode per case for typed fan-out;
- relation delta = 0;
- candidate truncation = false;
- T0 exact;
- inference bridge frozen.

## 8. Reference gate

The W28 external NLI panel may be used only to validate semantics.

Reference output must never become a HIRA input, target, loss term, selector, or feature.

Reference failure has absolute precedence.

## 9. Frozen outcomes

1. `W30_REFERENCE_INADEQUATE`
2. `W30_TRANSFER_BRIDGE_FAIL`
3. `HIRA_V0_TRANSFER_CORE_READY`

No partial promotion.

## 10. Evidence firewall

Forbidden for W30 fitting/selection:
- W29 EW/EX/EY/EZ rows, labels, predictions and wording;
- W28 and older empirical authority rows;
- Banking77 final/test;
- typed final/test;
- Laya/JEV result cells.

W29 aggregate metrics are historical motivation only.

## 11. Deliverables

Before empirical exposure:
1. bridge implementation;
2. runtime mode;
3. unit + backward compatibility tests;
4. fresh authority generator;
5. train/dev cache builder;
6. trainer/dev selector;
7. sealed confirm evaluator;
8. CI/workflow gates;
9. exact evidence-overlap firewall.

At closure:
- authoritative audit;
- `R8-W30-CLOSURE.md`;
- FULL source/research ZIP;
- standalone HANDOFF;
- SHA256 integrity manifest.

## 12. Promotion boundary

Only `HIRA_V0_TRANSFER_CORE_READY` permits the next gates:
- calibration;
- OOD/null abstention;
- high-K;
- latency/RAM;
- broader transfer.

No Laya/JEV superiority claim is authorized by W30 alone.
