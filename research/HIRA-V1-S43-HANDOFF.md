# HIRA V1 S43 handoff — to S44 Private Correction Representation Fork

S43 is frozen as:

`HIRA_V1_S43_MATCHED_CROSS_VIEW_RELATIVE_GAP_GEOMETRY_DEV_COMPLETE`

Interpretation:
**Case C — transport-critical relative-gap protection suppresses useful correctness co-adaptation.**

## Evidence chain

S38:
- unrestricted native co-adaptation
- strong correctness gain
- transport geometry collapses

S39:
- hard-isolated W-only correction
- transport preserved
- most correctness gain disappears

S41/S42:
- constrained native co-adaptation
- correctness partly retained
- discrimination/transport still degrades

S43:
- directly protects same-option-vs-wrong relative gaps
- signature-margin degradation shrinks dramatically
- fused/primary correctness collapses

This establishes a representation-ownership conflict.

## S44 scientific question

> Can Hira retain native transport geometry exactly while giving correctness a separate post-encoder representation that co-adapts only inside the correction branch?

## Architectural principle

Do not modify the encoder count.

Use the single already-computed native:
- option signature `s_native [256]`
- query summary `q [256]`
- native relation logits
- primary logits

Native runtime:
- exact matched S35/S17 trajectory
- trained exactly as the reference arm
- no correctness-branch gradient may enter native LoRA/projection/signatures

Private correction branch:
- receives **detached** native signature and detached query summary
- produces a private correction representation
- only the private representation and full bilinear W may receive correction gradients
- private representation is never consumed by native relation transport, primary path, or transport metrics

This preserves one encoder pass and state-once semantics.

## Candidate private adapter

Use one shared post-encoder nonlinear residual adapter per option:

Input:
`x = concat(stopgrad(s_native), stopgrad(q))` of width 512.

Hidden:
`h = GELU(A x)`

Private residual:
`d = B h`

Private signature:
`s_private = L2Norm(stopgrad(s_native) + d)`

Frozen initial architecture:
- hidden width **64**
- A: 64 x 512 = **32,768**
- B: 256 x 64 = **16,384**
- total private adapter **49,152**
- B zero-initialized
- A deterministic fixed initialization at authority seed
- both A and B trainable after initialization
- no bias
- no layer norm
- no gate
- no learned scale

Full bilinear correction:
`r_k = s_private_k^T W q`

W:
- 256 x 256
- **65,536**
- zero-init
- no bias/nonlinearity/scale beyond the private adapter.

Total correction-only trainable surface:
**114,688** = private adapter 49,152 + W 65,536.

Native runtime trainable surface remains **49,152**, but must match reference trajectory exactly.

## Why this is genuinely new

A purely linear private signature transform before W would be algebraically absorbable into W and would not add a new representational family.

The fixed-width GELU bottleneck makes the private post-encoder representation non-linear and non-absorbable into one bilinear matrix, while remaining small and state-once.

## Training ownership

Reference:
- exact native S35/S17 objective.

Treatment native runtime:
- exact same native objective, rows/order/init/optimizer as reference;
- must produce identical runtime trajectory fingerprints.

Correction branch:
- detached native signatures/query/primary logits as inputs;
- private adapter + W only;
- optimize a frozen correction objective using the corrected relation/fused decision path;
- no correction gradient to native LoRA/projection.

The correction objective must be preregistered before A0/DEV. It may reuse already-frozen S17/S38 loss coefficients, but no new coefficient may be chosen from S43 DEV.

## Required S44-A0

A fresh mechanical court must prove:
- native treatment/reference zero-init and update trajectory identity;
- private adapter parameter count exactly 49,152;
- W exactly 65,536;
- correction-only total exactly 114,688;
- one encoder batch only / state-once retained;
- private branch consumes detached native signature/query;
- correction loss -> A/B/W finite and nonzero after a deterministic warm-start probe;
- correction loss -> native LoRA/projection exactly zero;
- native objective -> private A/B/W exactly zero;
- B zero-init gives initial private/native signature identity;
- initial corrected/native logits identity with W zero-init;
- nonlinear private adapter is not reducible to the W-only S39 path under a nonzero probe;
- option permutation equivariance;
- question/padding invariance;
- arbitrary K=3/K=7;
- checkpoint roundtrip;
- full-K/probability mass;
- no projected relation path.

## Fresh authority

S44 must use:
- wholly fresh A0 diagnostic rows;
- wholly fresh TRAIN/DEV rows;
- no exact S0-S43 exposed rows;
- no S44-A0 rows in matched authority;
- one DEV only.

## Stop rule

After one S44 DEV:
- no hidden-width sweep;
- no activation sweep;
- no bias/gate/layernorm;
- no residual-scale tuning;
- no native-gradient leakage;
- no second encoder path;
- no second DEV;
- no seed/LR/epoch/batch retry;
- no gate weakening.

If S44 fails, accept the result and change family rather than tune exposed DEV.

No external Laya/Jev evaluation before a separately confirmed DEV_READY candidate.
