# HIRA V1 S53 A0 receipt — Token-Level Query↔Option Late Interaction

Status: **QUALIFIED**

Run: `37205099128`  
Artifact: `11304188134`  
Artifact digest: `sha256:08dbedcf29c7c71646c0c17a7e079de7df4d763ea0888db01282887867dbc841`  
Authorization head: `35ae7f9aa2aa9dc90e5f1a74d172527ffc14ac8c`

Outcome:
`HIRA_V1_S53_A0_TOKEN_QUERY_OPTION_LATE_INTERACTION_READY`

## Parent authority

- S51 native run `37192490832`
- runtime/native digest:
  `ef205661a9a8e8fb96518444be3baf8588968f685fb4622dc2aed423dd30a628`
- checkpoint SHA:
  `19104b46c88cb3dcc0e4f4fe3c5f61a98cb7bb646184075dee3762566e6d5916`
- native trainable params **0**

## Matched capacity

Reference:
- correction **114,688**
- private trainable **114,688**

Treatment:
- correction **114,688**
- late-interaction params **0**
- identity params **0**
- private trainable **114,688**

Correction initialization is bit-identical.

## Actual cached-evidence mechanics

- cache digest `d9ce0e242d8b38ae5ca0bd46b18935f821f6b140d6a350c68cab6e114ea8d283`
- inference tensors **0**
- requires-grad tensors **0**
- attention mass max error **2.384185791015625e-7**
- context norm max error **1.1920928955078125e-7**
- per-option context diversity max abs **0.005238795652985573**
- pooled query bypass absent **true**
- informative-token context sensitivity **0.1946394294500351**

At the exact frozen zero-init checkpoint:
- informative-token logit sensitivity = **0**

This is expected and is **not a broken path**:
the inherited S44 correction initializes both `adapter_b` and
`bilinear_weight` to exact zero, so every correction residual is exactly zero
before any private update.

The A0 contract suite separately executed
`test_s53_informative_token_perturbation_changes_context_and_logits`,
which deterministically activates the existing correction B/W weights without
changing architecture or scientific training initialization and requires both:
- context sensitivity > 0
- logit sensitivity > 0.

That test passed inside the qualified A0 workflow before the receipt was accepted.

Therefore:
- treatment token path is mechanically live;
- actual scientific training still starts from the frozen zero-init correction;
- no initialization variable was changed.

## Full-K / invariants

- K=3 PASS
- K=7 PASS
- K=255 PASS
- probability mass max error **1.1920928955078125e-7**
- max context norm error **1.7881393432617188e-7**
- query padding/mask invariance PASS
- query-token permutation invariance PASS
- option permutation equivariance PASS
- all-masked query rejection PASS
- second encoder pass **false**

A0 is diagnostic only:
- model selection **false**
- production-ready claim **false**

Fresh S53 TRAIN/DEV remains separately gated.
