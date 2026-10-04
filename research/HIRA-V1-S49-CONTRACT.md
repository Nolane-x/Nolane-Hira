# HIRA V1 S49 contract — Private Query-Free State–Option Identity Signature

Status: **PREREGISTERED / NO S49-A0 EXPOSURE**

Issue: #281

Parent:
- S48 fresh matched run `37172187841`
- S48 classification **DOMINATED NEGATIVE**
- S48 merged main `721b506688044f75b142e7a0f41a63cf74663674`

## Scientific question

> Can Hira stabilize the private correctness branch by constructing option identity without question conditioning, while retaining raw query semantics only at the final private readout?

## Frozen native/runtime surface

Keep exact S44/S45 native runtime:
- native trainable **49,152**
- correction-only **114,688**
- total treatment **163,840**
- one encoder batch / state-once
- exact native relation path unchanged
- correction gradients cannot update native runtime.

## Matched arms

Reference:
- existing question-conditioned native relation signature;
- existing raw normalized query summary;
- exact private A/B/W correction.

Treatment:
- same A/B/W shape, initialization and optimizer;
- raw query summary unchanged;
- replace only the private option signature with a query-free state↔option identity signature.

No correction parameter is added or removed.

## Frozen query-free identity operator

Inputs:
- detached adapted A13 state tokens `s_i`
- state mask
- detached adapted option-view tokens `o_{kvt}`
- option token/view masks.

Question tokens are forbidden.

1. Normalize state and option tokens.
2. Masked state center = normalized masked mean of state tokens.
3. Masked option center = normalized masked mean for each option view.
4. Relative state token = normalize(`s_i - state_center`).
5. Relative option token = normalize(`o_{kvt} - option_center_{kv}`).
6. Pair score:
   `0.5 * (dot(state,option) + dot(relative_state,relative_option))`.
7. Mask invalid pairs.
8. Softmax over all state×option-token pairs with frozen temperature **0.10**.
9. Aggregate matched state feature and matched option feature.
10. Center direction = normalize(`option_center - state_center`).
11. View identity = normalize(`matched_state + matched_option + center_direction`).
12. Mean active views and normalize to one 256D identity signature per option.

Identity trainable params: **0**.

## Private correctness readout

Treatment identity is passed to the existing private correction fork as the `signatures` input.

Raw normalized query still enters:
- A/B concatenation
- bilinear W readout

exactly as in S44/S45.

Thus:
- option identity is question-free;
- requested relation remains query-dependent;
- native transport path is unchanged.

## Required S49-A0

Identity:
- params exactly 0
- K=3,7,255
- question substitution has no effect on identity
- option permutation equivariance
- state-token permutation invariance
- option-token permutation invariance inside each view
- option-view permutation invariance
- masked-padding invariance
- finite degenerate geometry
- controlled distinct options yield distinct identities.

Private correction:
- correction params exactly **114,688**
- total treatment exactly **163,840**
- raw query remains live after identity construction
- relation-distinct queries can change correction logits/ranking while identity is fixed
- zero-B/W initialization retains expected identity mechanics
- full-K/probability mass valid.

Ownership/runtime:
- one encoder batch/state-once
- no second encoder
- private correction -> native runtime gradient exactly 0
- native objective -> private A/B/W gradient exactly 0
- matched native one-step parameter/output identity
- W -> B -> A warm-start live
- checkpoint roundtrip exact.

A0 is diagnostic only.

## Fresh TRAIN/DEV authority

Only after:
1. S49 contract and interpretation plan frozen;
2. core + tests frozen;
3. exact-head generic CI PASS;
4. one qualified S49-A0;
5. A0 receipt frozen;
6. fresh S49 authority/trainer/workflow frozen;
7. exact final pre-DEV CI PASS;
8. separate one-shot marker.

Intended:
- seed **70001**
- TRAIN **768**
- DEV **192**
- 12 wholly fresh S49 domains
- K=4
- 24 epochs
- batch 16
- one DEV only.

## Stop rule

After one S49 DEV:
- no pair-temperature sweep
- no direct/relative mixture sweep
- no query leak into identity construction
- no raw/native signature blend
- no learned identity projector
- no identity width/rank sweep
- no correction capacity/optimizer/loss-weight change
- no second encoder
- no seed/LR/epoch retry
- no gate weakening
- no second DEV.

Scientific failure is valid.
