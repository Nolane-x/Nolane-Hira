# HIRA V1 S48 handoff — to S49 Private Query-Free State–Option Identity Signature

S48 is scientifically closed as a **dominated negative**.

## Combined evidence from S35-S48

The sequence now localizes the bottleneck tightly:

- **S35:** native 256D relation geometry improves transport but loses correctness.
- **S36/S37:** global/diagonal readouts cannot recover enough correctness.
- **S38:** full bilinear query×signature readout unlocks correctness but harms transport.
- **S39-S43:** gradient isolation/anchoring cannot retain both correctness and relative option geometry.
- **S44:** private correction solves ownership conflict and keeps native transport bitwise intact.
- **S45:** JS regularization does not stabilize the private expert.
- **S46:** magnitude-robust median fusion fails.
- **S47:** magnitude-free ordinal consensus fails.
- **S48:** removing query components orthogonal to option differences also fails; correctness and selected-choice stability both regress.

The common residual in S44-S48 is:

> the private correction consumes a **native relation signature already produced by a question-conditioned relation operator**.

Changing the query readout after that point cannot undo option identity drift that has already entered the signature.

## S49 scientific direction

Build a **private query-free state–option identity signature** from the same adapted A13 tokens.

The native S35/S44 path remains untouched.

For the private branch only:

1. use detached adapted state tokens;
2. use detached adapted option-view tokens;
3. do **not** use question tokens to choose state roles, option roles, or pair weights;
4. compute a zero-parameter state↔option identity signature;
5. feed that query-free option identity into the existing private A/B/W correction;
6. let the raw normalized query enter only at the final private correctness readout.

Thus:
- option identity is formed without wording dependence;
- query semantics still control which relation is requested;
- the native relation representation remains untouched;
- one encoder pass is preserved.

## What S49 must not repeat

Do not recreate:
- S33 query-explicit additive coordinates;
- S34 query-conditioned entropic transport;
- S35 native relation replacement;
- S36/S37 low-rank correctness readouts;
- S38 native-path bilinear co-adaptation;
- S42/S43 anchoring;
- S48 query quotient.

The new variable is specifically **where query dependence first enters the private representation**.

## Proposed private identity operator

Given normalized detached state tokens `s_i` and option-view tokens `o_{kvt}`:

- state center = masked mean of state tokens;
- option center = masked mean per option view;
- relative state tokens = normalize(`s_i - state_center`);
- relative option tokens = normalize(`o_{kvt} - option_center_{kv}`);
- pair score = average of direct cosine and relative cosine;
- pair softmax is over state×option-token pairs using the existing frozen pair temperature;
- aggregate matched state and option features;
- add the query-free option-center minus state-center direction;
- average active option views;
- normalize to one **query-free 256D identity signature per option**.

No question token participates in this identity operator.

The existing raw query summary is then used by the private A/B/W correction against this identity signature.

## S49-A0 must prove

- identity signatures are exactly invariant to question-token substitution;
- relation-distinct raw queries remain able to alter correction logits through A/B/W;
- option permutation equivariance;
- state-token and option-token permutation invariance;
- K=3/7/255;
- finite degenerate geometry;
- one encoder pass/state-once;
- zero added identity parameters;
- same correction capacity **114,688**;
- same total treatment **163,840**;
- no gradient from private correction into native runtime;
- native objective cannot update private branch;
- W -> B -> A warm-start remains live.

## Fresh matched court

Reference:
- exact S44/S45 raw-query correction over question-conditioned native relation signatures.

Treatment:
- same A/B/W parameters and initialization;
- raw query unchanged;
- only the private option signature becomes query-free state↔option identity.

Use wholly fresh S49 TRAIN/DEV rows and one DEV authority only.

If S49 fails, the next bottleneck is no longer query contamination of private option identity; it points to the underlying state↔option semantic matching geometry itself.
