# HIRA V1 S51 handoff — to S52 Paired-View Query Relation Canonicalization

S51 closes as **Case C**.

## Key evidence

The query-free option identity is geometrically stable:
- same-option cosine **0.943834**
- same-vs-strongest-wrong margin **0.177716**

Treatment correctness improves modestly:
- canonical **+2.08 pp**
- paraphrase **+2.08 pp**
- paired both-correct **+4.69 pp**

Yet stability does not improve:
- fused agreement **-0.52 pp**
- fused JS worse
- corrected relation agreement **-10.94 pp**.

Because the option identity is fixed/query-free, this isolates the remaining instability primarily to the **query-conditioned relation readout**.

## S52 scientific direction

**S52 — Paired-View Query Relation Canonicalization**

Keep the S51 persisted native authority and query-free option identity substrate.

Do not change native evidence generation or final option identity geometry first.

Introduce a small private query-relation canonicalizer operating only on detached query tokens.

### Controlled family

Both arms use:
- exact same persisted native authority;
- exact same fresh S52 TRAIN/DEV cache;
- query-free state↔option identity;
- identical correction A/B/W surface;
- identical query-canonicalizer architecture/parameter count and initialization;
- zero native trainability.

Reference:
- query canonicalizer trained only through the existing private correctness objective.

Treatment:
- same canonicalizer plus a preregistered paired-view **relation-code consistency** objective that aligns canonical/paraphrase query codes for the same latent relation.

The controlled variable is the internal relation-code consistency objective, not capacity.

### Canonicalizer target

Use a compact residual bottleneck:
- input/output 256
- bottleneck 64
- A: 64×256
- B: 256×64
- added parameters **32,768 per arm**
- identical initialization
- zero-init residual output path for stable warm start.

The final query code remains 256D and feeds the same private correction readout.

### Anti-collapse requirement

Paired-view alignment alone may collapse.

A0/training must include a fixed relation-separation constraint using the paired A-vs-B questions within each semantic case:
- same-relation paraphrases should align;
- different-relation questions from the same state should remain separated by a preregistered margin or contrastive objective.

No data-derived margin sweep.

### Why S52 differs from S48

S48 projected raw query through the current option-difference subspace with zero learned capacity and destroyed useful information.

S52 does not quotient the query against current option scores.

It learns a tiny **relation-code canonicalization map** directly from paired paraphrases while preserving a contrastive relation-separation signal.

## S52 required mechanics

- persisted native artifact pinned before S52 data exposure;
- fresh S52 TRAIN/DEV rows;
- native optimizer absent;
- one shared immutable evidence cache;
- query-free option identity unchanged;
- query canonicalizer 32,768 params in both arms;
- correction 114,688 params in both arms;
- total private trainable per arm **147,456**
- bit-identical initialization;
- no second encoder;
- full-K;
- K=3/7/255 A0;
- option permutation;
- query-code paraphrase alignment diagnostic;
- relation-separation diagnostic;
- no raw-query bypass around the canonicalizer in either arm.

## Fresh interpretation

A — treatment improves stability while retaining useful correctness:
direct query-relation canonicalization is useful.

B — stability improves but correctness collapses:
canonicalizer over-smooths relation information.

C — correctness remains but stability still does not improve:
instability lies deeper than query relation code.

D — both regress:
reject paired-view query canonicalization.

E — full DEV_READY:
freeze and open confirmation before external Laya/Jev evaluation.

One S52 DEV only.
