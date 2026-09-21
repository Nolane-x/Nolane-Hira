# RD-SD-HIRA architecture contract

```text
State -> semantic compiler (once) -> hierarchical StateMemory
                                      |        |
Registered schema -> compile/cache ---+        |
Dynamic schema ------ encode q/options --------+
Sparse evidence -------------------------------+
                                               v
                                      full-K coarse logits
                                               |
                                multi-prototype -> logical options
                                               |
                                  candidate authority / budget
                                               |
                                      HIRA relation refinement
                                               |
                                 coarse logits + selected deltas
                                               |
                                   full-K typed normalization
                                               |
                                OOD + calibration + authority
                                               |
                                    Choice / Score / Noul
```

## Frozen invariants

- one semantic state encode per state;
- registered schemas may cache question/options and invalidate cache by encoder+tokenizer+schema hash;
- K is logical-option cardinality;
- selected candidates receive deltas, not replacement logits;
- unselected tails remain in the final normalizer;
- OOD is an independent receipt and may force escalation;
- dynamic mode forbids sparse-only early exit;
- every decision can emit an auditable receipt.

## Backbone roles

- **A6 rival:** size/dynamic-routing challenger; never assumed weaker because it is smaller.
- **A13:** first proof model / semantic capacity reference.
- **A22:** diagnostic capacity control only if A13 capacity trigger fires.
- **A7-FE24:** post-proof compression lane using 24k EN+VI tokenizer, 96-d factorized embedding, 6x256 contextual body.
