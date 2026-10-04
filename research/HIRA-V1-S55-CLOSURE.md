# HIRA V1 S55 closure — Learned Joint Relation Interaction

Status: **SCIENTIFICALLY CLOSED — CASE C**

Issue: #293  
PR: #294

## Valid court

Fresh scientific run:
- `37212962160`
- artifact `11307227880`
- digest `sha256:7059cf8477eb4bd775e89badb0c734a2d351afe2e1a1d23ceaf4f7189c7744e4`

The earlier run `37212252407` was a documented pre-DEV mechanical abort and exposed no private DEV/model-selection evidence.

## Verdict

**Case C.**

Treatment minus reference:
- fused canonical **+4.43 pp**
- fused paraphrase **-3.39 pp**
- paired both-correct **+4.69 pp**
- fused agreement **-2.86 pp**
- fused JS **-0.002865** better
- canonical relation accuracy **+6.77 pp**
- paraphrase relation accuracy **-2.34 pp**
- relation agreement **-4.95 pp**
- relation JS **-0.011288** better.

Both arms remain DEV_READY false.

## Scientific conclusion

The learned joint transform is active and the explicit state channel is genuinely used, but it still does not improve selected-choice stability.

Across S51–S55 we have now tested:
- query-free option identity;
- learned global query canonicalization;
- token-level query↔option interaction;
- parameter-free state-query-option interaction;
- learned state-query-option relation mapping.

None produced the required stability improvement.

Therefore the **correction-factorization family is exhausted** for Hira v1.

S56 must move to **explicit cross-view decision consistency**: train the final decision distribution/ordering itself to remain invariant across paraphrase/state-view variants while preserving correctness and relation separation.

## Stop-rule compliance

No:
- hidden-dimension/seed/channel-scale sweep
- neutral-channel variant
- bypass
- native retraining
- identity/correction capacity change
- selector change
- scientific retry
- gate weakening
- second S55 DEV
- external Laya/Jev evaluation.

S55 is closed.
