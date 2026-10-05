# HIRA V1 S67 closure

Status: **CLOSED / CASE B**

Unique fresh S67 scientific run:
- run `37336779415`
- scientific head `2d74cab9ad6357ac8f2bc6224df46d97dd7e1f4f`
- outcome `HIRA_V1_S67_SAFE_ORACLE_ALPHA_DEV_COMPLETE`.

The fresh court completed TRAIN/DEV successfully. Its post-run receipt verifier failed on stale key `parent_s65`; the emitted receipt correctly points to `parent_s66`.

No second DEV is authorized.

No Actions artifact exists because upload was skipped after the verifier failure. The unique raw scientific receipt is frozen from the run log in `HIRA-V1-S67-MATCHED-RECEIPT.json`.

Treatment vs reference:
- canonical **0.00 pp**
- paraphrase **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap **0.00 pp**
- agreement **0.00 pp**
- JS **-0.000000599927**.

TRAIN oracle supervision was not degenerate:
- five target levels were active;
- non-binary targets **14.68%**;
- view-target disagreement **49.41%**.

Frozen verdict: **Case B**.

Scientific conclusion:
the S62→S67 sequence has now exhausted shared/pair, context, cross-view, per-view binary and multi-level safe-alpha supervision under the scalar bounded-residual mechanism. The next intervention must move upstream into the pairwise evidence/decision mechanism.

No lattice change, target smoothing, target mixing, retry, second DEV or external Laya/Jev evaluation is authorized.
