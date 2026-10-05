# HIRA V1 S63 closure

Status: **CLOSED / CASE B**

S63 asked whether a compact learned permutation-invariant representation of the full detached fused/pairwise decision surfaces could predict S62 reliability better than four hand-crafted scalar features.

Fresh authority:
- run `37314015283`
- artifact `11347706059`
- digest `sha256:dcb69358dfa97f3644198534faa1c2f5f214b4f1773d79008eec13cafaf2a97e`
- scientific head `ddfa724edc8665518489dc82a7b22267081d82ed`.

Answer: **no, not from decision-surface geometry alone.**

Treatment vs reference:
- canonical accuracy **+0.26 pp**
- paraphrase accuracy **0.00 pp**
- paired both-correct **0.00 pp**
- question-swap discrimination **+0.52 pp**
- selected-choice agreement **0.00 pp**
- cross-view JS **+0.00000927 worse**.

The 61-param treatment clearly changes gate policy, but does not materially improve stability.

Frozen verdict: **Case B**.

Authorized handoff:
- retain S62 correctness-veto reliability target;
- retain bounded residual and no pairwise-only path;
- retain detached gate ownership;
- move reliability prediction beyond fused/pairwise surface geometry into richer state/query/option-conditioned representation;
- do not retry or retune S63.

S63 is scientifically exhausted.
