# R8-W16 prior-art boundary — matched semantic-regime transfer decomposition

Frozen before any W16 empirical exposure.

W16 does not claim novelty for paraphrase robustness, query/schema semantic alignment, late interaction, query-noise analysis, or token masking.

## 1. Paraphrase sensitivity is a real reliability axis

Alhetelah & Ahmad, WASSA 2026, "Measuring LLMs' Sensitivity to Paraphrased Opinion Prompts":
https://aclanthology.org/2026.wassa-1.5/

The paper studies deterministic decisions under meaning-preserving paraphrases and shows that semantically equivalent wording can change model behavior. W16 uses this only to motivate controlling wording/rendering as an experimental variable.

## 2. Explicit schema semantics matter for routing

Wu, Wang & Nguyen, Findings of ACL 2026, "Bidirectional Semantic Enhancement for Schema Routing Across Large-Scale Databases":
https://aclanthology.org/2026.findings-acl.369/

The work emphasizes query-intent/schema-description correspondence for compact dense routing. W16 does not import its method. It supports treating schema rendering and semantic alignment as first-class variables rather than assuming task labels are interchangeable.

## 3. Not every extra term is useful

Chen et al., ACL 2025, "Not All Terms Matter: Recall-Oriented Adaptive Learning for PLM-aided Query Expansion in Open-Domain Question Answering":
https://aclanthology.org/2025.acl-long.1076/

The paper motivates the general warning that uniformly adding terms can be harmful when their relevance differs. W16 tests a narrower HIRA-specific case: diagnosis state text is augmented with severity/confidence metadata that is useful for other typed primitives but orthogonal to diagnosis intent.

## 4. Meaning-preserving robustness should be separated from meaning-changing sensitivity

Lee, Pattern Recognition Letters 2026, "Language-guided invariance probing of vision-language models":
https://www.sciencedirect.com/science/article/pii/S0167865526000590

This motivates controlled invariance probes. W16 similarly keeps gold intent and candidate definitions fixed while changing only state rendering.

## 5. W16-specific scientific boundary

The prior art does not establish the HIRA causal mechanism.

The HIRA-specific hypothesis comes from exact internal evidence:
- W14 multiview semantic anchor was very strong on 4/4 fresh domains;
- W15 multiview diagnosis anchor collapsed on CE/CF;
- W15 diagnosis state appends severity/confidence metadata;
- the frozen HIRA anchor includes a state->definition MaxSim mean over all active state tokens.

Therefore W16 freezes a matched three-rendering experiment:
- R0 bare intent state;
- R1 identical state plus W15 reliability suffix;
- R2 identical encoded R1 but scoring only the R0 prefix positions.

Directional D2S and S2D are reported separately.

No external paper selects:
- W16 thresholds;
- fresh domains;
- seeds;
- renderings;
- prefix mask;
- directional weights;
- classifier precedence.

No post-exposure prior-art reading may be used to modify the scientific protocol.
