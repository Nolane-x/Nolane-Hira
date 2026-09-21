# R8 research corrections from the frozen Laya benchmark source

These corrections override earlier planning where it conflicts.

## C1 — MASSIVE/XNLI cannot be training data for the direct Laya generalization lane

Laya's frozen T4 artifact states that neither base checkpoint trained on typed-decisions, MASSIVE or XNLI. It also states that SST-5, DAIR Emotion, prompt-injections and Banking77 were held out, while AG News and BoolQ were in the training mix.

Therefore:

- the **direct Laya** MASSIVE/XNLI lane must keep those corpora out of HIRA weight training, task-specific distillation, threshold selection, tokenizer selection and architecture selection;
- EN/VI capability for that lane must come from other multilingual corpora, synthetic/teacher data or general pretraining;
- a separate **adapted** lane may train on MASSIVE/XNLI, but its result cannot replace the direct cell.

This corrects the R6 idea of using paired MASSIVE/XNLI directly as the main EN/VI training stage. They remain excellent evaluation sets, but direct parity requires them to be frozen tests.

## C2 — Banking77 mechanism probes are not direct Laya/Jev evidence

R3–R5 sparse Banking77 experiments used labeled training examples/prototypes. They remain useful architecture probes, but they cannot be counted as wins against Laya's held-out Banking77 number or the Jev zero-shot BTZSC pilot.

The Jev 92.40% full-test experiment is a different adapted protocol: it supplies 24 retrieved labeled training examples per prediction. HIRA can compete with that number only under the same information-budget rule.

## C3 — typed-decisions has two valid lanes

- **zero-shot/base lane:** compare without typed-decisions task training to Laya base checkpoints.
- **fine-tuned lane:** task training is allowed because Laya's 0.766 checkpoint was fine-tuned on the benchmark training split.

Never mix the two.

## C4 — benchmark optimization requires a private/fresh confirmation

Even correctly matched public protocols can be benchmaxxed by repeated model selection. A final parity claim still requires a frozen confirmatory suite unused during R8 optimization.
