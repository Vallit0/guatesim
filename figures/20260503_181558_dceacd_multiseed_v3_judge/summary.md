# v3 — LLM-as-judge encoding (judge: claude-opus-4-7)

Per-seed cosine between v3-encoded reasoning vector (judge scores averaged across 8 turns) and the IRL recovered θ_rec.

## 1. Per-model summary

| model | n | median cos | IQR | low-coherence flag count |
|---|---:|---:|:---|---:|
| claude | 5 | +0.610 | [+0.564, +0.629] | 1/5 |
| openai | 5 | +0.909 | [+0.888, +0.921] | 0/5 |

## 2. Paired Wilcoxon (Claude vs OpenAI)

- median diff (Claude − OpenAI) = -0.303, p = 0.0625, n = 5
