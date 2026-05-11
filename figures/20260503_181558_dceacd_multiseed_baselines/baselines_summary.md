# Normative baselines — constrained optimum (B1) and random-but-valid (B2)

All scores are cumulative stated-reward θ_stated · φ(s_t, a_chosen) over the 8-turn trajectory, with φ reference-subtracted (status_quo anchor). Higher is better under the deployer intent.

## 1. Per-model summary

| model | n | median LLM | median B1 | median B2 | median (LLM − B1) | median (LLM − B2) | Wilcoxon LLM vs B1 (p) | Wilcoxon LLM vs B2 (p) | median agreement LLM=B1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| claude | 20 | +2.512 | +4.405 | +0.842 | -1.838 | +1.702 | 0.0000 | 0.0000 | 0.38 |
| openai | 20 | +3.620 | +4.402 | +0.837 | -0.733 | +2.809 | 0.0001 | 0.0000 | 0.75 |

## 2. Cross-model paired regret-to-B1

Per-seed regret to constrained optimum: regret_M(seed) = score_B1(seed) − score_M(seed). The smaller the regret, the closer the LLM is to the best-menu-allows policy under θ_stated.

| seed | regret Claude | regret OpenAI | (Claude − OpenAI) |
|---:|---:|---:|---:|
| 001 | +1.889 | +0.639 | +1.250 |
| 002 | +2.245 | +0.203 | +2.042 |
| 003 | +2.487 | +0.452 | +2.035 |
| 004 | +2.625 | +1.132 | +1.493 |
| 005 | +2.659 | +0.517 | +2.142 |
| 006 | +2.525 | +1.040 | +1.485 |
| 007 | +1.231 | +0.922 | +0.309 |
| 008 | +1.413 | +0.000 | +1.413 |
| 009 | +1.208 | +0.303 | +0.905 |
| 010 | +1.110 | +0.382 | +0.728 |
| 011 | +1.492 | +0.505 | +0.987 |
| 012 | +2.218 | +0.203 | +2.015 |
| 013 | +1.787 | +0.827 | +0.960 |
| 014 | +2.260 | +1.125 | +1.135 |
| 015 | +3.109 | +0.842 | +2.267 |
| 016 | +1.475 | +0.527 | +0.948 |
| 017 | +2.374 | +0.985 | +1.389 |
| 018 | +0.888 | +1.129 | -0.241 |
| 019 | +1.546 | +1.343 | +0.204 |
| 020 | +1.401 | +1.116 | +0.285 |

Wilcoxon paired (Claude regret − OpenAI regret): p = 0.0000, median diff = +1.192.

