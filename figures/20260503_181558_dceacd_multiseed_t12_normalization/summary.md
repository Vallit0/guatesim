# T1.2 — Normalization sweep

Re-fits over 40 (seed, model) pairs under 4 deterministic variants and 10 multiplicative-scaling draws.

## Cosine of recovered weights vs identity baseline

| model | variant | n | median cos | min cos | p10 cos |
|---|---|---:|---:|---:|---:|
| claude | centered | 20 | 1.0000 | 1.0000 | 1.0000 |
| claude | minmax | 20 | 0.9323 | 0.7299 | 0.8617 |
| claude | multscale | 200 | 0.9970 | 0.9649 | 0.9878 |
| claude | zscore | 20 | 0.7532 | 0.6186 | 0.6565 |
| openai | centered | 20 | 1.0000 | 0.9999 | 1.0000 |
| openai | minmax | 20 | 0.9761 | 0.9531 | 0.9546 |
| openai | multscale | 200 | 0.9989 | 0.9799 | 0.9955 |
| openai | zscore | 20 | 0.9086 | 0.8080 | 0.8302 |

Pre-registered decision rule (TUNING_PREREG §T1.2): robust if median cosine ≥ 0.95 for zscore/minmax/centered, and for multscale the 10-draw 10th percentile ≥ 0.85.