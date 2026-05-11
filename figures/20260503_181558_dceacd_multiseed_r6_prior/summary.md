# R6 — Prior sigma sensitivity

NUTS re-fits over 40 (seed, model) pairs at prior_sigma ∈ {0.5, 1, 2}. Reference is sigma=1 (the configuration of the main results).

## 1. Direction stability (cosine of recovered weights vs sigma=1)

| model | sigma | median cos to sigma=1 | min cos | n |
|---|---:|---:|---:|---:|
| claude | 0.5 | 0.9976 | 0.9785 | 20 |
| claude | 2 | 0.9838 | 0.9065 | 20 |
| openai | 0.5 | 0.9995 | 0.9984 | 20 |
| openai | 2 | 0.9963 | 0.9906 | 20 |

## 2. Misalignment classification stability

- Pairs flagged misaligned at sigma=0.5: 40/40
- Pairs flagged misaligned at sigma=1.0: 40/40
- Pairs flagged misaligned at sigma=2.0: 40/40

- Pairs whose classification changes between sigma=0.5 and sigma=2.0: 0/40.

## 3. Norm and per-dimension scaling

|                 |   median |   mean |
|:----------------|---------:|-------:|
| ('claude', 0.5) |    0.481 |  0.487 |
| ('claude', 1.0) |    1.178 |  1.193 |
| ('claude', 2.0) |    2.062 |  2.038 |
| ('openai', 0.5) |    0.797 |  0.789 |
| ('openai', 1.0) |    1.943 |  1.93  |
| ('openai', 2.0) |    3.401 |  3.358 |
