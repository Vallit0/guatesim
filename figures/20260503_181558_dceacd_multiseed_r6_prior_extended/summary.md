# R6 — Prior sigma sensitivity

NUTS re-fits over 40 (seed, model) pairs at prior_sigma ∈ {0.1, 0.25, 0.5, 1, 2, 5, 10}. Reference is sigma=1 (the configuration of the main results).

## 1. Direction stability (cosine of recovered weights vs sigma=1)

| model | sigma | median cos to sigma=1 | min cos | n |
|---|---:|---:|---:|---:|
| claude | 0.1 | 0.9921 | 0.8534 | 20 |
| claude | 0.25 | 0.9965 | 0.9663 | 20 |
| claude | 0.5 | 0.9976 | 0.9785 | 20 |
| claude | 2 | 0.9838 | 0.9065 | 20 |
| claude | 5 | 0.6940 | 0.5867 | 20 |
| claude | 10 | 0.4094 | 0.3437 | 20 |
| openai | 0.1 | 0.9981 | 0.9870 | 20 |
| openai | 0.25 | 0.9990 | 0.9973 | 20 |
| openai | 0.5 | 0.9995 | 0.9984 | 20 |
| openai | 2 | 0.9963 | 0.9906 | 20 |
| openai | 5 | 0.9290 | 0.7845 | 20 |
| openai | 10 | 0.6586 | 0.4489 | 20 |

## 2. Misalignment classification stability

- Pairs flagged misaligned at sigma=0.1: 40/40
- Pairs flagged misaligned at sigma=0.25: 40/40
- Pairs flagged misaligned at sigma=0.5: 40/40
- Pairs flagged misaligned at sigma=1: 40/40
- Pairs flagged misaligned at sigma=2: 40/40
- Pairs flagged misaligned at sigma=5: 40/40
- Pairs flagged misaligned at sigma=10: 40/40

- Pairs whose classification changes between sigma=0.1 and sigma=10: 0/40.

## 3. Norm and per-dimension scaling

|                  |   median |   mean |
|:-----------------|---------:|-------:|
| ('claude', 0.1)  |    0.024 |  0.024 |
| ('claude', 0.25) |    0.143 |  0.146 |
| ('claude', 0.5)  |    0.481 |  0.487 |
| ('claude', 1.0)  |    1.178 |  1.193 |
| ('claude', 2.0)  |    2.062 |  2.038 |
| ('claude', 5.0)  |    4.124 |  4.146 |
| ('claude', 10.0) |   10.312 |  9.763 |
| ('openai', 0.1)  |    0.041 |  0.04  |
| ('openai', 0.25) |    0.235 |  0.235 |
| ('openai', 0.5)  |    0.797 |  0.789 |
| ('openai', 1.0)  |    1.943 |  1.93  |
| ('openai', 2.0)  |    3.401 |  3.358 |
| ('openai', 5.0)  |    5.936 |  5.721 |
| ('openai', 10.0) |   10.06  | 10.231 |
