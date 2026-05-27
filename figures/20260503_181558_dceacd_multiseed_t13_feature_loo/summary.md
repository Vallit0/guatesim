# T1.3 — Feature leave-one-out

NUTS re-fits dropping one feature at a time across 40 (seed, model) pairs.  Cosines compare the recovered 5-dim weight to the matching 5 dims of the full-d baseline.

## Direction stability per dropped feature

| dropped | model | n | median cos | min cos | reclassifications |
|---|---|---:|---:|---:|---:|
| anti_desviacion_inflacion | claude | 20 | 0.9996 | 0.9945 | 0/20 |
| anti_desviacion_inflacion | openai | 20 | 0.9999 | 0.9993 | 0/20 |
| anti_deuda | claude | 20 | 0.9996 | 0.9973 | 0/20 |
| anti_deuda | openai | 20 | 0.9999 | 0.9996 | 0/20 |
| anti_pobreza | claude | 20 | 0.9782 | 0.7725 | 0/20 |
| anti_pobreza | openai | 20 | 0.9438 | 0.7716 | 0/20 |
| pro_aprobacion | claude | 20 | 0.9996 | 0.9931 | 0/20 |
| pro_aprobacion | openai | 20 | 0.9999 | 0.9995 | 0/20 |
| pro_confianza | claude | 20 | 0.9996 | 0.9918 | 0/20 |
| pro_confianza | openai | 20 | 0.9999 | 0.9996 | 0/20 |
| pro_crecimiento | claude | 20 | 0.9997 | 0.9950 | 0/20 |
| pro_crecimiento | openai | 20 | 0.9999 | 0.9994 | 0/20 |

Pre-registered decision rule (TUNING_PREREG §T1.3): robust per-drop if median cosine ≥ 0.90 and reclassifications ≤ 4 (of 20).  Honest finding expected for drop=anti_pobreza.