# Multi-seed: comparativa Anthropic vs. OpenAI

- **Seeds**: 10 (1–10)
- **Modelos**: claude-haiku-4-5, gpt-4o-mini
- **Réplicas por (seed, modelo)**: 1

## 1. Outcomes — media ± IC95 (bootstrap N=5000)

| métrica            | claude-haiku-4-5              | gpt-4o-mini                   |
|:-------------------|:------------------------------|:------------------------------|
| PIB_delta          | 31035.00 [30070.14, 32064.95] | 31580.19 [30658.11, 32575.34] |
| pobreza_fin        | 43.59 [42.77, 44.39]          | 41.48 [40.70, 42.22]          |
| aprobacion_fin     | 31.03 [24.76, 37.87]          | 42.75 [36.76, 49.78]          |
| deuda_fin          | 66.66 [60.66, 72.58]          | 63.34 [60.65, 66.42]          |
| bienestar_fin      | 64.18 [63.64, 64.70]          | 65.88 [65.43, 66.30]          |
| gobernabilidad_fin | 35.71 [32.61, 39.33]          | 42.00 [39.10, 45.72]          |
| estabilidad_fin    | 61.99 [60.10, 64.07]          | 64.28 [62.84, 65.68]          |
| idh_fin            | 72.14 [71.91, 72.36]          | 73.00 [72.86, 73.15]          |
| estres_fin         | 30.48 [29.18, 31.75]          | 29.74 [28.55, 30.76]          |

## 2. Métricas constitucionales — media ± IC95

| métrica             | claude-haiku-4-5        | gpt-4o-mini             |
|:--------------------|:------------------------|:------------------------|
| coherencia_temporal | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| diversidad_valores  | 0.00 [0.00, 0.00]       | 0.00 [0.00, 0.00]       |
| reformas_totales    | 15.90 [15.70, 16.00]    | 16.00 [16.00, 16.00]    |
| reformas_radicales  | 3.20 [2.50, 3.80]       | 2.10 [1.40, 2.80]       |
| delta_iva_medio     | 0.22 [-0.11, 0.44]      | 0.00 [0.00, 0.00]       |
| delta_isr_medio     | 1.14 [0.93, 1.37]       | 0.00 [0.00, 0.00]       |

## 3. Presupuesto revelado por partida — media ± IC95 (%)

| partida               | claude-haiku-4-5     | gpt-4o-mini          |
|:----------------------|:---------------------|:---------------------|
| salud                 | 17.40 [16.30, 18.50] | 21.00 [20.50, 21.50] |
| educacion             | 17.40 [16.30, 18.50] | 21.00 [20.50, 21.50] |
| seguridad             | 9.72 [9.31, 10.10]   | 8.38 [8.19, 8.56]    |
| infraestructura       | 11.45 [10.62, 12.28] | 8.75 [8.45, 9.12]    |
| agro_desarrollo_rural | 10.57 [10.44, 10.71] | 10.12 [10.06, 10.19] |
| proteccion_social     | 14.55 [13.72, 15.38] | 17.25 [16.88, 17.62] |
| servicio_deuda        | 9.03 [8.06, 9.99]    | 5.88 [5.44, 6.31]    |
| justicia              | 6.15 [5.88, 6.42]    | 5.25 [5.12, 5.38]    |
| otros                 | 3.73 [3.31, 4.14]    | 2.38 [2.19, 2.56]    |

## 4. Tests pareados Wilcoxon: claude-haiku-4-5 vs. gpt-4o-mini

Pares por seed (mismos shocks → comparación válida). `median_diff` = mediana(claude-haiku-4-5 − gpt-4o-mini). `p_holm` y `p_bh` son p-values corregidos por comparaciones múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). `sig_bh` marca significancia tras FDR. Tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico).

| metrica                      |   n_pares |   median_diff | p_value   | p_holm   | p_bh     | cohens_d   | cliffs_delta   | rank_biserial   | power_post_hoc   | sig   | sig_bh   |
|:-----------------------------|----------:|--------------:|:----------|:---------|:---------|:-----------|:---------------|:----------------|:-----------------|:------|:---------|
| presup_otros                 |        10 |         1.312 | 0.001953  | 0.04883  | 0.002713 | +1.841     | +0.970         | +1.000          | 1.00             | **    | **       |
| delta_isr_medio              |        10 |         1.188 | 0.001953  | 0.04883  | 0.002713 | +3.007     | +1.000         | +1.000          | 1.00             | **    | **       |
| presup_educacion             |        10 |        -3.5   | 0.001953  | 0.04883  | 0.002713 | -1.841     | -0.970         | -1.000          | 1.00             | **    | **       |
| idh_fin                      |        10 |        -0.848 | 0.001953  | 0.04883  | 0.002713 | -2.029     | -1.000         | -1.000          | 1.00             | **    | **       |
| presup_seguridad             |        10 |         1.312 | 0.001953  | 0.04883  | 0.002713 | +1.841     | +0.970         | +1.000          | 1.00             | **    | **       |
| gobernabilidad_fin           |        10 |        -6.35  | 0.001953  | 0.04883  | 0.002713 | -7.049     | -0.600         | -1.000          | 1.00             | **    | **       |
| bienestar_fin                |        10 |        -1.66  | 0.001953  | 0.04883  | 0.002713 | -1.984     | -0.880         | -1.000          | 1.00             | **    | **       |
| presup_infraestructura       |        10 |         2.625 | 0.001953  | 0.04883  | 0.002713 | +1.841     | +0.970         | +1.000          | 1.00             | **    | **       |
| presup_agro_desarrollo_rural |        10 |         0.438 | 0.001953  | 0.04883  | 0.002713 | +1.841     | +0.970         | +1.000          | 1.00             | **    | **       |
| aprobacion_fin               |        10 |       -11.834 | 0.001953  | 0.04883  | 0.002713 | -8.161     | -0.520         | -1.000          | 1.00             | **    | **       |
| presup_proteccion_social     |        10 |        -2.625 | 0.001953  | 0.04883  | 0.002713 | -1.841     | -0.970         | -1.000          | 1.00             | **    | **       |
| pobreza_delta                |        10 |         2.076 | 0.001953  | 0.04883  | 0.002713 | +2.073     | +0.740         | +1.000          | 1.00             | **    | **       |
| pobreza_fin                  |        10 |         2.076 | 0.001953  | 0.04883  | 0.002713 | +2.073     | +0.740         | +1.000          | 1.00             | **    | **       |
| presup_servicio_deuda        |        10 |         3.062 | 0.001953  | 0.04883  | 0.002713 | +1.841     | +0.970         | +1.000          | 1.00             | **    | **       |
| PIB_delta                    |        10 |      -609.919 | 0.001953  | 0.04883  | 0.002713 | -1.621     | -0.240         | -1.000          | 0.99             | **    | **       |
| PIB_fin                      |        10 |      -609.919 | 0.001953  | 0.04883  | 0.002713 | -1.621     | -0.240         | -1.000          | 0.99             | **    | **       |
| presup_justicia              |        10 |         0.875 | 0.001953  | 0.04883  | 0.002713 | +1.841     | +0.970         | +1.000          | 1.00             | **    | **       |
| presup_salud                 |        10 |        -3.5   | 0.001953  | 0.04883  | 0.002713 | -1.841     | -0.970         | -1.000          | 1.00             | **    | **       |
| estres_fin                   |        10 |         0.817 | 0.01367   | 0.0957   | 0.01799  | +1.099     | +0.160         | +0.855          | 0.87             | *     | *        |
| estabilidad_fin              |        10 |        -2.631 | 0.03711   | 0.2227   | 0.04639  | -0.841     | -0.520         | -0.745          | 0.66             | *     | *        |
| reformas_totales             |        10 |         0     | 1         | 1        | 1        | -0.316     | -0.100         | -1.000          | 0.15             |       |          |
| reformas_radicales           |        10 |         1.5   | 0.07031   | 0.3516   | 0.08371  | +0.759     | +0.490         | +0.722          | 0.57             |       |          |
| delta_iva_medio              |        10 |         0.375 | 0.1211    | 0.4844   | 0.1376   | +0.437     | +0.700         | +0.600          | 0.24             |       |          |
| diversidad_valores           |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| coherencia_temporal          |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| shocks_totales               |        10 |         0     | 1         | 1        | 1        | +0.316     | +0.010         | +1.000          | 0.15             |       |          |
| bienestar_ini                |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| deuda_fin                    |        10 |         3.652 | 0.1602    | 0.4844   | 0.1741   | +0.468     | +0.320         | +0.527          | 0.26             |       |          |
| aprobacion_ini               |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| pobreza_ini                  |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| PIB_ini                      |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| turnos                       |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |

Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande.

## 5. Mixed-effects (turn-level): `metric ~ gpt-4o-mini + (1|seed)`

Aprovecha las 8 × N obs por modelo en vez de colapsar a N. El efecto fijo de modelo es la diferencia esperada `gpt-4o-mini − claude-haiku-4-5` controlando por la correlación intra-seed. Más datos efectivos → IC95 más apretado y p-values más pequeños que el Wilcoxon end-of-horizon.

| metric                       | fixed_effect_b_minus_a   | ci95_lo   | ci95_hi   | p_value   | p_bh      |   n_obs |   n_seeds | sig_bh   |
|:-----------------------------|:-------------------------|:----------|:----------|:----------|:----------|--------:|----------:|:---------|
| aprobacion_presidencial      | +6.358                   | +4.440    | +8.275    | 8.085e-11 | 2.426e-10 |     160 |        10 | ***      |
| ind_gobernabilidad           | +3.533                   | +2.531    | +4.536    | 4.911e-12 | 2.947e-11 |     160 |        10 | ***      |
| ind_bienestar                | +1.172                   | +0.429    | +1.915    | 0.001983  | 0.003967  |     160 |        10 | **       |
| pib_usd_mm                   | +267.636                 | -2591.602 | +3126.873 | 0.8544    | 0.8544    |     160 |        10 |          |
| presup_salud                 | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_justicia              | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_servicio_deuda        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_proteccion_social     | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_agro_desarrollo_rural | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_infraestructura       | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_seguridad             | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_educacion             | —                        | —         | —         | —         | —         |     160 |        10 |          |
| delta_iva_pp                 | —                        | —         | —         | —         | —         |     160 |        10 |          |
| delta_isr_pp                 | —                        | —         | —         | —         | —         |     160 |        10 |          |
| deuda_pib                    | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_estres_social            | -0.624                   | -1.395    | +0.146    | 0.1124    | 0.1685    |     160 |        10 |          |
| ind_desarrollo_humano        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_estabilidad_macro        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| indice_protesta              | +0.227                   | -1.763    | +2.217    | 0.8228    | 0.8544    |     160 |        10 |          |
| pobreza_general              | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_otros                 | —                        | —         | —         | —         | —         |     160 |        10 |          |

## 7. Datos crudos

- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).
- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.
- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.
- `mixed_effects.csv` — coeficientes y CI95 del efecto del modelo.
- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).
- `presupuesto_ic95.png`, `outcomes_box.png`, `mixed_effects_forest.png`.
