# Multi-seed: comparativa Anthropic vs. OpenAI

- **Seeds**: 10 (1–10)
- **Modelos**: claude-haiku-4-5, gpt-4o-mini
- **Réplicas por (seed, modelo)**: 1

## 1. Outcomes — media ± IC95 (bootstrap N=5000)

| métrica            | claude-haiku-4-5              | gpt-4o-mini                   |
|:-------------------|:------------------------------|:------------------------------|
| PIB_delta          | 30867.20 [29935.68, 31815.16] | 31580.23 [30653.73, 32574.78] |
| pobreza_fin        | 43.63 [42.72, 44.44]          | 41.48 [40.82, 42.15]          |
| aprobacion_fin     | 31.40 [24.60, 39.54]          | 42.94 [36.65, 49.95]          |
| deuda_fin          | 68.20 [61.00, 75.37]          | 63.29 [60.12, 66.43]          |
| bienestar_fin      | 64.16 [63.64, 64.74]          | 65.88 [65.51, 66.21]          |
| gobernabilidad_fin | 35.75 [32.31, 40.09]          | 42.03 [38.98, 45.80]          |
| estabilidad_fin    | 61.63 [59.28, 64.13]          | 64.28 [62.86, 65.69]          |
| idh_fin            | 72.12 [71.92, 72.33]          | 73.00 [72.87, 73.12]          |
| estres_fin         | 31.15 [30.15, 32.09]          | 29.66 [28.63, 30.52]          |

## 2. Métricas constitucionales — media ± IC95

| métrica             | claude-haiku-4-5      | gpt-4o-mini             |
|:--------------------|:----------------------|:------------------------|
| coherencia_temporal | 94.29 [85.71, 100.00] | 100.00 [100.00, 100.00] |
| diversidad_valores  | 0.11 [0.00, 0.27]     | 0.00 [0.00, 0.00]       |
| reformas_totales    | 16.00 [16.00, 16.00]  | 16.00 [16.00, 16.00]    |
| reformas_radicales  | 2.20 [1.40, 2.90]     | 2.90 [2.10, 3.70]       |
| delta_iva_medio     | 0.40 [0.20, 0.57]     | 0.00 [0.00, 0.00]       |
| delta_isr_medio     | 1.06 [0.81, 1.26]     | 0.00 [0.00, 0.00]       |

## 3. Presupuesto revelado por partida — media ± IC95 (%)

| partida               | claude-haiku-4-5     | gpt-4o-mini          |
|:----------------------|:---------------------|:---------------------|
| salud                 | 17.40 [16.50, 18.30] | 21.00 [20.60, 21.40] |
| educacion             | 17.40 [16.50, 18.30] | 21.00 [20.60, 21.40] |
| seguridad             | 9.72 [9.39, 10.06]   | 8.38 [8.22, 8.53]    |
| infraestructura       | 11.45 [10.78, 12.05] | 8.75 [8.45, 9.05]    |
| agro_desarrollo_rural | 10.57 [10.46, 10.69] | 10.12 [10.07, 10.18] |
| proteccion_social     | 14.55 [13.88, 15.22] | 17.25 [16.95, 17.55] |
| servicio_deuda        | 9.03 [8.24, 9.81]    | 5.88 [5.53, 6.22]    |
| justicia              | 6.15 [5.92, 6.38]    | 5.25 [5.15, 5.35]    |
| otros                 | 3.73 [3.39, 4.03]    | 2.38 [2.23, 2.52]    |

## 4. Tests pareados Wilcoxon: claude-haiku-4-5 vs. gpt-4o-mini

Pares por seed (mismos shocks → comparación válida). `median_diff` = mediana(claude-haiku-4-5 − gpt-4o-mini). `p_holm` y `p_bh` son p-values corregidos por comparaciones múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). `sig_bh` marca significancia tras FDR. Tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico).

| metrica                      |   n_pares |   median_diff | p_value   | p_holm   | p_bh     | cohens_d   | cliffs_delta   | rank_biserial   | power_post_hoc   | sig   | sig_bh   |
|:-----------------------------|----------:|--------------:|:----------|:---------|:---------|:-----------|:---------------|:----------------|:-----------------|:------|:---------|
| estres_fin                   |        10 |         1.391 | 0.001953  | 0.05078  | 0.002673 | +1.565     | +0.500         | +1.000          | 0.99             | **    | **       |
| gobernabilidad_fin           |        10 |        -6.09  | 0.001953  | 0.05078  | 0.002673 | -3.423     | -0.620         | -1.000          | 1.00             | **    | **       |
| presup_justicia              |        10 |         0.75  | 0.001953  | 0.05078  | 0.002673 | +1.897     | +0.980         | +1.000          | 1.00             | **    | **       |
| presup_servicio_deuda        |        10 |         2.625 | 0.001953  | 0.05078  | 0.002673 | +1.897     | +0.980         | +1.000          | 1.00             | **    | **       |
| presup_proteccion_social     |        10 |        -2.25  | 0.001953  | 0.05078  | 0.002673 | -1.897     | -0.980         | -1.000          | 1.00             | **    | **       |
| presup_agro_desarrollo_rural |        10 |         0.375 | 0.001953  | 0.05078  | 0.002673 | +1.897     | +0.980         | +1.000          | 1.00             | **    | **       |
| presup_infraestructura       |        10 |         2.25  | 0.001953  | 0.05078  | 0.002673 | +1.897     | +0.980         | +1.000          | 1.00             | **    | **       |
| presup_seguridad             |        10 |         1.125 | 0.001953  | 0.05078  | 0.002673 | +1.897     | +0.980         | +1.000          | 1.00             | **    | **       |
| presup_educacion             |        10 |        -3     | 0.001953  | 0.05078  | 0.002673 | -1.897     | -0.980         | -1.000          | 1.00             | **    | **       |
| presup_salud                 |        10 |        -3     | 0.001953  | 0.05078  | 0.002673 | -1.897     | -0.980         | -1.000          | 1.00             | **    | **       |
| delta_isr_medio              |        10 |         1.25  | 0.001953  | 0.05078  | 0.002673 | +2.633     | +1.000         | +1.000          | 1.00             | **    | **       |
| idh_fin                      |        10 |        -0.773 | 0.001953  | 0.05078  | 0.002673 | -2.110     | -0.980         | -1.000          | 1.00             | **    | **       |
| presup_otros                 |        10 |         1.125 | 0.001953  | 0.05078  | 0.002673 | +1.897     | +0.980         | +1.000          | 1.00             | **    | **       |
| bienestar_fin                |        10 |        -1.492 | 0.001953  | 0.05078  | 0.002673 | -2.059     | -0.860         | -1.000          | 1.00             | **    | **       |
| aprobacion_fin               |        10 |       -11.88  | 0.001953  | 0.05078  | 0.002673 | -4.056     | -0.520         | -1.000          | 1.00             | **    | **       |
| PIB_fin                      |        10 |      -777.562 | 0.001953  | 0.05078  | 0.002673 | -1.918     | -0.320         | -1.000          | 1.00             | **    | **       |
| PIB_delta                    |        10 |      -777.562 | 0.001953  | 0.05078  | 0.002673 | -1.918     | -0.320         | -1.000          | 1.00             | **    | **       |
| pobreza_fin                  |        10 |         1.915 | 0.001953  | 0.05078  | 0.002673 | +2.159     | +0.740         | +1.000          | 1.00             | **    | **       |
| pobreza_delta                |        10 |         1.915 | 0.001953  | 0.05078  | 0.002673 | +2.159     | +0.740         | +1.000          | 1.00             | **    | **       |
| delta_iva_medio              |        10 |         0.469 | 0.01367   | 0.0957   | 0.01777  | +1.307     | +0.800         | +0.855          | 0.96             | *     | *        |
| PIB_ini                      |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| pobreza_ini                  |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| aprobacion_ini               |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| deuda_fin                    |        10 |         4.655 | 0.2324    | 0.9297   | 0.2627   | +0.471     | +0.240         | +0.455          | 0.27             |       |          |
| estabilidad_fin              |        10 |        -3.017 | 0.08398   | 0.5039   | 0.104    | -0.740     | -0.420         | -0.636          | 0.55             |       |          |
| reformas_radicales           |        10 |         0     | 0.4375    | 1        | 0.474    | -0.350     | -0.300         | -0.429          | 0.17             |       |          |
| reformas_totales             |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| diversidad_valores           |        10 |         0     | 0.5       | 1        | 0.5      | +0.474     | +0.200         | +1.000          | 0.27             |       |          |
| coherencia_temporal          |        10 |         0     | 0.5       | 1        | 0.5      | -0.474     | -0.200         | -1.000          | 0.27             |       |          |
| shocks_totales               |        10 |         0     | 0.125     | 0.625    | 0.1477   | +0.775     | +0.130         | +1.000          | 0.59             |       |          |
| bienestar_ini                |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| turnos                       |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |

Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande.

## 5. Mixed-effects (turn-level): `metric ~ gpt-4o-mini + (1|seed)`

Aprovecha las 8 × N obs por modelo en vez de colapsar a N. El efecto fijo de modelo es la diferencia esperada `gpt-4o-mini − claude-haiku-4-5` controlando por la correlación intra-seed. Más datos efectivos → IC95 más apretado y p-values más pequeños que el Wilcoxon end-of-horizon.

| metric                       | fixed_effect_b_minus_a   | ci95_lo   | ci95_hi   | p_value   | p_bh      |   n_obs |   n_seeds | sig_bh   |
|:-----------------------------|:-------------------------|:----------|:----------|:----------|:----------|--------:|----------:|:---------|
| delta_iva_pp                 | -0.398                   | -0.524    | -0.272    | 5.947e-10 | 9.911e-10 |     160 |        10 | ***      |
| aprobacion_presidencial      | +6.483                   | +4.539    | +8.427    | 6.288e-11 | 1.572e-10 |     160 |        10 | ***      |
| ind_gobernabilidad           | +3.616                   | +2.576    | +4.657    | 9.667e-12 | 4.834e-11 |     160 |        10 | ***      |
| presup_salud                 | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_justicia              | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_servicio_deuda        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_proteccion_social     | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_agro_desarrollo_rural | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_infraestructura       | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_seguridad             | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_educacion             | —                        | —         | —         | —         | —         |     160 |        10 |          |
| pib_usd_mm                   | —                        | —         | —         | —         | —         |     160 |        10 |          |
| delta_isr_pp                 | —                        | —         | —         | —         | —         |     160 |        10 |          |
| deuda_pib                    | -0.859                   | -4.814    | +3.096    | 0.6703    | 0.6703    |     160 |        10 |          |
| ind_estres_social            | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_desarrollo_humano        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_estabilidad_macro        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_bienestar                | —                        | —         | —         | —         | —         |     160 |        10 |          |
| indice_protesta              | -1.339                   | -3.290    | +0.613    | 0.1787    | 0.2234    |     160 |        10 |          |
| pobreza_general              | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_otros                 | —                        | —         | —         | —         | —         |     160 |        10 |          |

## 7. Datos crudos

- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).
- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.
- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.
- `mixed_effects.csv` — coeficientes y CI95 del efecto del modelo.
- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).
- `presupuesto_ic95.png`, `outcomes_box.png`, `mixed_effects_forest.png`.
