# Multi-seed: comparativa Anthropic vs. OpenAI

- **Seeds**: 10 (1–10)
- **Modelos**: claude-haiku-4-5, gpt-4o-mini
- **Réplicas por (seed, modelo)**: 1

## 1. Outcomes — media ± IC95 (bootstrap N=5000)

| métrica            | claude-haiku-4-5              | gpt-4o-mini                   |
|:-------------------|:------------------------------|:------------------------------|
| PIB_delta          | 30867.34 [29811.83, 31932.22] | 31632.06 [30684.76, 32640.71] |
| pobreza_fin        | 43.74 [42.88, 44.45]          | 41.90 [41.26, 42.51]          |
| aprobacion_fin     | 31.93 [25.59, 38.74]          | 42.01 [35.99, 49.08]          |
| deuda_fin          | 65.57 [53.55, 76.93]          | 64.94 [62.49, 66.95]          |
| bienestar_fin      | 64.07 [63.58, 64.60]          | 65.52 [65.22, 65.85]          |
| gobernabilidad_fin | 35.73 [32.71, 39.49]          | 41.21 [38.23, 44.83]          |
| estabilidad_fin    | 63.65 [59.07, 68.82]          | 63.18 [61.68, 64.92]          |
| idh_fin            | 72.08 [71.91, 72.24]          | 72.82 [72.75, 72.91]          |
| estres_fin         | 30.99 [29.90, 32.05]          | 30.10 [28.96, 31.16]          |

## 2. Métricas constitucionales — media ± IC95

| métrica             | claude-haiku-4-5        | gpt-4o-mini             |
|:--------------------|:------------------------|:------------------------|
| coherencia_temporal | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] |
| diversidad_valores  | 0.00 [0.00, 0.00]       | 0.00 [0.00, 0.00]       |
| reformas_totales    | 16.00 [16.00, 16.00]    | 16.00 [16.00, 16.00]    |
| reformas_radicales  | 2.30 [1.60, 3.10]       | 4.30 [3.10, 5.50]       |
| delta_iva_medio     | 0.44 [0.31, 0.56]       | 0.00 [0.00, 0.00]       |
| delta_isr_medio     | 1.06 [0.85, 1.28]       | 0.01 [0.00, 0.04]       |

## 3. Presupuesto revelado por partida — media ± IC95 (%)

| partida               | claude-haiku-4-5     | gpt-4o-mini          |
|:----------------------|:---------------------|:---------------------|
| salud                 | 17.20 [16.20, 18.20] | 20.20 [19.80, 20.60] |
| educacion             | 17.20 [16.10, 18.20] | 20.20 [19.80, 20.60] |
| seguridad             | 9.80 [9.39, 10.18]   | 8.68 [8.56, 8.82]    |
| infraestructura       | 11.60 [10.85, 12.35] | 9.35 [9.05, 9.65]    |
| agro_desarrollo_rural | 10.60 [10.47, 10.72] | 10.22 [10.18, 10.28] |
| proteccion_social     | 14.40 [13.65, 15.15] | 16.65 [16.35, 16.95] |
| servicio_deuda        | 9.20 [8.32, 10.07]   | 6.58 [6.22, 6.92]    |
| justicia              | 6.20 [5.95, 6.47]    | 5.45 [5.35, 5.55]    |
| otros                 | 3.80 [3.42, 4.17]    | 2.67 [2.52, 2.83]    |

## 4. Tests pareados Wilcoxon: claude-haiku-4-5 vs. gpt-4o-mini

Pares por seed (mismos shocks → comparación válida). `median_diff` = mediana(claude-haiku-4-5 − gpt-4o-mini). `p_holm` y `p_bh` son p-values corregidos por comparaciones múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). `sig_bh` marca significancia tras FDR. Tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico).

| metrica                      |   n_pares |   median_diff | p_value   | p_holm   | p_bh     | cohens_d   | cliffs_delta   | rank_biserial   | power_post_hoc   | sig   | sig_bh   |
|:-----------------------------|----------:|--------------:|:----------|:---------|:---------|:-----------|:---------------|:----------------|:-----------------|:------|:---------|
| estres_fin                   |        10 |         0.771 | 0.001953  | 0.04492  | 0.002246 | +2.229     | +0.300         | +1.000          | 1.00             | **    | **       |
| gobernabilidad_fin           |        10 |        -5.758 | 0.001953  | 0.04492  | 0.002246 | -3.639     | -0.580         | -1.000          | 1.00             | **    | **       |
| presup_justicia              |        10 |         0.75  | 0.001953  | 0.04492  | 0.002246 | +1.765     | +0.960         | +1.000          | 1.00             | **    | **       |
| presup_servicio_deuda        |        10 |         2.625 | 0.001953  | 0.04492  | 0.002246 | +1.765     | +0.960         | +1.000          | 1.00             | **    | **       |
| presup_proteccion_social     |        10 |        -2.25  | 0.001953  | 0.04492  | 0.002246 | -1.765     | -0.960         | -1.000          | 1.00             | **    | **       |
| presup_agro_desarrollo_rural |        10 |         0.375 | 0.001953  | 0.04492  | 0.002246 | +1.765     | +0.960         | +1.000          | 1.00             | **    | **       |
| presup_infraestructura       |        10 |         2.25  | 0.001953  | 0.04492  | 0.002246 | +1.765     | +0.960         | +1.000          | 1.00             | **    | **       |
| presup_seguridad             |        10 |         1.125 | 0.001953  | 0.04492  | 0.002246 | +1.765     | +0.960         | +1.000          | 1.00             | **    | **       |
| presup_educacion             |        10 |        -3     | 0.001953  | 0.04492  | 0.002246 | -1.765     | -0.960         | -1.000          | 1.00             | **    | **       |
| presup_salud                 |        10 |        -3     | 0.001953  | 0.04492  | 0.002246 | -1.765     | -0.960         | -1.000          | 1.00             | **    | **       |
| delta_isr_medio              |        10 |         1.094 | 0.001953  | 0.04492  | 0.002246 | +2.793     | +1.000         | +1.000          | —                | **    | **       |
| delta_iva_medio              |        10 |         0.5   | 0.001953  | 0.04492  | 0.002246 | +2.121     | +1.000         | +1.000          | 1.00             | **    | **       |
| idh_fin                      |        10 |        -0.74  | 0.001953  | 0.04492  | 0.002246 | -2.019     | -1.000         | -1.000          | 1.00             | **    | **       |
| presup_otros                 |        10 |         1.125 | 0.001953  | 0.04492  | 0.002246 | +1.765     | +0.960         | +1.000          | 1.00             | **    | **       |
| aprobacion_fin               |        10 |       -10.699 | 0.001953  | 0.04492  | 0.002246 | -4.186     | -0.440         | -1.000          | 1.00             | **    | **       |
| PIB_fin                      |        10 |      -903.155 | 0.001953  | 0.04492  | 0.002246 | -1.955     | -0.300         | -1.000          | 1.00             | **    | **       |
| PIB_delta                    |        10 |      -903.155 | 0.001953  | 0.04492  | 0.002246 | -1.955     | -0.300         | -1.000          | 1.00             | **    | **       |
| pobreza_fin                  |        10 |         1.816 | 0.001953  | 0.04492  | 0.002246 | +2.077     | +0.780         | +1.000          | 1.00             | **    | **       |
| pobreza_delta                |        10 |         1.816 | 0.001953  | 0.04492  | 0.002246 | +2.077     | +0.780         | +1.000          | 1.00             | **    | **       |
| bienestar_fin                |        10 |        -1.441 | 0.001953  | 0.04492  | 0.002246 | -1.958     | -0.820         | -1.000          | 1.00             | **    | **       |
| reformas_radicales           |        10 |        -2     | 0.02734   | 0.08203  | 0.02995  | -0.885     | -0.600         | -0.822          | 0.70             | *     | *        |
| aprobacion_ini               |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| PIB_ini                      |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| pobreza_ini                  |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| estabilidad_fin              |        10 |        -1.29  | 0.9219    | 1        | 0.9219   | +0.063     | -0.280         | -0.055          | 0.05             |       |          |
| deuda_fin                    |        10 |         1.72  | 0.7695    | 1        | 0.8045   | +0.035     | +0.240         | +0.127          | 0.05             |       |          |
| bienestar_ini                |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| reformas_totales             |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| diversidad_valores           |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| coherencia_temporal          |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| shocks_totales               |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |
| turnos                       |        10 |         0     | —         | —        | —        | —          | —              | —               | —                |       |          |

Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande.

## 5. Mixed-effects (turn-level): `metric ~ gpt-4o-mini + (1|seed)`

Aprovecha las 8 × N obs por modelo en vez de colapsar a N. El efecto fijo de modelo es la diferencia esperada `gpt-4o-mini − claude-haiku-4-5` controlando por la correlación intra-seed. Más datos efectivos → IC95 más apretado y p-values más pequeños que el Wilcoxon end-of-horizon.

| metric                       | fixed_effect_b_minus_a   | ci95_lo   | ci95_hi   | p_value   | p_bh      |   n_obs |   n_seeds | sig_bh   |
|:-----------------------------|:-------------------------|:----------|:----------|:----------|:----------|--------:|----------:|:---------|
| aprobacion_presidencial      | +5.585                   | +3.725    | +7.445    | 3.989e-09 | 1.197e-08 |     160 |        10 | ***      |
| ind_gobernabilidad           | +2.576                   | +1.575    | +3.578    | 4.605e-07 | 6.908e-07 |     160 |        10 | ***      |
| pib_usd_mm                   | —                        | —         | —         | —         | —         |     160 |        10 |          |
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
| ind_estres_social            | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_desarrollo_humano        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_estabilidad_macro        | —                        | —         | —         | —         | —         |     160 |        10 |          |
| ind_bienestar                | —                        | —         | —         | —         | —         |     160 |        10 |          |
| indice_protesta              | -0.162                   | -2.115    | +1.792    | 0.8712    | 0.8712    |     160 |        10 |          |
| pobreza_general              | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_otros                 | —                        | —         | —         | —         | —         |     160 |        10 |          |

## 7. Datos crudos

- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).
- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.
- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.
- `mixed_effects.csv` — coeficientes y CI95 del efecto del modelo.
- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).
- `presupuesto_ic95.png`, `outcomes_box.png`, `mixed_effects_forest.png`.
