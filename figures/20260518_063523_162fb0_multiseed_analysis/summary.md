# Multi-seed: comparativa Anthropic vs. OpenAI

- **Seeds**: 10 (1–10)
- **Modelos**: claude-haiku-4-5, gpt-4o-mini
- **Réplicas por (seed, modelo)**: 1

## 1. Outcomes — media ± IC95 (bootstrap N=5000)

| métrica            | claude-haiku-4-5              | gpt-4o-mini                   |
|:-------------------|:------------------------------|:------------------------------|
| PIB_delta          | 31194.77 [30248.45, 32174.79] | 31516.17 [30557.30, 32563.79] |
| pobreza_fin        | 44.71 [44.03, 45.38]          | 44.27 [43.53, 45.04]          |
| aprobacion_fin     | 31.97 [25.85, 39.12]          | 38.26 [31.47, 46.03]          |
| deuda_fin          | 64.87 [58.84, 71.32]          | 59.11 [55.88, 62.47]          |
| bienestar_fin      | 63.25 [62.92, 63.59]          | 63.58 [63.13, 64.01]          |
| gobernabilidad_fin | 36.94 [33.78, 40.65]          | 38.85 [35.13, 43.16]          |
| estabilidad_fin    | 62.49 [60.28, 64.90]          | 66.49 [65.02, 67.92]          |
| idh_fin            | 71.67 [71.56, 71.78]          | 71.84 [71.68, 72.00]          |
| estres_fin         | 31.32 [30.16, 32.46]          | 32.26 [30.89, 33.56]          |

## 2. Métricas constitucionales — media ± IC95

| métrica             | claude-haiku-4-5     | gpt-4o-mini           |
|:--------------------|:---------------------|:----------------------|
| coherencia_temporal | 84.29 [74.29, 94.29] | 98.57 [95.71, 100.00] |
| diversidad_valores  | 0.36 [0.11, 0.64]    | 0.05 [0.00, 0.16]     |
| reformas_totales    | 15.90 [15.70, 16.00] | 15.70 [15.20, 16.00]  |
| reformas_radicales  | 0.90 [0.50, 1.30]    | 2.00 [1.30, 2.70]     |
| delta_iva_medio     | 0.30 [0.15, 0.44]    | 0.20 [0.06, 0.35]     |
| delta_isr_medio     | 0.82 [0.66, 0.97]    | 0.04 [-0.03, 0.14]    |

## 3. Presupuesto revelado por partida — media ± IC95 (%)

| partida               | claude-haiku-4-5     | gpt-4o-mini          |
|:----------------------|:---------------------|:---------------------|
| salud                 | 15.30 [14.90, 15.70] | 15.81 [15.25, 16.36] |
| educacion             | 15.30 [14.90, 15.70] | 15.81 [15.24, 16.35] |
| seguridad             | 10.51 [10.36, 10.66] | 10.38 [10.13, 10.65] |
| infraestructura       | 13.03 [12.72, 13.32] | 11.96 [11.46, 12.38] |
| agro_desarrollo_rural | 10.84 [10.79, 10.89] | 10.43 [10.19, 10.64] |
| proteccion_social     | 12.97 [12.68, 13.28] | 13.71 [13.32, 14.16] |
| servicio_deuda        | 10.86 [10.51, 11.21] | 10.76 [10.11, 11.46] |
| justicia              | 6.67 [6.58, 6.78]    | 6.60 [6.44, 6.79]    |
| otros                 | 4.51 [4.36, 4.66]    | 4.53 [4.22, 4.85]    |

## 4. Tests pareados Wilcoxon: claude-haiku-4-5 vs. gpt-4o-mini

Pares por seed (mismos shocks → comparación válida). `median_diff` = mediana(claude-haiku-4-5 − gpt-4o-mini). `p_holm` y `p_bh` son p-values corregidos por comparaciones múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). `sig_bh` marca significancia tras FDR. Tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico).

| metrica                      |   n_pares |   median_diff | p_value   | p_holm   | p_bh    | cohens_d   | cliffs_delta   | rank_biserial   | power_post_hoc   | sig   | sig_bh   |
|:-----------------------------|----------:|--------------:|:----------|:---------|:--------|:-----------|:---------------|:----------------|:-----------------|:------|:---------|
| estres_fin                   |        10 |        -1.08  | 0.003906  | 0.09375  | 0.03385 | -1.683     | -0.340         | -0.964          | 1.00             | **    | *        |
| presup_agro_desarrollo_rural |        10 |         0.312 | 0.007812  | 0.1797   | 0.04063 | +1.059     | +0.740         | +0.956          | 0.85             | **    | *        |
| presup_infraestructura       |        10 |         1     | 0.007812  | 0.1797   | 0.04063 | +1.121     | +0.760         | +0.956          | 0.88             | **    | *        |
| aprobacion_fin               |        10 |        -6.814 | 0.001953  | 0.05078  | 0.02539 | -2.504     | -0.400         | -1.000          | 1.00             | **    | *        |
| delta_isr_medio              |        10 |         0.812 | 0.001953  | 0.05078  | 0.02539 | +2.597     | +0.980         | +1.000          | 1.00             | **    | *        |
| coherencia_temporal          |        10 |        -7.143 | 0.0625    | 0.9375   | 0.1289  | -0.866     | -0.450         | -1.000          | 0.68             |       |          |
| presup_justicia              |        10 |         0.125 | 0.6211    | 1        | 0.694   | +0.201     | +0.250         | +0.200          | 0.09             |       |          |
| presup_servicio_deuda        |        10 |         0.243 | 0.4805    | 1        | 0.5724  | +0.082     | +0.200         | +0.289          | 0.06             |       |          |
| presup_proteccion_social     |        10 |        -0.75  | 0.04688   | 0.75     | 0.1108  | -0.842     | -0.530         | -0.778          | 0.66             | *     |          |
| presup_seguridad             |        10 |         0.25  | 0.4844    | 1        | 0.5724  | +0.225     | +0.200         | +0.289          | 0.10             |       |          |
| presup_educacion             |        10 |        -0.375 | 0.1602    | 1        | 0.2192  | -0.445     | -0.400         | -0.556          | 0.24             |       |          |
| presup_salud                 |        10 |        -0.375 | 0.1602    | 1        | 0.2192  | -0.445     | -0.400         | -0.556          | 0.24             |       |          |
| delta_iva_medio              |        10 |         0.362 | 0.6406    | 1        | 0.694   | +0.215     | +0.160         | +0.182          | 0.09             |       |          |
| reformas_radicales           |        10 |        -1     | 0.02734   | 0.5195   | 0.08887 | -1.000     | -0.570         | -0.844          | 0.80             | *     |          |
| reformas_totales             |        10 |         0     | 0.75      | 1        | 0.78    | +0.254     | +0.110         | +0.500          | 0.11             |       |          |
| diversidad_valores           |        10 |         0     | 0.125     | 1        | 0.2167  | +0.654     | +0.410         | +1.000          | 0.46             |       |          |
| turnos                       |        10 |         0     | —         | —        | —       | —          | —              | —               | —                |       |          |
| shocks_totales               |        10 |         0     | —         | —        | —       | —          | —              | —               | —                |       |          |
| idh_fin                      |        10 |        -0.185 | 0.06445   | 0.9375   | 0.1289  | -0.752     | -0.400         | -0.673          | 0.56             |       |          |
| estabilidad_fin              |        10 |        -4.457 | 0.01953   | 0.4102   | 0.07254 | -0.890     | -0.580         | -0.818          | 0.71             | *     |          |
| gobernabilidad_fin           |        10 |        -1.714 | 0.01953   | 0.4102   | 0.07254 | -1.102     | -0.240         | -0.818          | 0.87             | *     |          |
| bienestar_fin                |        10 |        -0.357 | 0.08398   | 1        | 0.156   | -0.707     | -0.280         | -0.636          | 0.51             |       |          |
| bienestar_ini                |        10 |         0     | —         | —        | —       | —          | —              | —               | —                |       |          |
| deuda_fin                    |        10 |         5.415 | 0.1934    | 1        | 0.2514  | +0.481     | +0.280         | +0.491          | 0.28             |       |          |
| aprobacion_ini               |        10 |         0     | —         | —        | —       | —          | —              | —               | —                |       |          |
| pobreza_delta                |        10 |         0.463 | 0.03711   | 0.668    | 0.09648 | +0.796     | +0.200         | +0.745          | 0.61             | *     |          |
| pobreza_fin                  |        10 |         0.463 | 0.03711   | 0.668    | 0.09648 | +0.796     | +0.200         | +0.745          | 0.61             | *     |          |
| pobreza_ini                  |        10 |         0     | —         | —        | —       | —          | —              | —               | —                |       |          |
| PIB_delta                    |        10 |      -684.624 | 0.1602    | 1        | 0.2192  | -0.476     | -0.100         | -0.527          | 0.27             |       |          |
| PIB_fin                      |        10 |      -684.624 | 0.1602    | 1        | 0.2192  | -0.476     | -0.100         | -0.527          | 0.27             |       |          |
| PIB_ini                      |        10 |         0     | —         | —        | —       | —          | —              | —               | —                |       |          |
| presup_otros                 |        10 |         0.062 | 0.9922    | 1        | 0.9922  | -0.025     | +0.080         | +0.022          | 0.05             |       |          |

Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande.

## 5. Mixed-effects (turn-level): `metric ~ gpt-4o-mini + (1|seed)`

Aprovecha las 8 × N obs por modelo en vez de colapsar a N. El efecto fijo de modelo es la diferencia esperada `gpt-4o-mini − claude-haiku-4-5` controlando por la correlación intra-seed. Más datos efectivos → IC95 más apretado y p-values más pequeños que el Wilcoxon end-of-horizon.

| metric                       | fixed_effect_b_minus_a   | ci95_lo   | ci95_hi   | p_value   | p_bh      |   n_obs |   n_seeds | sig_bh   |
|:-----------------------------|:-------------------------|:----------|:----------|:----------|:----------|--------:|----------:|:---------|
| aprobacion_presidencial      | +3.824                   | +1.846    | +5.801    | 0.0001512 | 0.0007561 |     160 |        10 | ***      |
| indice_protesta              | +3.456                   | +1.332    | +5.580    | 0.001426  | 0.003564  |     160 |        10 | **       |
| pib_usd_mm                   | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_salud                 | —                        | —         | —         | —         | —         |     160 |        10 |          |
| presup_justicia              | -0.074                   | -0.395    | +0.248    | 0.6532    | 0.6532    |     160 |        10 |          |
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
| ind_gobernabilidad           | +1.037                   | -0.027    | +2.102    | 0.05608   | 0.09347   |     160 |        10 |          |
| ind_bienestar                | —                        | —         | —         | —         | —         |     160 |        10 |          |
| pobreza_general              | -0.288                   | -1.286    | +0.710    | 0.5717    | 0.6532    |     160 |        10 |          |
| presup_otros                 | —                        | —         | —         | —         | —         |     160 |        10 |          |

## 7. Datos crudos

- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).
- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.
- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.
- `mixed_effects.csv` — coeficientes y CI95 del efecto del modelo.
- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).
- `presupuesto_ic95.png`, `outcomes_box.png`, `mixed_effects_forest.png`.
