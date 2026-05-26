# Multi-seed: comparativa Anthropic vs. OpenAI

- **Seeds**: 20 (1–20)
- **Modelos**: claude-haiku-4-5, gpt-4o-mini
- **Réplicas por (seed, modelo)**: 1

## 1. Outcomes — media ± IC95 (bootstrap N=5000)

| métrica            | claude-haiku-4-5              | gpt-4o-mini                   |
|:-------------------|:------------------------------|:------------------------------|
| PIB_delta          | 32336.98 [31199.72, 33495.50] | 32546.42 [31469.14, 33618.23] |
| pobreza_fin        | 44.82 [44.19, 45.43]          | 40.91 [40.39, 41.38]          |
| aprobacion_fin     | 33.86 [29.43, 38.15]          | 45.96 [41.86, 49.92]          |
| deuda_fin          | 66.08 [60.61, 71.75]          | 66.82 [64.16, 69.71]          |
| bienestar_fin      | 63.14 [62.75, 63.53]          | 66.30 [66.06, 66.57]          |
| gobernabilidad_fin | 37.60 [35.08, 40.14]          | 43.89 [41.49, 46.18]          |
| estabilidad_fin    | 63.02 [60.94, 65.21]          | 62.75 [61.47, 63.99]          |
| idh_fin            | 71.70 [71.56, 71.83]          | 73.26 [73.17, 73.35]          |
| estres_fin         | 30.90 [29.98, 31.81]          | 28.73 [27.87, 29.72]          |

## 2. Métricas constitucionales — media ± IC95

| métrica             | claude-haiku-4-5     | gpt-4o-mini           |
|:--------------------|:---------------------|:----------------------|
| coherencia_temporal | 93.23 [87.97, 97.74] | 98.57 [95.71, 100.00] |
| diversidad_valores  | 0.22 [0.08, 0.39]    | 0.03 [0.00, 0.08]     |
| reformas_totales    | 15.89 [15.74, 16.00] | 15.85 [15.60, 16.00]  |
| reformas_radicales  | 1.89 [1.47, 2.37]    | 2.35 [1.65, 3.10]     |
| delta_iva_medio     | 0.31 [0.21, 0.41]    | 0.04 [-0.01, 0.11]    |
| delta_isr_medio     | 0.90 [0.76, 1.05]    | 0.02 [0.00, 0.04]     |

## 3. Presupuesto revelado por partida — media ± IC95 (%)

| partida               | claude-haiku-4-5     | gpt-4o-mini          |
|:----------------------|:---------------------|:---------------------|
| salud                 | 14.96 [14.42, 15.54] | 21.80 [21.60, 21.95] |
| educacion             | 14.96 [14.42, 15.51] | 21.80 [21.60, 21.95] |
| seguridad             | 10.41 [10.19, 10.62] | 8.07 [8.02, 8.15]    |
| infraestructura       | 13.46 [12.97, 13.96] | 8.15 [8.04, 8.30]    |
| agro_desarrollo_rural | 11.16 [10.90, 11.45] | 10.03 [10.01, 10.05] |
| proteccion_social     | 12.84 [12.45, 13.25] | 17.85 [17.70, 17.96] |
| servicio_deuda        | 10.86 [10.39, 11.32] | 5.17 [5.04, 5.35]    |
| justicia              | 6.78 [6.59, 7.02]    | 5.05 [5.01, 5.10]    |
| otros                 | 4.57 [4.37, 4.76]    | 2.08 [2.02, 2.15]    |

## 4. Tests pareados Wilcoxon: claude-haiku-4-5 vs. gpt-4o-mini

Pares por seed (mismos shocks → comparación válida). `median_diff` = mediana(claude-haiku-4-5 − gpt-4o-mini). `p_holm` y `p_bh` son p-values corregidos por comparaciones múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). `sig_bh` marca significancia tras FDR. Tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico).

| metrica                      |   n_pares |   median_diff | p_value   | p_holm    | p_bh      | cohens_d   | cliffs_delta   | rank_biserial   | power_post_hoc   | sig   | sig_bh   |
|:-----------------------------|----------:|--------------:|:----------|:----------|:----------|:-----------|:---------------|:----------------|:-----------------|:------|:---------|
| estres_fin                   |        19 |         2.251 | 7.629e-06 | 0.0001602 | 2.943e-05 | +2.151     | +0.546         | +0.989          | —                | ***   | ***      |
| gobernabilidad_fin           |        19 |        -5.865 | 3.815e-06 | 0.000103  | 1.717e-05 | -3.356     | -0.607         | -1.000          | 1.00             | ***   | ***      |
| presup_justicia              |        19 |         1.75  | 0.0001197 | 0.002393  | 0.0002048 | +3.365     | +1.000         | +1.000          | 1.00             | ***   | ***      |
| presup_servicio_deuda        |        19 |         6.125 | 0.0001273 | 0.002393  | 0.0002048 | +4.943     | +1.000         | +1.000          | 1.00             | ***   | ***      |
| presup_proteccion_social     |        19 |        -5.25  | 0.0001271 | 0.002393  | 0.0002048 | -4.983     | -1.000         | -1.000          | 1.00             | ***   | ***      |
| presup_agro_desarrollo_rural |        19 |         0.875 | 0.0001267 | 0.002393  | 0.0002048 | +1.774     | +1.000         | +1.000          | 1.00             | ***   | ***      |
| presup_infraestructura       |        19 |         5.25  | 0.0001253 | 0.002393  | 0.0002048 | +4.324     | +1.000         | +1.000          | —                | ***   | ***      |
| presup_seguridad             |        19 |         2.25  | 0.0001218 | 0.002393  | 0.0002048 | +4.323     | +1.000         | +1.000          | —                | ***   | ***      |
| presup_educacion             |        19 |        -7     | 0.0001253 | 0.002393  | 0.0002048 | -4.845     | -1.000         | -1.000          | 1.00             | ***   | ***      |
| presup_salud                 |        19 |        -7     | 0.0001253 | 0.002393  | 0.0002048 | -4.845     | -1.000         | -1.000          | 1.00             | ***   | ***      |
| delta_isr_medio              |        19 |         0.875 | 0.000129  | 0.002393  | 0.0002048 | +2.749     | +1.000         | +1.000          | —                | ***   | ***      |
| idh_fin                      |        19 |        -1.643 | 3.815e-06 | 0.000103  | 1.717e-05 | -5.006     | -1.000         | -1.000          | 1.00             | ***   | ***      |
| presup_otros                 |        19 |         2.625 | 0.0001197 | 0.002393  | 0.0002048 | +4.878     | +1.000         | +1.000          | —                | ***   | ***      |
| bienestar_fin                |        19 |        -3.237 | 3.815e-06 | 0.000103  | 1.717e-05 | -4.992     | -1.000         | -1.000          | 1.00             | ***   | ***      |
| aprobacion_fin               |        19 |       -11.657 | 3.815e-06 | 0.000103  | 1.717e-05 | -4.353     | -0.601         | -1.000          | 1.00             | ***   | ***      |
| pobreza_fin                  |        19 |         3.991 | 3.815e-06 | 0.000103  | 1.717e-05 | +5.017     | +0.978         | +1.000          | —                | ***   | ***      |
| pobreza_delta                |        19 |         3.991 | 3.815e-06 | 0.000103  | 1.717e-05 | +5.017     | +0.978         | +1.000          | —                | ***   | ***      |
| PIB_fin                      |        19 |      -206.324 | 0.003342  | 0.03342   | 0.004642  | -0.652     | -0.091         | -0.737          | 0.77             | **    | **       |
| PIB_delta                    |        19 |      -206.324 | 0.003342  | 0.03342   | 0.004642  | -0.652     | -0.091         | -0.737          | 0.77             | **    | **       |
| delta_iva_medio              |        19 |         0.375 | 0.003438  | 0.03342   | 0.004642  | +0.916     | +0.676         | +0.763          | 0.97             | **    | **       |
| PIB_ini                      |        19 |         0     | —         | —         | —         | —          | —              | —               | —                |       |          |
| pobreza_ini                  |        19 |         0     | —         | —         | —         | —          | —              | —               | —                |       |          |
| aprobacion_ini               |        19 |         0     | —         | —         | —         | —          | —              | —               | —                |       |          |
| deuda_fin                    |        19 |        -1.252 | 0.5678    | 1         | 0.6132    | -0.047     | -0.080         | -0.158          | 0.05             |       |          |
| estabilidad_fin              |        19 |         0.185 | 1         | 1         | 1         | +0.053     | -0.091         | +0.000          | 0.06             |       |          |
| reformas_radicales           |        19 |         0     | 0.2698    | 1         | 0.3035    | -0.310     | -0.213         | -0.341          | 0.25             |       |          |
| reformas_totales             |        19 |         0     | 0.7055    | 1         | 0.7326    | +0.085     | +0.006         | +0.200          | 0.06             |       |          |
| diversidad_valores           |        19 |         0     | 0.05367   | 0.322     | 0.06586   | +0.485     | +0.269         | +0.786          | 0.52             |       |          |
| coherencia_temporal          |        19 |         0     | 0.1405    | 0.7023    | 0.1649    | -0.386     | -0.255         | -0.607          | 0.36             |       |          |
| shocks_totales               |        19 |         0     | 0.0455    | 0.3185    | 0.0585    | +0.503     | +0.050         | +1.000          | 0.55             | *     |          |
| bienestar_ini                |        19 |         0     | —         | —         | —         | —          | —              | —               | —                |       |          |
| turnos                       |        19 |         0     | —         | —         | —         | —          | —              | —               | —                |       |          |

Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande.

## 5. Mixed-effects (turn-level): `metric ~ gpt-4o-mini + (1|seed)`

Aprovecha las 8 × N obs por modelo en vez de colapsar a N. El efecto fijo de modelo es la diferencia esperada `gpt-4o-mini − claude-haiku-4-5` controlando por la correlación intra-seed. Más datos efectivos → IC95 más apretado y p-values más pequeños que el Wilcoxon end-of-horizon.

| metric                       | fixed_effect_b_minus_a   | ci95_lo   | ci95_hi   | p_value    | p_bh       |   n_obs |   n_seeds | sig_bh   |
|:-----------------------------|:-------------------------|:----------|:----------|:-----------|:-----------|--------:|----------:|:---------|
| presup_otros                 | -2.497                   | -2.677    | -2.317    | 5.255e-163 | 3.153e-162 |     312 |        20 | ***      |
| aprobacion_presidencial      | +6.534                   | +5.397    | +7.670    | 1.848e-29  | 3.696e-29  |     312 |        20 | ***      |
| ind_gobernabilidad           | +3.343                   | +2.734    | +3.952    | 5.364e-27  | 8.046e-27  |     312 |        20 | ***      |
| presup_infraestructura       | -5.310                   | -5.776    | -4.844    | 1.101e-110 | 3.302e-110 |     312 |        20 | ***      |
| indice_protesta              | -1.671                   | -3.181    | -0.161    | 0.03011    | 0.03613    |     312 |        20 | *        |
| presup_salud                 | —                        | —         | —         | —          | —          |     312 |        20 |          |
| presup_justicia              | —                        | —         | —         | —          | —          |     312 |        20 |          |
| presup_servicio_deuda        | —                        | —         | —         | —          | —          |     312 |        20 |          |
| presup_proteccion_social     | —                        | —         | —         | —          | —          |     312 |        20 |          |
| presup_agro_desarrollo_rural | —                        | —         | —         | —          | —          |     312 |        20 |          |
| presup_seguridad             | —                        | —         | —         | —          | —          |     312 |        20 |          |
| presup_educacion             | —                        | —         | —         | —          | —          |     312 |        20 |          |
| pib_usd_mm                   | —                        | —         | —         | —          | —          |     312 |        20 |          |
| delta_isr_pp                 | —                        | —         | —         | —          | —          |     312 |        20 |          |
| deuda_pib                    | +0.405                   | -2.497    | +3.307    | 0.7845     | 0.7845     |     312 |        20 |          |
| ind_estres_social            | —                        | —         | —         | —          | —          |     312 |        20 |          |
| ind_desarrollo_humano        | —                        | —         | —         | —          | —          |     312 |        20 |          |
| ind_estabilidad_macro        | —                        | —         | —         | —          | —          |     312 |        20 |          |
| ind_bienestar                | —                        | —         | —         | —          | —          |     312 |        20 |          |
| pobreza_general              | —                        | —         | —         | —          | —          |     312 |        20 |          |
| delta_iva_pp                 | —                        | —         | —         | —          | —          |     312 |        20 |          |

## 7. Datos crudos

- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).
- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.
- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.
- `mixed_effects.csv` — coeficientes y CI95 del efecto del modelo.
- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).
- `presupuesto_ic95.png`, `outcomes_box.png`, `mixed_effects_forest.png`.
