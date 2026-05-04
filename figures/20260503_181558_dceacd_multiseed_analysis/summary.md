# Multi-seed: comparativa Anthropic vs. OpenAI

- **Seeds**: 20 (1–20)
- **Modelos**: Claude, OpenAI
- **Réplicas por (seed, modelo)**: 1

## 1. Outcomes — media ± IC95 (bootstrap N=5000)

| métrica            | Claude                        | OpenAI                        |
|:-------------------|:------------------------------|:------------------------------|
| PIB_delta          | 31986.72 [30855.39, 33141.57] | 32717.81 [31631.11, 33813.64] |
| pobreza_fin        | 43.57 [42.93, 44.20]          | 41.84 [41.37, 42.29]          |
| aprobacion_fin     | 34.21 [30.12, 38.56]          | 45.19 [41.34, 49.07]          |
| deuda_fin          | 60.93 [55.12, 67.17]          | 58.67 [56.04, 61.48]          |
| bienestar_fin      | 64.15 [63.73, 64.56]          | 65.53 [65.27, 65.80]          |
| gobernabilidad_fin | 37.63 [35.32, 40.05]          | 43.40 [41.41, 45.46]          |
| estabilidad_fin    | 65.88 [63.00, 68.52]          | 67.06 [65.57, 68.55]          |
| idh_fin            | 72.17 [71.99, 72.34]          | 72.87 [72.76, 72.98]          |
| estres_fin         | 30.20 [29.38, 30.99]          | 29.18 [28.46, 29.94]          |

## 2. Métricas constitucionales — media ± IC95

| métrica             | Claude                | OpenAI                  |
|:--------------------|:----------------------|:------------------------|
| coherencia_temporal | 97.14 [93.57, 100.00] | 100.00 [100.00, 100.00] |
| diversidad_valores  | 0.11 [0.00, 0.23]     | 0.00 [0.00, 0.00]       |
| reformas_totales    | 15.95 [15.85, 16.00]  | 15.95 [15.85, 16.00]    |
| reformas_radicales  | 2.00 [1.45, 2.55]     | 3.85 [3.00, 4.70]       |
| delta_iva_medio     | 0.36 [0.26, 0.45]     | 0.02 [0.00, 0.04]       |
| delta_isr_medio     | 1.16 [0.94, 1.36]     | 0.02 [0.00, 0.03]       |

## 3. Presupuesto revelado por partida — media ± IC95 (%)

| partida               | Claude               | OpenAI               |
|:----------------------|:---------------------|:---------------------|
| salud                 | 17.20 [16.50, 17.90] | 20.03 [19.63, 20.43] |
| educacion             | 17.20 [16.50, 17.90] | 20.03 [19.61, 20.45] |
| seguridad             | 9.80 [9.54, 10.06]   | 8.73 [8.58, 8.88]    |
| infraestructura       | 11.60 [11.07, 12.16] | 9.44 [9.14, 9.76]    |
| agro_desarrollo_rural | 10.60 [10.51, 10.69] | 10.24 [10.20, 10.30] |
| proteccion_social     | 14.40 [13.88, 14.93] | 16.53 [16.23, 16.84] |
| servicio_deuda        | 9.20 [8.59, 9.81]    | 6.70 [6.36, 7.06]    |
| justicia              | 6.20 [6.03, 6.38]    | 5.51 [5.40, 5.63]    |
| otros                 | 3.80 [3.54, 4.08]    | 2.77 [2.60, 2.94]    |

## 4. Tests pareados Wilcoxon: Claude vs. OpenAI

Pares por seed (mismos shocks → comparación válida). `median_diff` = mediana(Claude − OpenAI). `p_holm` y `p_bh` son p-values corregidos por comparaciones múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). `sig_bh` marca significancia tras FDR. Tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico).

| metrica                      |   n_pares |   median_diff | p_value   | p_holm   | p_bh      | cohens_d   | cliffs_delta   | rank_biserial   | power_post_hoc   | sig   | sig_bh   |
|:-----------------------------|----------:|--------------:|:----------|:---------|:----------|:-----------|:---------------|:----------------|:-----------------|:------|:---------|
| estres_fin                   |        20 |         1.069 | 0.0003948 | 0.004783 | 0.0005922 | +1.039     | +0.315         | +0.838          | 0.99             | ***   | ***      |
| gobernabilidad_fin           |        20 |        -5.697 | 1.907e-06 | 5.15e-05 | 6.437e-06 | -3.057     | -0.565         | -1.000          | 1.00             | ***   | ***      |
| presup_justicia              |        20 |         0.75  | 0.000481  | 0.004783 | 0.0006494 | +1.321     | +0.835         | +0.961          | 1.00             | ***   | ***      |
| presup_servicio_deuda        |        20 |         2.625 | 0.0002795 | 0.004783 | 0.0004439 | +1.487     | +0.875         | +1.000          | 1.00             | ***   | ***      |
| presup_proteccion_social     |        20 |        -2.25  | 0.0002795 | 0.004783 | 0.0004439 | -1.469     | -0.870         | -1.000          | 1.00             | ***   | ***      |
| presup_agro_desarrollo_rural |        20 |         0.375 | 0.0002795 | 0.004783 | 0.0004439 | +1.472     | +0.870         | +1.000          | 1.00             | ***   | ***      |
| presup_infraestructura       |        20 |         2.25  | 0.0002795 | 0.004783 | 0.0004439 | +1.508     | +0.875         | +1.000          | 1.00             | ***   | ***      |
| presup_seguridad             |        20 |         1.125 | 0.0002795 | 0.004783 | 0.0004439 | +1.477     | +0.870         | +1.000          | 1.00             | ***   | ***      |
| presup_educacion             |        20 |        -3     | 0.0002795 | 0.004783 | 0.0004439 | -1.455     | -0.870         | -1.000          | 1.00             | ***   | ***      |
| presup_salud                 |        20 |        -3     | 0.0002795 | 0.004783 | 0.0004439 | -1.455     | -0.870         | -1.000          | 1.00             | ***   | ***      |
| delta_isr_medio              |        20 |         1.25  | 0.0001023 | 0.001945 | 0.000307  | +2.283     | +0.900         | +0.990          | 1.00             | ***   | ***      |
| delta_iva_medio              |        20 |         0.375 | 0.0002657 | 0.004783 | 0.0004439 | +1.391     | +0.875         | +0.929          | 1.00             | ***   | ***      |
| idh_fin                      |        20 |        -0.732 | 1.907e-06 | 5.15e-05 | 6.437e-06 | -1.659     | -0.890         | -1.000          | 1.00             | ***   | ***      |
| presup_otros                 |        20 |         1.125 | 0.000481  | 0.004783 | 0.0006494 | +1.322     | +0.835         | +0.961          | 1.00             | ***   | ***      |
| aprobacion_fin               |        20 |       -11.863 | 1.907e-06 | 5.15e-05 | 6.437e-06 | -3.876     | -0.590         | -1.000          | 1.00             | ***   | ***      |
| pobreza_delta                |        20 |         1.793 | 1.907e-06 | 5.15e-05 | 6.437e-06 | +1.706     | +0.635         | +1.000          | 1.00             | ***   | ***      |
| PIB_fin                      |        20 |      -719.07  | 1.907e-06 | 5.15e-05 | 6.437e-06 | -2.063     | -0.185         | -1.000          | 1.00             | ***   | ***      |
| PIB_delta                    |        20 |      -719.07  | 1.907e-06 | 5.15e-05 | 6.437e-06 | -2.063     | -0.185         | -1.000          | 1.00             | ***   | ***      |
| pobreza_fin                  |        20 |         1.793 | 1.907e-06 | 5.15e-05 | 6.437e-06 | +1.706     | +0.635         | +1.000          | 1.00             | ***   | ***      |
| bienestar_fin                |        20 |        -1.43  | 1.907e-06 | 5.15e-05 | 6.437e-06 | -1.611     | -0.780         | -1.000          | 1.00             | ***   | ***      |
| reformas_radicales           |        20 |        -2     | 0.0008698 | 0.006089 | 0.001118  | -1.112     | -0.557         | -1.000          | 1.00             | ***   | **       |
| shocks_totales               |        20 |         0     | 0.01431   | 0.08584  | 0.01756   | +0.638     | +0.062         | +1.000          | 0.77             | *     | *        |
| PIB_ini                      |        20 |         0     | —         | —        | —         | —          | —              | —               | —                |       |          |
| pobreza_ini                  |        20 |         0     | —         | —        | —         | —          | —              | —               | —                |       |          |
| estabilidad_fin              |        20 |         0.171 | 0.5217    | 1        | 0.5634    | -0.214     | -0.100         | -0.171          | 0.15             |       |          |
| aprobacion_ini               |        20 |         0     | —         | —        | —         | —          | —              | —               | —                |       |          |
| deuda_fin                    |        20 |        -3.26  | 0.8408    | 1        | 0.8732    | +0.170     | +0.080         | +0.057          | 0.11             |       |          |
| bienestar_ini                |        20 |         0     | —         | —        | —         | —          | —              | —               | —                |       |          |
| reformas_totales             |        20 |         0     | 1         | 1        | 1         | +0.000     | +0.000         | +0.000          | 0.05             |       |          |
| diversidad_valores           |        20 |         0     | 0.1025    | 0.5124   | 0.1153    | +0.402     | +0.150         | +1.000          | 0.40             |       |          |
| coherencia_temporal          |        20 |         0     | 0.1025    | 0.5124   | 0.1153    | -0.382     | -0.150         | -1.000          | 0.37             |       |          |
| turnos                       |        20 |         0     | —         | —        | —         | —          | —              | —               | —                |       |          |

Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande.

## 5. Mixed-effects (turn-level): `metric ~ OpenAI + (1|seed)`

Aprovecha las 8 × N obs por modelo en vez de colapsar a N. El efecto fijo de modelo es la diferencia esperada `OpenAI − Claude` controlando por la correlación intra-seed. Más datos efectivos → IC95 más apretado y p-values más pequeños que el Wilcoxon end-of-horizon.

| metric                       | fixed_effect_b_minus_a   | ci95_lo   | ci95_hi   | p_value   | p_bh      |   n_obs |   n_seeds | sig_bh   |
|:-----------------------------|:-------------------------|:----------|:----------|:----------|:----------|--------:|----------:|:---------|
| aprobacion_presidencial      | +6.157                   | +5.071    | +7.244    | 1.152e-28 | 5.759e-28 |     320 |        20 | ***      |
| ind_gobernabilidad           | +3.016                   | +2.420    | +3.611    | 3.231e-23 | 8.078e-23 |     320 |        20 | ***      |
| presup_proteccion_social     | +2.132                   | +1.525    | +2.739    | 6.009e-12 | 1.002e-11 |     320 |        20 | ***      |
| pobreza_general              | -1.117                   | -1.918    | -0.316    | 0.006274  | 0.007843  |     320 |        20 | **       |
| pib_usd_mm                   | —                        | —         | —         | —         | —         |     320 |        20 |          |
| presup_salud                 | —                        | —         | —         | —         | —         |     320 |        20 |          |
| presup_justicia              | —                        | —         | —         | —         | —         |     320 |        20 |          |
| presup_servicio_deuda        | —                        | —         | —         | —         | —         |     320 |        20 |          |
| presup_agro_desarrollo_rural | —                        | —         | —         | —         | —         |     320 |        20 |          |
| presup_infraestructura       | —                        | —         | —         | —         | —         |     320 |        20 |          |
| presup_seguridad             | —                        | —         | —         | —         | —         |     320 |        20 |          |
| presup_educacion             | —                        | —         | —         | —         | —         |     320 |        20 |          |
| delta_iva_pp                 | —                        | —         | —         | —         | —         |     320 |        20 |          |
| delta_isr_pp                 | —                        | —         | —         | —         | —         |     320 |        20 |          |
| deuda_pib                    | —                        | —         | —         | —         | —         |     320 |        20 |          |
| ind_estres_social            | —                        | —         | —         | —         | —         |     320 |        20 |          |
| ind_desarrollo_humano        | —                        | —         | —         | —         | —         |     320 |        20 |          |
| ind_estabilidad_macro        | —                        | —         | —         | —         | —         |     320 |        20 |          |
| ind_bienestar                | —                        | —         | —         | —         | —         |     320 |        20 |          |
| indice_protesta              | -0.362                   | -1.797    | +1.074    | 0.6215    | 0.6215    |     320 |        20 |          |
| presup_otros                 | —                        | —         | —         | —         | —         |     320 |        20 |          |

## 7. Datos crudos

- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).
- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.
- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.
- `mixed_effects.csv` — coeficientes y CI95 del efecto del modelo.
- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).
- `presupuesto_ic95.png`, `outcomes_box.png`, `mixed_effects_forest.png`.
