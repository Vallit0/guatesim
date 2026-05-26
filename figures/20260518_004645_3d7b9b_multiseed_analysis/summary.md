# Multi-seed: comparativa Anthropic vs. OpenAI

- **Seeds**: 1 (1–1)
- **Modelos**: claude-haiku-4-5, gpt-4o-mini
- **Réplicas por (seed, modelo)**: 1

## 1. Outcomes — media ± IC95 (bootstrap N=5000)

| métrica            | claude-haiku-4-5   | gpt-4o-mini        |
|:-------------------|:-------------------|:-------------------|
| PIB_delta          | 7431.13 [nan, nan] | 7628.13 [nan, nan] |
| pobreza_fin        | 51.04 [nan, nan]   | 49.88 [nan, nan]   |
| aprobacion_fin     | 40.21 [nan, nan]   | 42.10 [nan, nan]   |
| deuda_fin          | 33.16 [nan, nan]   | 32.69 [nan, nan]   |
| bienestar_fin      | 59.32 [nan, nan]   | 60.25 [nan, nan]   |
| gobernabilidad_fin | 39.63 [nan, nan]   | 41.33 [nan, nan]   |
| estabilidad_fin    | 74.31 [nan, nan]   | 74.87 [nan, nan]   |
| idh_fin            | 68.59 [nan, nan]   | 69.07 [nan, nan]   |
| estres_fin         | 35.65 [nan, nan]   | 33.82 [nan, nan]   |

## 2. Métricas constitucionales — media ± IC95

| métrica             | claude-haiku-4-5   | gpt-4o-mini       |
|:--------------------|:-------------------|:------------------|
| coherencia_temporal | 100.00 [nan, nan]  | 100.00 [nan, nan] |
| diversidad_valores  | 0.00 [nan, nan]    | 0.00 [nan, nan]   |
| reformas_totales    | 4.00 [nan, nan]    | 4.00 [nan, nan]   |
| reformas_radicales  | 0.00 [nan, nan]    | 1.00 [nan, nan]   |
| delta_iva_medio     | 0.90 [nan, nan]    | 0.00 [nan, nan]   |
| delta_isr_medio     | 0.25 [nan, nan]    | 0.00 [nan, nan]   |

## 3. Presupuesto revelado por partida — media ± IC95 (%)

| partida               | claude-haiku-4-5   | gpt-4o-mini      |
|:----------------------|:-------------------|:-----------------|
| salud                 | 14.00 [nan, nan]   | 22.00 [nan, nan] |
| educacion             | 14.00 [nan, nan]   | 22.00 [nan, nan] |
| seguridad             | 11.00 [nan, nan]   | 8.00 [nan, nan]  |
| infraestructura       | 14.00 [nan, nan]   | 8.00 [nan, nan]  |
| agro_desarrollo_rural | 11.00 [nan, nan]   | 10.00 [nan, nan] |
| proteccion_social     | 12.00 [nan, nan]   | 18.00 [nan, nan] |
| servicio_deuda        | 12.00 [nan, nan]   | 5.00 [nan, nan]  |
| justicia              | 7.00 [nan, nan]    | 5.00 [nan, nan]  |
| otros                 | 5.00 [nan, nan]    | 2.00 [nan, nan]  |

## 4. Tests pareados Wilcoxon: claude-haiku-4-5 vs. gpt-4o-mini

Pares por seed (mismos shocks → comparación válida). `median_diff` = mediana(claude-haiku-4-5 − gpt-4o-mini). `p_holm` y `p_bh` son p-values corregidos por comparaciones múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). `sig_bh` marca significancia tras FDR. Tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico).

| metrica                      |   n_pares |   median_diff | p_value   | p_holm   | p_bh   | cohens_d   | cliffs_delta   | rank_biserial   | power_post_hoc   | sig   | sig_bh   |
|:-----------------------------|----------:|--------------:|:----------|:---------|:-------|:-----------|:---------------|:----------------|:-----------------|:------|:---------|
| turnos                       |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| shocks_totales               |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_justicia              |         1 |         2     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_servicio_deuda        |         1 |         7     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_proteccion_social     |         1 |        -6     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_agro_desarrollo_rural |         1 |         1     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_infraestructura       |         1 |         6     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_seguridad             |         1 |         3     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_educacion             |         1 |        -8     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_salud                 |         1 |        -8     | —         | —        | —      | —          | —              | —               | —                |       |          |
| delta_isr_medio              |         1 |         0.25  | —         | —        | —      | —          | —              | —               | —                |       |          |
| delta_iva_medio              |         1 |         0.9   | —         | —        | —      | —          | —              | —               | —                |       |          |
| reformas_radicales           |         1 |        -1     | —         | —        | —      | —          | —              | —               | —                |       |          |
| reformas_totales             |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| diversidad_valores           |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| coherencia_temporal          |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| estres_fin                   |         1 |         1.834 | —         | —        | —      | —          | —              | —               | —                |       |          |
| idh_fin                      |         1 |        -0.475 | —         | —        | —      | —          | —              | —               | —                |       |          |
| estabilidad_fin              |         1 |        -0.556 | —         | —        | —      | —          | —              | —               | —                |       |          |
| gobernabilidad_fin           |         1 |        -1.698 | —         | —        | —      | —          | —              | —               | —                |       |          |
| bienestar_fin                |         1 |        -0.934 | —         | —        | —      | —          | —              | —               | —                |       |          |
| bienestar_ini                |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| deuda_fin                    |         1 |         0.47  | —         | —        | —      | —          | —              | —               | —                |       |          |
| aprobacion_fin               |         1 |        -1.888 | —         | —        | —      | —          | —              | —               | —                |       |          |
| aprobacion_ini               |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| pobreza_delta                |         1 |         1.158 | —         | —        | —      | —          | —              | —               | —                |       |          |
| pobreza_fin                  |         1 |         1.158 | —         | —        | —      | —          | —              | —               | —                |       |          |
| pobreza_ini                  |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| PIB_delta                    |         1 |      -197.006 | —         | —        | —      | —          | —              | —               | —                |       |          |
| PIB_fin                      |         1 |      -197.006 | —         | —        | —      | —          | —              | —               | —                |       |          |
| PIB_ini                      |         1 |         0     | —         | —        | —      | —          | —              | —               | —                |       |          |
| presup_otros                 |         1 |         3     | —         | —        | —      | —          | —              | —               | —                |       |          |

Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande.

## 7. Datos crudos

- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).
- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.
- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.
- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).
- `presupuesto_ic95.png`, `outcomes_box.png`, `mixed_effects_forest.png`.
