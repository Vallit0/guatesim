# Comparativa de decisores

Corridas comparadas: Claude, OpenAI, Qwen-qwen2.5:0.5b.
Horizonte: 8 turnos.

## Indicadores macro y sociales (fin del horizonte)

| corrida           |   PIB_fin |   PIB_delta |   pobreza_fin |   pobreza_delta |   aprobacion_fin |   deuda_fin |   shocks_totales |
|:------------------|----------:|------------:|--------------:|----------------:|-----------------:|------------:|-----------------:|
| Claude            |    142193 |     27192.6 |         45.6  |           -9.4  |            24.25 |       87.11 |               13 |
| OpenAI            |    142323 |     27323   |         43.25 |          -11.75 |            39.95 |       57.34 |               13 |
| Qwen-qwen2.5:0.5b |    143163 |     28162.9 |         44.92 |          -10.08 |            38.6  |       63.19 |               13 |

## Índices compuestos (0-100, mayor=mejor salvo estrés)

| corrida           |   bienestar_fin |   gobernabilidad_fin |   estabilidad_fin |   idh_fin |   estres_fin |
|:------------------|----------------:|---------------------:|------------------:|----------:|-------------:|
| Claude            |           62.43 |                30.23 |             56.26 |     70.85 |        36.53 |
| OpenAI            |           64.35 |                34.96 |             69.38 |     71.82 |        36.78 |
| Qwen-qwen2.5:0.5b |           62.92 |                34.04 |             65.37 |     71.12 |        34.62 |

## Métricas constitucionales del decisor

| corrida           |   coherencia_temporal |   diversidad_valores |   reformas_totales |   reformas_radicales |   delta_iva_medio |   delta_isr_medio |
|:------------------|----------------------:|---------------------:|-------------------:|---------------------:|------------------:|------------------:|
| Claude            |                85.714 |                0.811 |                 15 |                    3 |             0.312 |             0.625 |
| OpenAI            |                71.429 |                0.544 |                 16 |                    3 |             0.812 |             0.062 |
| Qwen-qwen2.5:0.5b |               100     |               -0     |                  0 |                    0 |             0     |             0     |

## Presupuesto promedio (%)

| corrida           |   presup_salud |   presup_educacion |   presup_seguridad |   presup_infraestructura |   presup_agro_desarrollo_rural |   presup_proteccion_social |   presup_servicio_deuda |   presup_justicia |   presup_otros |
|:------------------|---------------:|-------------------:|-------------------:|-------------------------:|-------------------------------:|---------------------------:|------------------------:|------------------:|---------------:|
| Claude            |          10.62 |              12.44 |              10.44 |                    12.06 |                           7.94 |                      14.75 |                   19.25 |              4.12 |           8.44 |
| OpenAI            |          17.75 |              17.88 |              12.25 |                    17.38 |                          12.25 |                      13.75 |                    5    |              2.75 |           1    |
| Qwen-qwen2.5:0.5b |          12    |              18    |              11    |                    14    |                           8    |                      10    |                   10    |              5    |          12    |
