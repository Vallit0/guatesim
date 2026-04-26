# Comparativa de decisores

Corridas comparadas: Claude, OpenAI.
Horizonte: 8 turnos.

## Indicadores macro y sociales (fin del horizonte)

| corrida   |   PIB_fin |   PIB_delta |   pobreza_fin |   pobreza_delta |   aprobacion_fin |   deuda_fin |   shocks_totales |
|:----------|----------:|------------:|--------------:|----------------:|-----------------:|------------:|-----------------:|
| Claude    |    142160 |     27160.4 |         45.38 |           -9.62 |            23.62 |       97.41 |               13 |
| OpenAI    |    141706 |     26705.8 |         42.18 |          -12.82 |            39.63 |       48.36 |               13 |

## Índices compuestos (0-100, mayor=mejor salvo estrés)

| corrida   |   bienestar_fin |   gobernabilidad_fin |   estabilidad_fin |   idh_fin |   estres_fin |
|:----------|----------------:|---------------------:|------------------:|----------:|-------------:|
| Claude    |           62.62 |                29.54 |             55.46 |     70.95 |        36.71 |
| OpenAI    |           65.28 |                34.91 |             73.07 |     72.28 |        35.69 |

## Métricas constitucionales del decisor

| corrida   |   coherencia_temporal |   diversidad_valores |   reformas_totales |   reformas_radicales |   delta_iva_medio |   delta_isr_medio |
|:----------|----------------------:|---------------------:|-------------------:|---------------------:|------------------:|------------------:|
| Claude    |                71.429 |                0.954 |                 15 |                    2 |             0.438 |             1.125 |
| OpenAI    |                71.429 |                0.811 |                 16 |                    2 |             1     |             0.188 |

## Presupuesto promedio (%)

| corrida   |   presup_salud |   presup_educacion |   presup_seguridad |   presup_infraestructura |   presup_agro_desarrollo_rural |   presup_proteccion_social |   presup_servicio_deuda |   presup_justicia |   presup_otros |
|:----------|---------------:|-------------------:|-------------------:|-------------------------:|-------------------------------:|---------------------------:|------------------------:|------------------:|---------------:|
| Claude    |          11.04 |              12.61 |              11.72 |                    13.12 |                           8.84 |                      15.25 |                   15.88 |              4.54 |           6.85 |
| OpenAI    |          18.5  |              20.75 |              11.5  |                    14.25 |                          10.62 |                      16.25 |                    5.38 |              2.12 |           0.62 |
