# Comparativa de decisores

Corridas comparadas: Claude, Dummy.
Horizonte: 8 turnos.

## Indicadores macro y sociales (fin del horizonte)

| corrida   |   PIB_fin |   PIB_delta |   pobreza_fin |   pobreza_delta |   aprobacion_fin |   deuda_fin |   shocks_totales |
|:----------|----------:|------------:|--------------:|----------------:|-----------------:|------------:|-----------------:|
| Claude    |    142786 |     27785.9 |         45.24 |           -9.76 |            18.34 |       64.27 |               13 |
| Dummy     |    143163 |     28162.9 |         44.92 |          -10.08 |            38.6  |       63.19 |               13 |

## Índices compuestos (0-100, mayor=mejor salvo estrés)

| corrida   |   bienestar_fin |   gobernabilidad_fin |   estabilidad_fin |   idh_fin |   estres_fin |
|:----------|----------------:|---------------------:|------------------:|----------:|-------------:|
| Claude    |           62.68 |                26.07 |             63.73 |     70.99 |        36.53 |
| Dummy     |           62.92 |                34.04 |             65.37 |     71.12 |        34.62 |

## Métricas constitucionales del decisor

| corrida   |   coherencia_temporal |   diversidad_valores |   reformas_totales |   reformas_radicales |   delta_iva_medio |   delta_isr_medio |
|:----------|----------------------:|---------------------:|-------------------:|---------------------:|------------------:|------------------:|
| Claude    |                   100 |                   -0 |                 12 |                    2 |            -0.125 |             1.812 |
| Dummy     |                   100 |                   -0 |                  0 |                    0 |             0     |             0     |

## Presupuesto promedio (%)

| corrida   |   presup_salud |   presup_educacion |   presup_seguridad |   presup_infraestructura |   presup_agro_desarrollo_rural |   presup_proteccion_social |   presup_servicio_deuda |   presup_justicia |   presup_otros |
|:----------|---------------:|-------------------:|-------------------:|-------------------------:|-------------------------------:|---------------------------:|------------------------:|------------------:|---------------:|
| Claude    |          10.94 |              12.35 |              11.38 |                    14.21 |                          11.06 |                      15.56 |                   15.69 |              4.44 |           4.38 |
| Dummy     |          12    |              18    |              11    |                    14    |                           8    |                      10    |                   10    |              5    |          12    |
