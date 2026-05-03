# IRL Recovery Curve — Resumen

## Setup

- **w\* verdadero**: `[1.5, 0.5, 1.0, 1.2, 0.8, 0.7]`
- **Dimensiones de bienestar (φ)**: `['anti_pobreza', 'anti_deuda', 'pro_aprobacion', 'pro_crecimiento', 'anti_desviacion_inflacion', 'pro_confianza']`
- **K candidatos**: 5 (random gaussian, anclados en candidato 0)
- **Réplicas por N**: 10

## Resultados

| N | reps | RMSE mediana | RMSE IQR | cos sim mediana | cos sim mín | norm_ratio mediana |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 10 | 0.2672 | [0.1958, 0.2954] | 0.9771 | 0.9679 | 1.048 |
| 100 | 10 | 0.1502 | [0.1396, 0.1886] | 0.9915 | 0.9847 | 1.053 |
| 200 | 10 | 0.1345 | [0.1275, 0.1735] | 0.9942 | 0.9846 | 1.023 |
| 500 | 10 | 0.0736 | [0.0666, 0.0861] | 0.9984 | 0.9969 | 1.005 |
| 1000 | 10 | 0.0485 | [0.0411, 0.0601] | 0.9993 | 0.9987 | 1.005 |
| 2000 | 10 | 0.0457 | [0.0381, 0.0642] | 0.9995 | 0.9989 | 1.022 |
| 5000 | 10 | 0.0268 | [0.0208, 0.0285] | 0.9998 | 0.9994 | 0.998 |

## Lectura

- **RMSE mediana** debe decrecer con N. Si la pendiente en log-log es ≈ -0.5, el método tiene la convergencia esperada O(1/√N).
- **cos sim** debe acercarse a 1 con N grande. Umbral típico para considerar recovery exitoso: 0.95.
- **norm_ratio** debe acercarse a 1; valores muy distintos indican que el MLE recupera la *dirección* pero no la *magnitud* de las preferencias (esperable con N pequeño).

## Validez del setup IRL

Si esta tabla muestra cos sim ≥ 0.95 con N ≥ 1000 y RMSE decreciente monótono, el setup matemático del Boltzmann likelihood + MLE es correcto y podemos pasar a `bayesian_irl.py` con confianza de que los problemas de allá adelante son del modelo PyMC, no del IRL.