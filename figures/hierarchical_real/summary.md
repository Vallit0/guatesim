# Hierarchical Bayesian IRL: Claude vs OpenAI (N=20 seeds)

Hierarchical comparison Claude vs OpenAI: constituciones 2/6 decisivas; volatilidades 0/6 decisivas. Cosine(μ_a, μ_b) = +0.372 (HDI95 [-0.152, +0.865]). P(constituciones anti-alineadas) = 0.112.

## Constituciones (μ)

| feature                   |   p_a_gt_b |   diff_mean |   diff_hdi95_lo |   diff_hdi95_hi | decisive   |
|:--------------------------|-----------:|------------:|----------------:|----------------:|:-----------|
| anti_pobreza              |     +0.000 |      -1.516 |          -2.480 |          -0.551 | True       |
| anti_deuda                |     +0.953 |      +4.431 |          -0.464 |          +9.519 | True       |
| pro_aprobacion            |     +0.330 |      -0.130 |          -0.671 |          +0.455 | False      |
| pro_crecimiento           |     +0.084 |      -2.565 |          -6.216 |          +1.254 | False      |
| anti_desviacion_inflacion |     +0.611 |      +0.644 |          -3.505 |          +5.184 | False      |
| pro_confianza             |     +0.390 |      -0.801 |          -6.347 |          +4.336 | False      |

## Volatilidades (τ)

| feature                   |   p_a_gt_b |   diff_mean |   diff_hdi95_lo |   diff_hdi95_hi | decisive   |
|:--------------------------|-----------:|------------:|----------------:|----------------:|:-----------|
| anti_pobreza              |     +0.464 |      -0.055 |          -0.798 |          +0.641 | False      |
| anti_deuda                |     +0.483 |      -0.016 |          -1.694 |          +1.717 | False      |
| pro_aprobacion            |     +0.271 |      -0.173 |          -0.710 |          +0.355 | False      |
| pro_crecimiento           |     +0.447 |      -0.124 |          -1.638 |          +1.570 | False      |
| anti_desviacion_inflacion |     +0.531 |      +0.057 |          -1.852 |          +1.730 | False      |
| pro_confianza             |     +0.480 |      -0.032 |          -1.678 |          +1.711 | False      |
