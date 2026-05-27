# B3 — Human-process anchor (MINFIN 2023) vs LLM trajectories

Per-seed mean budget allocation across the 8 quarterly turns compared to the MINFIN 2023 appropriated/executed shares (ICEFI Tables 7 + 8, primary SICOIN data).

## 1. Per-model summary

| model   |   n |   L1 median (pp) | L1 IQR         |   cos median | cos IQR        |
|:--------|----:|-----------------:|:---------------|-------------:|:---------------|
| claude  |  20 |           62.520 | [62.02, 63.65] |        0.761 | [0.747, 0.765] |
| openai  |  20 |           64.900 | [64.13, 66.15] |        0.737 | [0.725, 0.747] |

## 2. Paired Wilcoxon (Claude vs OpenAI)

- L1 deviation vs MINFIN: median diff = -2.380 pp, p = 0.0003
- cos similarity vs MINFIN: median diff = +0.0228, p = 0.0003

## 3. Mean budget per model (% of total) vs MINFIN

| partida               |   MINFIN 2023 |   claude |   openai |
|:----------------------|--------------:|---------:|---------:|
| salud                 |          9    |     17.2 |    20.03 |
| educacion             |         22.74 |     17.2 |    20.03 |
| seguridad             |          9.19 |      9.8 |     8.73 |
| infraestructura       |          5.02 |     11.6 |     9.44 |
| agro_desarrollo_rural |          1.31 |     10.6 |    10.24 |
| proteccion_social     |          9.97 |     14.4 |    16.53 |
| servicio_deuda        |         13.98 |      9.2 |     6.7  |
| justicia              |          4    |      6.2 |     5.51 |
| otros                 |         24.79 |      3.8 |     2.77 |

## 4. Lectura

- Mediana de desviación L1 vs MINFIN: Claude 62.52 pp, GPT-4o-mini 64.90 pp. Claude se aleja menos del proceso humano que GPT-4o-mini.
- Comparación: el menu candidate más cercano a MINFIN (`status_quo_uniforme`) está a 52.9 pp; el más lejano (`seguridad_primero`) a 78.8 pp. Las trayectorias LLM se ubican dentro de ese rango.
