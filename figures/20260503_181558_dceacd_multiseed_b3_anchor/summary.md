# B3 — Human-process anchor (MINFIN 2024) vs LLM trajectories

Per-seed mean budget allocation across the 8 quarterly turns compared to the MINFIN 2024 appropriated/executed shares (ICEFI Tables 7 + 8, primary SICOIN data).

## 1. Per-model summary

| model   |   n |   L1 median (pp) | L1 IQR         |   cos median | cos IQR        |
|:--------|----:|-----------------:|:---------------|-------------:|:---------------|
| claude  |  20 |           59.100 | [58.60, 60.10] |        0.791 | [0.777, 0.793] |
| openai  |  20 |           60.600 | [60.10, 61.55] |        0.767 | [0.757, 0.777] |

## 2. Paired Wilcoxon (Claude vs OpenAI)

- L1 deviation vs MINFIN: median diff = -1.500 pp, p = 0.0003
- cos similarity vs MINFIN: median diff = +0.0218, p = 0.0003

## 3. Mean budget per model (% of total) vs MINFIN

| partida               |   MINFIN_2024 |   claude |   openai |
|:----------------------|--------------:|---------:|---------:|
| salud                 |          10.6 |     17.2 |    20.03 |
| educacion             |          22.1 |     17.2 |    20.03 |
| seguridad             |           8.6 |      9.8 |     8.73 |
| infraestructura       |           4.9 |     11.6 |     9.44 |
| agro_desarrollo_rural |           2   |     10.6 |    10.24 |
| proteccion_social     |          10.1 |     14.4 |    16.53 |
| servicio_deuda        |          14.7 |      9.2 |     6.7  |
| justicia              |           4   |      6.2 |     5.51 |
| otros                 |          23   |      3.8 |     2.77 |

## 4. Lectura

- Mediana de desviación L1 vs MINFIN: Claude 59.10 pp, GPT-4o-mini 60.60 pp. Claude se aleja menos del proceso humano que GPT-4o-mini.
- Comparación: el menu candidate más cercano a MINFIN (`status_quo_uniforme`) está a 52.9 pp; el más lejano (`seguridad_primero`) a 78.8 pp. Las trayectorias LLM se ubican dentro de ese rango.
