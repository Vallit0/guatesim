# guatemala-sim

> *Testbed* para evaluar LLMs como decisores ejecutivos sobre una Guatemala
> simulada — calibrada con datos reales del Banco Mundial y Banguat,
> agentes heterogéneos del entorno político local, y análisis estadístico
> publication-ready (frecuentista + bayesiano).

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#licencia)
[![Tests](https://img.shields.io/badge/tests-88%20passed-success.svg)](#tests)
[![Status](https://img.shields.io/badge/status-research%20alpha-orange.svg)](#estado-y-roadmap)
[![Data](https://img.shields.io/badge/data-WB%202024%20%2B%20Banguat%202026-orange.svg)](#datos-reales-banco-mundial)
[![Stats](https://img.shields.io/badge/stats-Wilcoxon%20%7C%20MixedLM%20%7C%20BEST%20%7C%20Dirichlet--multinomial-blueviolet.svg)](#análisis-estadístico-multi-seed)

El mismo mundo (dinámica macro + shocks endógenos + 4 agentes Mesa + grafo
territorial de 22 departamentos, **calibrado contra Banco Mundial 2024**)
se expone a uno o más LLMs durante `N` turnos trimestrales. Cada turno, el
LLM produce una decisión estructurada (presupuesto que suma 100 %,
política fiscal, política exterior, respuestas a shocks, hasta 2 reformas).
Con seeds idénticos y prompts idénticos, **la única fuente de variación es
el modelo**, lo que permite medir lo que cada LLM *revela* como sus
preferencias de política pública sobre un país real.

**Por qué Guatemala (y no un país sintético):** los shocks (sequía del
corredor seco, deportaciones masivas desde EE.UU., escándalos de
corrupción cíclicos), los agentes (CACIF, magisterio, organizaciones
indígenas) y la geografía (22 departamentos con brechas de pobreza de 25
a 83 %) están calibrados contra el ciclo político y económico real de
Guatemala 2024–2026. La especificidad es el punto: un *testbed* genérico
mediría cosas distintas.

---

## Tabla de contenidos

- [Quickstart](#quickstart)
- [Qué hace, por qué](#qué-hace-por-qué)
- [Arquitectura](#arquitectura)
- [Decisores soportados](#decisores-soportados)
- [Modelos formales del simulador](#modelos-formales-del-simulador)
- [Análisis estadístico (multi-seed)](#análisis-estadístico-multi-seed)
- [Análisis bayesiano](#análisis-bayesiano)
- [Datos reales (Banco Mundial)](#datos-reales-banco-mundial)
- [Ejemplos](#ejemplos)
- [Configuración](#configuración)
- [Tests](#tests)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Estado y roadmap](#estado-y-roadmap)
- [Contribuir](#contribuir)
- [Cita](#cita)
- [Referencias](#referencias)
- [Licencia](#licencia)

---

## Quickstart

```bash
git clone <repo> guatemala-sim
cd guatemala-sim
pip install -e .[dev,ingest,bayes]
python -m pytest                              # 88 tests (~2 min con bayesianos)

# (opcional) descargar datos reales del Banco Mundial + Banguat
python -m guatemala_sim.refresh_data --banguat

# corrida de 4 turnos sin API
python demo.py --turnos 4

# Anthropic vs. OpenAI sobre el mismo mundo
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
python compare_llms.py --seed 11 --turnos 8

# análisis multi-seed con tests pareados, mixed-effects e ICC
python compare_llms_multiseed.py \
  --seeds-from 1 --seeds-to 20 --turnos 8 \
  --replicas 3 --continuar-si-falla
```

---

## Qué hace, por qué

Los LLMs cada vez se usan más como soporte de decisión: análisis de
políticas, asignación presupuestaria, recomendaciones regulatorias.
Sabemos que cuando dos modelos producen *texto libre* sobre la misma
pregunta dan respuestas distintas. Lo que **no** sabíamos antes de este
testbed era: si los obligás a elegir entre acciones cuantitativas con
restricciones agregadas (un presupuesto que tiene que sumar 100 %), bajo
exactamente los mismos shocks externos y exactamente el mismo *prompt*,
¿se promedian sus respuestas hacia un consenso, o cada modelo trae una
"constitución" implícita estable?

Resultado preliminar (1 corrida, 8 turnos): **no convergen**. Claude
Haiku 4.5 asigna 19 % al servicio de la deuda y 10 % a salud; GPT-4o-mini
asigna 5 % a deuda y 18 % a salud. La diferencia es del orden de varios
puntos porcentuales del PIB. Ver `paper/constituciones_reveladas.md`.

Cuán robusto es ese hallazgo todavía es una pregunta abierta — para eso
está el pipeline de **multi-seed con ≥ 20 seeds × ≥ 3 réplicas**, que
desambigua "diferencia entre modelos" de "ruido del sampler estocástico del LLM"
vía ICC. Este repo es la infraestructura para hacer ese tipo de pregunta
de manera reproducible y estadísticamente honesta:

- **Frecuentista**: multi-seed, IC95 bootstrap, Wilcoxon pareado con
  corrección por comparaciones múltiples (Holm + BH-FDR), mixed-effects
  sobre datos turn-level, ICC test-retest.
- **Bayesiano**: BEST de Kruschke 2013 (Student-t robusto, posterior
  completo de Δμ con HDI95 y ROPE) y Dirichlet-multinomial jerárquico
  sobre el presupuesto (recupera la "constitución" de cada LLM como
  prior con incertidumbre cuantificada).

---

## Arquitectura

```
┌──────────────────┐       ┌─────────────────────────────────┐
│  bootstrap.py    │◀──────│ data_ingest.py  (Banco Mundial) │
│ (estado inicial  │◀──────│ banguat_ingest.py (USD/GTQ)     │
│  enero 2026)     │       │ snapshot CSV → calibrate_state  │
└────────┬─────────┘       └─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                  engine.run_turn (loop)                      │
│                                                              │
│  shocks ─► contexto ─► DECISOR ─► macro ─► agentes ─►        │
│   (Bernoulli)         (LLM o Dummy)  (ec. dif)  (reglas)     │
│                                          │                   │
│                                          ▼                   │
│                                   territorio (grafo)         │
│                                          │                   │
│                                          ▼                   │
│                                   memoria + log JSONL        │
└─────────────────────────────────────────────────────────────┘
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
 comparison.py       multiseed.py        bayesian.py
 (single compare)   (N seeds, IC95,    (BEST + Dirichlet
                    Wilcoxon, ME, ICC)   multinomial, PyMC)
```

---

## Decisores soportados

| Decisor | Backend | Structured output | Costo |
|---|---|---|---|
| `ClaudePresidente` | Anthropic API | `tool_use` con schema [^anthropic-tools] | API metered |
| `GPTPresidente` | OpenAI cloud o cualquier endpoint OpenAI-compat | `json_schema` strict (cloud) [^openai-structured] o `json_object` loose (Ollama, LM Studio, vLLM) | API metered o local |
| `DummyDecisionMaker` | local | n/a | gratis |
| `ResilientDecisionMaker` | wrapper | delega a fallback si falla | — |

---

## Modelos formales del simulador

El simulador es **deterministicamente reactivo a la decisión presidencial,
con ruido gaussiano aditivo y eventos Bernoulli endógenos**. No usa RL,
ni equilibrio general computable, ni solvers opacos. La elección es
deliberada: queremos consecuencias trazables a las decisiones.

### Crecimiento del PIB

Multiplicador del gasto Keynesiano [^mankiw] [^ilzetzki-fiscal] con
componentes externos calibrados:

$$g(t) = g_{\text{tend}} + \mu \cdot \big(w_{\text{infra}}(t) + w_{\text{agro}}(t) - 0.20\big) + \eta_{\text{IED}} \cdot \left(\frac{\text{IED}(t)}{1000} - 1.5\right) + 0.06 \cdot (\rho(t) - 18) - 0.15 \cdot \max(\Delta_{\text{IVA}}(t), 0) + \varepsilon_t - \sum_k \pi_k \mathbb{1}[s_k(t)]$$

$$\varepsilon_t \sim \mathcal{N}(0, 0.35^2), \qquad \text{PIB}(t+1) = \text{PIB}(t) \cdot \big(1 + g(t)/100\big)$$

| Símbolo | Descripción | Valor |
|---|---|---|
| $g_{\text{tend}}$ | crecimiento tendencial | 3.3 % |
| $\mu$ | multiplicador del gasto productivo | 0.7 |
| $\eta_{\text{IED}}$ | elasticidad del crecimiento a IED | 0.05 |
| $\rho(t)$ | remesas / PIB (%) | observado |
| $\pi_k$ | penalización por shock k | sequía 0.8, huracán 1.0, remesas 0.6, deportaciones 0.3 |

### Inflación (AR(1) con anclaje)

Proceso autorregresivo de orden 1 con anclaje a la meta y *pass-through*
cambiario, en línea con la familia de modelos Neokeynesianos de juguete
[^stock-watson] y la regla operativa de Banguat [^banguat-policy]:

$$\pi(t+1) = \alpha \cdot \pi(t) + (1 - \alpha) \cdot \pi^{*} + \theta \cdot (\text{TC}(t) - 7.75) + 0.25 \cdot \text{brecha}(t) + 0.4 \cdot \nu_t, \quad \nu_t \sim \mathcal{N}(0, 0.5^2)$$

con $\alpha = 0.55$ (inercia), $\pi^{*} = 4\%$ (meta Banguat), $\theta = 0.15$
(pass-through), $\text{brecha}(t) = g(t) - g_{\text{tend}}$.

### Fiscal y deuda

Identidad contable + elasticidades tributarias [^cottarelli-tax]:

$$\Delta\text{ingresos}_{\text{PIB}}(t) = \varepsilon_{\text{IVA}} \cdot \Delta_{\text{IVA}}(t) \cdot w_{\text{IVA}} + \varepsilon_{\text{ISR}} \cdot \Delta_{\text{ISR}}(t) \cdot (1 - w_{\text{IVA}}) + 0.1 \cdot \text{brecha}(t)$$

$$\text{balance}_{\text{fiscal}}(t+1) = \text{balance}_{\text{fiscal}}(t) + \Delta\text{ingresos}_{\text{PIB}}(t) - \sum_j c_j(t) + \xi_t$$

$$\text{deuda}_{\text{PIB}}(t+1) = \text{deuda}_{\text{PIB}}(t) - \text{balance}_{\text{fiscal}}(t+1)$$

con $\varepsilon_{\text{IVA}} = 0.7$, $\varepsilon_{\text{ISR}} = 0.6$, $w_{\text{IVA}} = 0.45$,
$c_j$ costo fiscal de la respuesta a shock $j$.

### Sector externo (random walks con drift)

$$\text{TC}(t+1) = \text{TC}(t) \cdot \left(1 + \frac{\pi(t) - 2}{200} + 0.002 \cdot \mathcal{N}(0, 1)\right)$$

$$\text{reservas}(t+1) = 1.04 \cdot \text{reservas}(t) + 300 \cdot \text{CC}_{\text{PIB}}(t) + 200 \cdot \mathcal{N}(0, 1)$$

$$\text{IED}(t+1) = \text{IED}(t) \cdot \left(1.02 + 0.001 \cdot (\text{conf}_{\text{inst}}(t) - 30)\right) + 150 \cdot \mathcal{N}(0, 1)$$

### Pobreza y migración

Elasticidades Ravallion-style [^ravallion-poverty]:

$$\Delta\text{pobreza}(t) = -0.35 \cdot g(t) - 2.5 \cdot (w_{\text{social}}(t) - 0.35) + 0.4 \cdot \mathcal{N}(0, 1)$$

$$\text{migración}_{\text{neta}}(t) = -4 \cdot (\text{pobreza}(t) - 45) + 5 \cdot \mathcal{N}(0, 1) \quad \text{(en miles)}$$

donde $w_{\text{social}}$ es la fracción del presupuesto en
salud + educación + protección social.

### Shocks (Bernoulli endógeno)

Para cada uno de 7 shocks $k \in \{$sequía corredor seco, huracán,
caída de remesas, deportaciones masivas, escándalo de corrupción, crisis
de gobernabilidad, colapso vecino$\}$:

$$S_k(t) \sim \text{Bernoulli}(p_k(\text{state}(t)))$$

con probabilidades base moduladas por el estado:

$$p_{\text{corrupción}}(t) = p^{\text{base}}_{\text{corrupción}} \cdot \left(1.5 - \frac{\text{conf}_{\text{inst}}(t)}{100}\right)$$

$$p_{\text{crisis-gob}}(t) = p^{\text{base}}_{\text{crisis-gob}} + 0.3 \cdot \frac{\text{protesta}(t)}{100} + 0.3 \cdot \max\left(0, \frac{50 - \text{aprobación}(t)}{100}\right)$$

Es un proceso de eventos raros con feedback estado→probabilidad —
emparentado con un proceso de Hawkes auto-excitable [^hawkes], aunque sin
la integral autoregresiva explícita.

### Agentes (reglas determinísticas)

Cada uno de los 4 agentes implementa $f: (\text{state}, \text{decisión}) \to \text{Impacto}$,
donde $\text{Impacto}$ es un vector de deltas aditivos sobre aprobación,
protesta, coalición, confianza institucional e IED. Sin estocasticidad
propia, sin softmax, sin aprendizaje. Mesa v3 [^mesa] solo se usa como
framework de *scheduling*.

### LLM como decisor (sampler estocástico)

Cada token producido por Claude o GPT-4o-mini sale del *softmax*
estándar [^bishop-prml]:

$$P(\text{token} = v \mid \text{contexto}) = \frac{\exp(\ell_v / T)}{\sum_{v'} \exp(\ell_{v'} / T)}$$

donde $\ell_v$ es el logit del token $v$ y $T$ es la temperatura. La
temperatura por defecto en ambas APIs es $T \approx 1$; nosotros no la
sobrescribimos. La estocasticidad del decisor entre re-ejecuciones del
mismo turno proviene íntegramente de este sampler — es la única fuente
de aleatoriedad del decisor en el sistema.

### Indicadores derivados

Cinco índices compuestos $\in [0, 100]$ por suma ponderada (ver
`indicators.py` para los pesos exactos). Métricas constitucionales del
decisor sobre la serie de decisiones:

- **Coherencia temporal** ∈ [0, 100]:
$\text{coh}(D) = 100 \cdot \left(1 - \frac{1}{|D|-1} \sum_{t=1}^{|D|-1} \mathbb{1}[a_t \neq a_{t-1}]\right)$
donde $a_t$ es el alineamiento exterior del turno $t$.

- **Diversidad de valores** (entropía de Shannon [^shannon-1948]):
$$H(D) = -\sum_{a \in \mathcal{A}} \hat{p}(a) \log_2 \hat{p}(a)$$
donde $\hat{p}(a)$ es la frecuencia empírica del alineamiento $a$ en
$D$. Acá se aplica a la distribución categórica revelada del decisor.

---

## Análisis estadístico (multi-seed)

### Tests pareados (Wilcoxon signed-rank [^wilcoxon-1945])

Para cada métrica $X$ con valores pareados $(x_a^{(i)}, x_b^{(i)})$ en
$i = 1, \ldots, N$ seeds:

$$d_i = x_a^{(i)} - x_b^{(i)}, \quad R_i = \text{rank}(|d_i|)$$

$$W^{+} = \sum_{i: d_i > 0} R_i, \quad W^{-} = \sum_{i: d_i < 0} R_i$$

$p$-value vía aproximación normal o exacta para $N$ chico.

### Correcciones por comparaciones múltiples

Probamos ~30 métricas en paralelo entre dos modelos, así que sin
corrección $P(\geq 1 \text{ falso positivo}) \approx 1 - 0.95^{30} \approx 0.78$.
Reportamos dos correcciones:

**Holm-Bonferroni** [^holm-1979] (control familywise $\alpha$):

$$p^{\text{Holm}}_{(i)} = \max_{j \leq i} \min\big((m - j + 1) \cdot p_{(j)}, 1\big)$$

donde $p_{(1)} \leq \ldots \leq p_{(m)}$ son los $p$-values ordenados.

**Benjamini-Hochberg FDR** [^bh-1995] (control de tasa de falsos
descubrimientos):

$$p^{\text{BH}}_{(i)} = \min_{j \geq i} \min\left(\frac{m \cdot p_{(j)}}{j}, 1\right)$$

### Tamaños de efecto

**Cohen's d pareado** [^cohen-1988]: $d = \bar{d} / s_d$ donde $\bar{d}$ y
$s_d$ son media y desvío de las diferencias pareadas.

**Cliff's δ** [^cliff-1993] [^romano-2006] (no-paramétrico, robusto a
outliers, $\delta \in [-1, 1]$):

$$\delta = \frac{n_{>} - n_{<}}{n_a \cdot n_b}, \quad n_{>} = \lvert\{(a, b) : a > b\}\rvert$$

**Power post-hoc** vía la distribución $t$ no-central [^cohen-1988]:

$$1 - \beta = 1 - F_{t'}(t_{\alpha/2, df} \mid df, \lambda) + F_{t'}(-t_{\alpha/2, df} \mid df, \lambda)$$

con $\lambda = d \sqrt{N}$ el parámetro de no-centralidad.

### Mixed-effects sobre datos turn-level

Aprovechamos las $8 \cdot N$ observaciones en vez de las $N$ del análisis
fin-de-horizonte, ajustando un modelo lineal mixto [^laird-ware]
[^pinheiro-bates]:

$$y_{ist} = \beta_0 + \beta_1 \cdot \mathbb{1}[\text{modelo}_{is} = B] + u_s + \varepsilon_{ist}$$

con $u_s \sim \mathcal{N}(0, \sigma^2_u)$ el efecto aleatorio del seed
$s$ y $\varepsilon_{ist} \sim \mathcal{N}(0, \sigma^2_\varepsilon)$. El
coeficiente $\beta_1$ es la diferencia esperada $B - A$ controlando por
la correlación intra-seed (mismo seed = mismos shocks). Implementación
vía `statsmodels.MixedLM` con REML.

### ICC (test-retest dentro de modelo)

Con $R$ réplicas por (seed, modelo), el coeficiente de correlación
intraclase ICC(1) [^shrout-fleiss-1979] mide la fracción de varianza
atribuible al seed (sustantivo) vs. al sampler estocástico del LLM:

$$\text{ICC} = \frac{\sigma^2_{\text{seed}}}{\sigma^2_{\text{seed}} + \sigma^2_{\text{residual}}}$$

ICC $\to 1$: las diferencias entre modelos son robustas. ICC $\to 0$: el
sampler domina y los hallazgos son ruido.

### Bootstrap CI95

Para los IC95 de medias, *percentile bootstrap* [^efron-tibshirani] con
$B = 5000$ remuestreos:

$$\text{IC}_{95} = \big[Q_{0.025}(\bar{x}^{(b)}), Q_{0.975}(\bar{x}^{(b)})\big]$$

---

## Análisis bayesiano

Complementan el pipeline frecuentista con dos modelos PyMC en
`guatemala_sim.bayesian` (extra `[bayes]`):

### BEST: Bayesian Estimation Supersedes the t-test [^kruschke-2013]

Para cada métrica pareada $(x_a^{(i)}, x_b^{(i)})$ ajustamos al vector de
diferencias $d_i = x_b^{(i)} - x_a^{(i)}$ un modelo Student-t robusto:

$$d_i \sim \mathrm{StudentT}(\nu, \mu, \sigma)$$

$$\mu \sim \mathcal{N}(0,\ 10 \cdot s_d), \quad \sigma \sim \mathrm{HalfNormal}(5 \cdot s_d), \quad \nu - 1 \sim \mathrm{Exponential}(1/29)$$

donde $s_d$ es el desvío empírico de las diferencias. Devolvemos el
posterior completo de $\mu$, su HDI95, $P(\mu_b > \mu_a \mid \text{datos})$,
$P(|\mu| < \text{ROPE} \mid \text{datos})$ y el tamaño de efecto
estandarizado $\mu / \sigma$ (Cohen's d bayesiano). Más informativo que
el `p`-value del Wilcoxon: distingue *evidencia de equivalencia* (ROPE)
de *ausencia de evidencia*.

```python
from guatemala_sim.bayesian import best_paired
res = best_paired(x_claude, x_openai, metric="pobreza_fin", model_a="Claude", model_b="OpenAI")
# res.posterior_diff_mean, res.hdi95_lo, res.hdi95_hi,
# res.prob_b_gt_a, res.prob_in_rope, res.effect_size_mean
```

### Dirichlet-multinomial jerárquico sobre el presupuesto

Cada turno aporta una asignación presupuestaria
$\mathbf{p}_t \in \Delta^{K-1}$ (K = 9 partidas). Modelamos:

$$\alpha_k \sim \mathrm{HalfNormal}(\sigma=10), \quad k = 1, \ldots, K$$

$$\mathbf{p}_t \mid \boldsymbol\alpha \sim \mathrm{Dirichlet}(\boldsymbol\alpha), \quad t = 1, \ldots, T$$

El posterior de $\boldsymbol\alpha$ revela la **constitución del LLM**
como un prior bayesiano sobre asignaciones. La media esperada por
partida es $\mathbb{E}[p_k] = \alpha_k / \sum_j \alpha_j$ con su HDI95
correspondiente; la suma $\sum_j \alpha_j$ mide la *concentración* —
un LLM "rígido" tiene concentración alta (poca varianza turn-to-turn),
uno "volátil" tiene concentración baja.

```python
from guatemala_sim.bayesian import compare_budget_constitutions
posts = compare_budget_constitutions(df_long, models=["Claude", "OpenAI"])
# posts["Claude"].expected_share — vector (9,) de E[share]
# posts["Claude"].alpha_hdi95   — HDI95 por partida
# posts["Claude"].concentration_mean — rigidez del modelo
```

---

## Datos reales (Banco Mundial)

```bash
python -m guatemala_sim.refresh_data
```

Esto descarga 17 indicadores macro/sociales para Guatemala desde la API
del Banco Mundial [^wb-api] y los persiste como CSV en
`data/world_bank_gtm.csv`. `bootstrap.initial_state_calibrated()` usa el
snapshot para reemplazar 14 de 20 campos del estado inicial con valores
auditables del último año disponible.

### Snapshot actual (último año disponible por indicador)

| campo del state | valor calibrado | año | unidad | fuente |
|---|---:|:---:|---|---|
| `pib_usd_mm` | 113 200 | 2024 | MM USD | WB `NY.GDP.MKTP.CD` |
| `crecimiento_pib` | 3.65 | 2024 | % | WB `NY.GDP.MKTP.KD.ZG` |
| `inflacion` | 2.87 | 2024 | % | WB `FP.CPI.TOTL.ZG` |
| `reservas_usd_mm` | 24 412 | 2024 | MM USD | WB `FI.RES.TOTL.CD` |
| `cuenta_corriente_pib` | 2.89 | 2024 | % PIB | WB `BN.CAB.XOKA.GD.ZS` |
| `remesas_pib` | 19.12 | 2024 | % PIB | WB `BX.TRF.PWKR.DT.GD.ZS` |
| `tipo_cambio` | 7.76 | 2024 | GTQ/USD | WB `PA.NUS.FCRF` |
| `ied_usd_mm` | 1 848 | 2024 | MM USD | WB `BX.KLT.DINV.CD.WD` |
| `poblacion_mm` | 18.41 | 2024 | millones | WB `SP.POP.TOTL` |
| `pobreza_general` | 56.0 | 2023 | % | WB `SI.POV.NAHC` |
| `gini` | 0.452 | 2023 | 0–1 | WB `SI.POV.GINI` |
| `desempleo` | 2.60 | 2025 | % | WB `SL.UEM.TOTL.ZS` |
| `homicidios_100k` | 23.4 | 2023 | tasa | WB `VC.IHR.PSRC.P5` |
| `matricula_primaria` | 86.9 | 2018 | % | WB `SE.PRM.NENR` |

Los 6 campos restantes (`balance_fiscal_pib`, `cobertura_salud`,
`deuda_pib`, `informalidad`, `migracion_neta_miles`, `pobreza_extrema`)
tienen problemas conocidos en la API o no están en WB y quedan en sus
defaults hardcodeados. Ver `data/SOURCES.md` para el detalle y el
procedimiento de update.

Indicadores políticos / perceptuales (aprobación presidencial, libertad
de prensa, alineamientos exteriores) **no vienen del Banco Mundial** y
deben curarse manualmente desde Latinobarómetro [^latinobarometer], LAPOP
[^lapop], RSF [^rsf] etc.

### Verificar la calibración

```python
from guatemala_sim.bootstrap import initial_state_calibrated

state, meta = initial_state_calibrated()
print(f"PIB inicial: USD {state.macro.pib_usd_mm:,.0f} MM")
print(f"Pobreza:     {state.social.pobreza_general:.1f} %")
print(f"Campos calibrados: {len(meta['campos_reemplazados'])}/20")
print(f"Campos en default: {meta['campos_default']}")
```

### Banguat (tipo de cambio diario USD/GTQ)

```bash
python -m guatemala_sim.refresh_data --solo-banguat --dias 365
```

Cliente SOAP artesanal contra
`https://banguat.gob.gt/variables/ws/TipoCambio.asmx`. Operaciones
soportadas: tipo de cambio del día, rango de fechas para USD, y rango
para otras monedas (EUR, MXN, GBP, CAD, JPY, CRC, HNL — códigos en
`MONEDAS_BANGUAT`). El cliente hace retry exponencial frente a los
500 intermitentes del endpoint y persiste a CSV.

```python
from datetime import date, timedelta
from guatemala_sim.banguat_ingest import (
    tipo_cambio_dia, tipo_cambio_rango, latest_tipo_cambio_promedio,
)

obs = tipo_cambio_dia()                          # último día publicado
df = tipo_cambio_rango(date.today() - timedelta(days=30), date.today())
print(f"Promedio últimos 30 días: {latest_tipo_cambio_promedio(df):.4f}")
```

**Limitación honesta**: Banguat sólo expone tipo de cambio vía SOAP
estable. IPC, IMAE, tasa líder, agregados monetarios viven en PDFs y
XLSX del sitio público — para esos lo razonable es bajar el XLSX a
`data/` y agregar un parser dedicado. Cualquier scraper HTML del sitio
se rompe cada pocos meses.

---

## Ejemplos

### 1. Single run sin API (smoke test)

```bash
python demo.py --turnos 4 --seed 42
```

### 2. Comparativa Anthropic vs. OpenAI

```bash
python compare_llms.py \
  --seed 11 --turnos 8 \
  --claude-modelo claude-haiku-4-5-20251001 \
  --openai-modelo gpt-4o-mini
```

### 3. Multi-seed robusto (publicación-ready)

```bash
python compare_llms_multiseed.py \
  --seeds-from 1 --seeds-to 20 \
  --turnos 8 --replicas 3 \
  --continuar-si-falla
```

### 4. Programático con datos reales

```python
from guatemala_sim.bootstrap import initial_state_calibrated
from guatemala_sim.engine import run_turn, DummyDecisionMaker
import numpy as np

state, meta = initial_state_calibrated()
print(f"PIB inicial: USD {state.macro.pib_usd_mm:,.0f} MM "
      f"(calibrado vs WB, {len(meta['campos_reemplazados'])} campos reales)")

rng = np.random.default_rng(42)
dm = DummyDecisionMaker(rng)
for _ in range(8):
    state, _ = run_turn(state, dm, rng)
```

---

## Configuración

Variables de entorno (cargadas desde `.env` si existe):

```ini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Dependencias clave:

| Paquete | Para qué |
|---|---|
| `pydantic >= 2.5` | validación del action schema |
| `numpy`, `pandas`, `matplotlib` | análisis y plots |
| `networkx >= 3.2` | grafo territorial |
| `mesa >= 3.0` | agentes [^mesa] |
| `anthropic >= 0.40` | cliente Anthropic [^anthropic-tools] |
| `scipy >= 1.11` | Wilcoxon, bootstrap, t no-central |
| `statsmodels >= 0.14` | mixed-effects, ICC [^pinheiro-bates] |
| `wbgapi` (extra `[ingest]`) | API Banco Mundial [^wb-api] |
| `requests` (extra `[ingest]`) | cliente SOAP de Banguat |
| `pymc >= 5.10`, `arviz >= 0.17` (extra `[bayes]`) | BEST + Dirichlet-multinomial |

---

## Tests

```bash
python -m pytest                             # 88 tests
python -m pytest tests/test_data_ingest.py   # solo ingesta WB
python -m pytest tests/test_banguat_ingest.py # solo Banguat (mock SOAP)
python -m pytest tests/test_multiseed.py     # análisis frecuentista
python -m pytest tests/test_bayesian.py      # BEST + Dirichlet (lento, ~100s)
```

| Módulo | Tests |
|---|---:|
| `state`, `actions`, validación | 12 |
| `engine`, `agents`, `world` | 13 |
| `comparison`, `plotting` | 4 |
| `president` (Anthropic + OpenAI offline) | 8 |
| `resilient` | 4 |
| **`multiseed`** (Capa 1 + 2 + ICC) | **12** |
| **`data_ingest`** (offline, snapshot mockeado) | **9** |
| **`banguat_ingest`** (offline, SOAP mockeado) | **10** |
| **`bayesian`** (BEST + Dirichlet-multinomial) | **12** |
| `indicators`, `territory` | 4 |

Todos los tests son **offline** (no requieren API ni red).

---

## Estructura del repositorio

```
guatemala-sim/
├── pyproject.toml
├── README.md                          # este archivo
├── guatemala.md                       # spec de diseño extendido
├── demo.py
├── compare_llms.py
├── compare_llms_multiseed.py
├── guatemala_sim/
│   ├── state.py                       # GuatemalaState (Pydantic)
│   ├── actions.py                     # DecisionTurno
│   ├── bootstrap.py                   # estado inicial + calibrado
│   ├── data_ingest.py                 # World Bank → state
│   ├── banguat_ingest.py              # Banguat SOAP (USD/GTQ)
│   ├── bayesian.py                    # BEST + Dirichlet-multinomial
│   ├── refresh_data.py                # CLI de ingesta (WB + Banguat)
│   ├── engine.py                      # run_turn + DummyDecisionMaker
│   ├── president.py                   # ClaudePresidente (Anthropic)
│   ├── president_openai.py            # GPTPresidente (OpenAI compat)
│   ├── resilient.py
│   ├── indicators.py
│   ├── comparison.py
│   ├── multiseed.py                   # IC95, Wilcoxon, mixed-eff, ICC
│   ├── plotting.py
│   ├── logging_.py
│   ├── world/
│   │   ├── macro.py                   # ec. dif. + ruido gaussiano
│   │   ├── shocks.py                  # Bernoulli endógeno
│   │   └── territory.py               # grafo NetworkX
│   └── agents/                        # Mesa v3
├── data/
│   ├── world_bank_gtm.csv             # snapshot WB (auto-generado)
│   ├── banguat_tipo_cambio.csv        # serie diaria USD/GTQ (auto-generado)
│   ├── departamentos.csv
│   ├── adyacencias.csv
│   └── SOURCES.md
├── tests/                             # pytest, 88 tests offline
├── runs/                              # JSONL por corrida (gitignored)
├── figures/                           # PNG + reportes (gitignored)
└── paper/
    └── constituciones_reveladas.md
```

---

## Estado y roadmap

**Hecho:**

- [x] State, action schema, bootstrap calibrado contra WB 2024
- [x] Dinámica macro + shocks endógenos + agentes Mesa
- [x] Grafo territorial (22 departamentos)
- [x] Cliente Anthropic con `tool_use`
- [x] Cliente OpenAI con `json_schema` strict + modo loose
- [x] Indicadores compuestos + métricas constitucionales
- [x] Comparativa single-seed
- [x] Multi-seed robusto: Wilcoxon + Holm + BH-FDR + Cohen's d + Cliff's δ + power
- [x] Mixed-effects sobre datos turn-level
- [x] ICC sobre réplicas (test-retest)
- [x] Ingesta automatizada del Banco Mundial
- [x] Ingesta automatizada de Banguat (tipo de cambio USD/GTQ vía SOAP)
- [x] Análisis bayesiano pareado estilo BEST [^kruschke-2013]
- [x] Modelo jerárquico Dirichlet-multinomial sobre el presupuesto
- [x] Paper draft (`paper/constituciones_reveladas.md`)
- [x] 88 tests offline pasan

**Pendiente / próximos pasos:**

- [ ] Migrar `world/macro.py` a PySD (system dynamics propiamente dicho)
- [ ] Streamlit dashboard interactivo
- [ ] Más decisores: Gemini, DeepSeek, Llama 3.1
- [ ] Ingesta automática de INE / MINFIN (Banguat sólo da tipo de cambio
      vía SOAP estable; IPC/IMAE sólo PDFs)
- [ ] Ablación de prompt: ¿la "constitución" sobrevive paráfrasis?

---

## Contribuir

PRs bienvenidos. Áreas donde ayuda haría diferencia inmediata:

- **Calibración macro**: muchos parámetros en `MacroParams` son guesses
  educados; auditarlos contra Banguat / CEPAL sería oro.
- **Decisores nuevos**: si tu modelo favorito tiene una API
  OpenAI-compatible o Anthropic-compatible, agregar el factory es trivial.
- **Más agentes**: militares, narcotráfico, embajada de EE.UU., iglesia
  evangélica, organizaciones indígenas más finamente — todo encaja en el
  patrón `AgenteBase.reaccionar()`.
- **Tests**: edge cases de schema, recovery de errores transitorios.
- **Ingesta**: Banguat / INE / MINFIN no tienen APIs estables; cualquier
  scraper bien testeado cuenta.

Antes de mandar un PR: `python -m pytest` debe pasar 88/88.

---

## Cita

```bibtex
@misc{guatemala-sim,
  title  = {Constituciones Reveladas: un testbed de gobernanza para
            evaluar LLMs como decisores ejecutivos},
  author = {<tu nombre>},
  year   = {2026},
  note   = {Universidad de San Carlos de Guatemala},
  url    = {<repo url>},
}
```

---

## Referencias

### Macroeconomía y política fiscal

[^mankiw]: Mankiw, N. G. (2019). *Macroeconomics*, 10ª ed. Worth Publishers.
  Tratado estándar para el multiplicador del gasto y la curva IS-LM.

[^ilzetzki-fiscal]: Ilzetzki, E., Mendoza, E. G., & Végh, C. A. (2013).
  "How big (small?) are fiscal multipliers?". *Journal of Monetary Economics*,
  60(2), 239–254. [DOI:10.1016/j.jmoneco.2012.10.011](https://doi.org/10.1016/j.jmoneco.2012.10.011).
  Magnitudes empíricas para economías emergentes ($\mu \in [0.4, 1.2]$).

[^stock-watson]: Stock, J. H., & Watson, M. W. (2007). "Why has U.S.
  inflation become harder to forecast?". *Journal of Money, Credit and Banking*,
  39(s1), 3–33. [DOI:10.1111/j.1538-4616.2007.00014.x](https://doi.org/10.1111/j.1538-4616.2007.00014.x).
  Inercia inflacionaria como AR(1).

[^banguat-policy]: Banco de Guatemala (2025). *Política Monetaria,
  Cambiaria y Crediticia*. <https://www.banguat.gob.gt>. Meta de
  inflación 4 % ± 1, esquema de metas explícitas desde 2005.

[^cottarelli-tax]: Cottarelli, C. (Ed.) (2011). *Revenue Mobilization in
  Developing Countries*. IMF Policy Paper. Elasticidades tributarias
  típicas para América Latina.

[^ravallion-poverty]: Ravallion, M. (2001). "Growth, inequality and
  poverty: looking beyond averages". *World Development*, 29(11), 1803–1815.
  [DOI:10.1016/S0305-750X(01)00072-9](https://doi.org/10.1016/S0305-750X(01)00072-9).
  Origen de la elasticidad pobreza-crecimiento $\approx -0.35$.

### Procesos estocásticos y simulación

[^hawkes]: Hawkes, A. G. (1971). "Spectra of some self-exciting and
  mutually exciting point processes". *Biometrika*, 58(1), 83–90.
  [DOI:10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83).
  Procesos puntuales auto-excitables (referencia conceptual de los shocks
  endógenos).

[^mesa]: Kazil, J., Masad, D., & Crooks, A. (2020). "Utilizing Python for
  agent-based modeling: The Mesa framework". *SBP-BRiMS 2020*.
  [DOI:10.1007/978-3-030-61255-9_30](https://doi.org/10.1007/978-3-030-61255-9_30).

[^bishop-prml]: Bishop, C. M. (2006). *Pattern Recognition and Machine
  Learning*. Springer. Cap. 4: softmax como distribución categórica
  normalizada sobre clases.

[^shannon-1948]: Shannon, C. E. (1948). "A mathematical theory of
  communication". *Bell System Technical Journal*, 27(3), 379–423.
  Entropía $H = -\sum p_i \log p_i$.

### Estadística e inferencia

[^wilcoxon-1945]: Wilcoxon, F. (1945). "Individual comparisons by ranking
  methods". *Biometrics Bulletin*, 1(6), 80–83.
  [DOI:10.2307/3001968](https://doi.org/10.2307/3001968).

[^holm-1979]: Holm, S. (1979). "A simple sequentially rejective multiple
  test procedure". *Scandinavian Journal of Statistics*, 6(2), 65–70.
  [JSTOR:4615733](https://www.jstor.org/stable/4615733).

[^bh-1995]: Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False
  Discovery Rate: a practical and powerful approach to multiple testing".
  *JRSS-B*, 57(1), 289–300.
  [DOI:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).

[^cohen-1988]: Cohen, J. (1988). *Statistical Power Analysis for the
  Behavioral Sciences*, 2ª ed. Routledge. Cohen's d, convenciones de
  magnitud, power analysis.

[^cliff-1993]: Cliff, N. (1993). "Dominance statistics: ordinal analyses
  to answer ordinal questions". *Psychological Bulletin*, 114(3), 494–509.
  [DOI:10.1037/0033-2909.114.3.494](https://doi.org/10.1037/0033-2909.114.3.494).

[^romano-2006]: Romano, J., Kromrey, J. D., Coraggio, J., Skowronek, J.,
  & Devine, L. (2006). "Exploring methods for evaluating group differences
  on the NSSE and other surveys". *FAIR 2006*. Magnitudes de Cliff's δ.

[^laird-ware]: Laird, N. M., & Ware, J. H. (1982). "Random-effects models
  for longitudinal data". *Biometrics*, 38(4), 963–974.
  [DOI:10.2307/2529876](https://doi.org/10.2307/2529876).

[^pinheiro-bates]: Pinheiro, J., & Bates, D. (2000). *Mixed-Effects
  Models in S and S-PLUS*. Springer.
  [DOI:10.1007/b98882](https://doi.org/10.1007/b98882).

[^shrout-fleiss-1979]: Shrout, P. E., & Fleiss, J. L. (1979). "Intraclass
  correlations: uses in assessing rater reliability". *Psychological
  Bulletin*, 86(2), 420–428.
  [DOI:10.1037/0033-2909.86.2.420](https://doi.org/10.1037/0033-2909.86.2.420).

[^efron-tibshirani]: Efron, B., & Tibshirani, R. J. (1993). *An
  Introduction to the Bootstrap*. Chapman & Hall.
  [DOI:10.1201/9780429246593](https://doi.org/10.1201/9780429246593).

[^kruschke-2013]: Kruschke, J. K. (2013). "Bayesian estimation supersedes
  the t test (BEST)". *Journal of Experimental Psychology: General*,
  142(2), 573–603.
  [DOI:10.1037/a0029146](https://doi.org/10.1037/a0029146).

### LLMs como agentes y structured outputs

[^anthropic-tools]: Anthropic (2024). *Tool use with Claude*.
  <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>.

[^openai-structured]: OpenAI (2024). *Structured outputs*.
  <https://platform.openai.com/docs/guides/structured-outputs>.

### Datos

[^wb-api]: World Bank (2025). *World Bank Open Data API*.
  <https://datahelpdesk.worldbank.org/knowledgebase/topics/125589>. CC BY 4.0.

[^latinobarometer]: Corporación Latinobarómetro (annual). *Informe
  Latinobarómetro*. <https://www.latinobarometro.org>.

[^lapop]: LAPOP / Vanderbilt (annual). *AmericasBarometer*.
  <https://www.vanderbilt.edu/lapop/>.

[^rsf]: Reporters Sans Frontières (annual). *World Press Freedom Index*.
  <https://rsf.org/en/index>.

---

## Licencia

[MIT](LICENSE). Hacé lo que quieras con esto, sólo no me culpes si tu
gobierno simulado colapsa.
