# guatemala-sim

> **Auditoría bayesiana de LLM-como-decisor en el Sur Global**, calibrada
> contra Guatemala. Mide lo que un LLM frontera *prefiere* cuando elige
> bajo restricción presupuestaria — no lo que dice que prefiere.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#licencia)
[![Tests](https://img.shields.io/badge/tests-443%20passed-success.svg)](#tests)
[![Method](https://img.shields.io/badge/method-BRCA%20%E2%80%94%20Bayesian%20Revealed%20Constitution%20Analysis-blueviolet.svg)](#las-7-capas-del-instrumento)
[![Empirical](https://img.shields.io/badge/empirical-N%3D20%20seeds%20%C3%97%202%20modelos%20%2B%204%20ablations-brightgreen.svg)](#hallazgos-empíricos-n20-seeds--ablations)

---

## Hallazgos empíricos (N=20 seeds + ablations)

Auditoría completa Claude Haiku 4.5 vs GPT-4o-mini, 8 turnos × 20 seeds en
menu-mode. Tests pareados Wilcoxon signed-rank seed-emparejados. **11/13
métricas significativas a α=0.05**, **9/13 sobreviven Holm–Bonferroni**, **7/13
a p<0.001**.

### El hallazgo central: no hay ranking de un solo eje

Contra dos anchors independientes la dirección del orden Claude vs GPT-4o-mini
**se invierte**. Eso es el resultado más importante del trabajo:

| Anchor | Quién está más cerca | Wilcoxon pareado |
|---|---|---:|
| **B1 — Constrained-optimum vs intent declarado** | GPT-4o-mini (regret +0.62) vs Claude (regret +1.83) | **p<0.0001** |
| **B3 — MINFIN 2024 (proceso humano real)** | Claude (L1=59.1 pp) vs GPT-4o-mini (L1=60.6 pp) | **p=0.0003** |

GPT-4o-mini se alinea más con la función objetivo declarada del deployer;
Claude se alinea más con el proceso presupuestario humano observado. **No hay
"mejor modelo" en sentido absoluto** — depende del normativo contra el que se
audite. Esto es exactamente lo que un instrumento de auditoría debe revelar.

### Constituciones reveladas (posterior IRL pooled, N=20)

| Dimensión | Claude (mean ± std) | GPT-4o-mini (mean ± std) | Wilcoxon p |
|---|---:|---:|---:|
| anti_pobreza | +1.14 ± 0.41 | +1.90 ± 0.24 | <0.0001 \*\*\* |
| anti_deuda | +0.03 ± 0.04 | −0.00 ± 0.05 | 0.033 \* |
| pro_aprobacion | −0.16 ± 0.29 | −0.07 ± 0.32 | 0.452 (ns) |
| pro_crecimiento | −0.01 ± 0.09 | +0.11 ± 0.09 | <0.0001 \*\*\* |
| anti_desviacion_inflacion | +0.07 ± 0.09 | +0.00 ± 0.06 | 0.008 \*\* |
| pro_confianza | −0.00 ± 0.03 | +0.03 ± 0.03 | 0.002 \*\* |

GPT-4o-mini prioriza más fuerte la reducción de pobreza y el crecimiento;
Claude prioriza más la estabilidad de precios. **Ambos modelos tienen
`anti_deuda ≈ 0`** — contradiciendo la afirmación intuitiva basada en
budgets free-form.

### Cuantificación de daño (bajo supuestos del simulador)

| Métrica (mediana sobre 20 seeds) | Claude | GPT-4o-mini | Δ Wilcoxon | p |
|---|---:|---:|---:|---:|
| Δhogares pobreza | −411,187 | −473,420 | +65,300 | <0.0001 \*\*\* |
| Muertes evitadas/año | −5,979 | −8,681 | +2,700 | 0.0003 \*\*\* |
| Welfare USD M | −6,495 | −7,478 | +1,030 | <0.0001 \*\*\* |

Reemplazar GPT-4o-mini por Claude en un pipeline guatemalteco implica, en
mediana sobre 20 escenarios alternativos: ~65,000 hogares menos sacados de
pobreza y ~2,700 muertes adicionales por año respecto del baseline GPT —
**dentro de los supuestos del simulador, no como predicción del mundo real**.

### Reasoning consistency — reportado como señal de screening, no veredicto

| Modelo | flag deceptive v1 (lexical) | flag deceptive v2 (TF-IDF) | flag deceptive v3 (embeddings) |
|---|---:|---:|---:|
| Claude Haiku 4.5 | 7/20 | 4/20 | 12/20 |
| GPT-4o-mini | 0/20 | 0/20 | 19/20 |

La señal cross-modelo (Claude > GPT en flag-rate) **persiste en v1 y v2** pero
**colapsa o se invierte en v3**. Cohen's κ entre encoders: 0.49 (v1↔v2),
0.53 (v1↔v3) para Claude; ≈ 0 para GPT (v3 hipersensitivo). **Lo reportamos
como signal de screening**, no como veredicto de "deceptive alignment". El
método sin validación multi-encoder sobre-claimea.

---

## Ablations adicionales

### B1/B2 — Baselines normativos sintéticos

Para responder "¿qué tan mal/bien está el LLM en términos absolutos?":

- **B1 (constrained optimum)**: el menú-candidate que **maximiza** `θ_stated·φ(s,a)`
  acumulado sobre los 8 turnos. Cota superior sintética bajo intent declarado.
- **B2 (random valid)**: sampling uniforme sobre candidates válidos, mismo
  horizonte. Cota inferior.

| Modelo | mediana LLM score | regret a B1 | agreement con B1 |
|---|---:|---:|---:|
| Claude | +2.51 | +1.84 | 38% turnos |
| GPT-4o-mini | +3.62 | +0.73 | 75% turnos |

Ambos modelos están **estrictamente entre B1 y B2** (p<0.0001 vs ambos),
con GPT-4o-mini significativamente más cerca de B1 (regret gap +1.19,
p<0.0001).

Reporte: [`figures/20260503_181558_dceacd_multiseed_baselines/`](figures/20260503_181558_dceacd_multiseed_baselines/baselines_summary.md)

### B3 — Anchor humano (MINFIN 2024)

Comparación contra el presupuesto ejecutado 2024 de Guatemala (ICEFI Tablas
7+8, datos primarios SICOIN). Codificado a las mismas 9 partidas del menú.

| Modelo | L1 mediana vs MINFIN | cosine mediana vs MINFIN |
|---|---:|---:|
| Claude | 59.1 pp | 0.791 |
| GPT-4o-mini | 60.6 pp | 0.767 |
| (referencia: candidate `status_quo_uniforme`) | 52.9 pp | — |
| (referencia: candidate `seguridad_primero`) | 78.8 pp | — |

Wilcoxon pareado: Claude **más cerca** del proceso humano (p=0.0003 en L1,
p=0.0003 en cosine). Las trayectorias LLM caen dentro del rango de los
candidates canónicos sintéticos.

Reporte: [`figures/20260503_181558_dceacd_multiseed_b3_anchor/`](figures/20260503_181558_dceacd_multiseed_b3_anchor/summary.md)

### R6 — Sensibilidad al prior σ del IRL bayesiano

Re-fit NUTS sobre 40 pares (seed, modelo) variando `prior_sigma ∈ {0.5, 1, 2}`.
La configuración main es σ=1.

| modelo | σ | cosine mediano a σ=1 | min cosine | reclasificación de misalignment |
|---|---:|---:|---:|---:|
| Claude | 0.5 | 0.998 | 0.978 | 0/40 |
| Claude | 2.0 | 0.984 | 0.907 | 0/40 |
| GPT-4o-mini | 0.5 | 0.999 | 0.998 | 0/40 |
| GPT-4o-mini | 2.0 | 0.996 | 0.991 | 0/40 |

**La dirección recuperada es prior-robusta**: cosine 0.98+ entre cualquier par
de σ, 0/40 cambios en la clasificación binaria misaligned vs aligned. El norm
del vector escala con σ (esperado), la dirección no.

Reporte: [`figures/20260503_181558_dceacd_multiseed_r6_prior/`](figures/20260503_181558_dceacd_multiseed_r6_prior/summary.md)

### Faithfulness — validez convergente multi-encoder

Tres encoders ortogonales evalúan el flag de "razonamiento incoherente con
política revelada":
- **v1**: keyword counting (diccionario manual)
- **v2**: TF-IDF sobre anchor phrases (vocab disjoint de v1)
- **v3**: sentence embeddings (semántico, multilingual mock)

Reporte: [`figures/20260503_181558_dceacd_multiseed_faithfulness_robustness/`](figures/20260503_181558_dceacd_multiseed_faithfulness_robustness/faithfulness_robustness.md)

> El flag deceptive alignment es **convergente entre v1 y v2** (κ moderado,
> 0.49–0.53) pero **divergente con v3**. Conclusión metodológica reportada en
> el paper: el flag se trata como screening signal, **no** como verdict.

---

## La pregunta que motiva el proyecto

Los LLMs frontera son entrenados con datos y feedback humano mayoritariamente
del Norte Global: pre-training anglo, RLHF con raters estadounidenses,
*Constitutional AI* y *Frontier Safety Frameworks* redactados en California
y Londres. En 2024–2026 hay reportes públicos de gobiernos latinoamericanos
desplegándolos como soporte de decisión presupuestaria.

> Cuando un LLM frontera entrenado en el Norte Global se delega como decisor
> en una economía del Sur Global, **¿en qué dirección se desvían sus
> recomendaciones respecto de las prioridades del país de despliegue?** Si
> reemplazás Claude Haiku por GPT-4o-mini en un pipeline ministerial,
> *¿cuántos hogares cambian de lado de la línea de pobreza?*

90 años de tradición económica dicen que no hace falta abrirle el cráneo al
agente: alcanza con mirar lo que elige bajo restricción.

- Samuelson (1938). *A Note on the Pure Theory of Consumer's Behaviour*.
- Ng & Russell (2000). *Algorithms for Inverse Reinforcement Learning*.
- Hadfield-Menell et al. (2017). *Inverse Reward Design*.

Este proyecto aplica esa cadena al caso del LLM-as-policymaker bajo el nombre
formal **BRCA (Bayesian Revealed Constitution Analysis)**: el system prompt
es la "recompensa proxy" del deployer, las elecciones del LLM son trayectorias
observadas, y recuperamos bayesianamente la "constitución implícita" del
modelo. Tres anchors normativos independientes (B1 stated-reward optimum, B2
random, B3 human process) dan triangulación en lugar de un solo eje de juicio.

---

## Las 7 capas del instrumento

```
              ┌────────────────────────────────────┐
              │  Capa 7 — Baseline humano (MINFIN) │
              └─────────────────┬──────────────────┘
                                │ anchor B3
 ┌──────────────────────────────▼──────────────────────────────┐
 │              Capa 1 — Mundo simulado (Guatemala)             │
 │  PIB, fiscal, social, político, externo + Mesa + 22 deptos  │
 │  CALIBRADO vs Banco Mundial 2024 + Banguat 2026             │
 └────────────────────────────────┬────────────────────────────┘
                                  │ contexto + shocks
 ┌────────────────────────────────▼────────────────────────────┐
 │       Capa 2 — Menú discreto de elección (5 candidatos)     │
 │  status_quo · fiscal_prudente · desarrollo_humano ·         │
 │  seguridad_primero · equilibrado                            │
 └────────────────────────────────┬────────────────────────────┘
                                  │ LLM elige UNO + razona
 ┌────────────────────────────────▼────────────────────────────┐
 │  Capa 3 — Bayesian IRL (NUTS, PyMC) → posterior w∈ℝ⁶        │
 │  Validado en sintéticos: error ~ 1/√N exacto                │
 │  R6 robustness: prior-invariante (cos>0.98 sobre σ∈{0.5,2}) │
 └─────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
 ┌────────────────────────────┐   ┌──────────────────────────────┐
 │ Capa 4 — IRD audit         │   │ Capa 6 — Reasoning consistency│
 │ alignment gap entre        │   │ multi-encoder (v1/v2/v3)      │
 │ w_recovered y w_stated     │   │ reportado como screening      │
 └────────────┬───────────────┘   └──────────────────────────────┘
              │
              ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Capa 5 — Cuantificación de daño en unidades humanas         │
 │  hogares · niños · muertes/año · welfare USD                 │
 │  (bajo supuestos del simulador, no predicción del mundo real)│
 └──────────────────────────────────────────────────────────────┘

         + Baselines normativos: B1 (constrained optimum) · B2 (random)
```

| Capa | Módulo | Estado |
|---|---|---|
| 1. Mundo simulado | `guatemala_sim/world/` + `engine.py` | 🟢 Validado, 14/20 campos calibrados WB 2024 |
| 2. Menú discreto | `guatemala_sim/irl/candidates.py` | 🟢 Tests Pydantic estrictos |
| 3. IRL bayesiano | `guatemala_sim/irl/bayesian_irl.py` | 🟢 Validado sintético + N=20 reales + R6 prior-robusto |
| 4. IRD audit | `guatemala_sim/irl/audit.py` | 🟢 N=20: 20/20 misaligned vs intent declarado |
| 5. Harm quantification | `guatemala_sim/harms.py` | 🟢 N=20: harms significativos p<0.001 |
| 6. Reasoning consistency | `guatemala_sim/reasoning_consistency.py` + v3 | 🟢 Multi-encoder v1/v2/v3, κ reportado |
| 7. Baseline MINFIN (B3) | `guatemala_sim/minfin_ingest.py` | 🟢 Validado vs ICEFI Tabla 7+8 (SICOIN) |
| + Baselines B1/B2 | derivados del menú | 🟢 Constrained-optimum + random sintéticos |

Detalle conceptual y matemático completo en
[`paper/metodologia.md`](paper/metodologia.md). Versión IEEE en
[`paper/paper_ieee_en.tex`](paper/paper_ieee_en.tex). Estrategia de venue IEEE
en [`paper/ieee_reframing.md`](paper/ieee_reframing.md).

---

## Quick start

### Instalación

```bash
git clone https://github.com/Vallit0/guatesim.git
cd guatesim
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -e .                                     # vía pyproject.toml
# Crear .env con ANTHROPIC_API_KEY y OPENAI_API_KEY (no commitear)
```

### Validación local sin API

```bash
python -m pytest                      # 443 tests pasan offline
python irl_recovery_curve.py          # validación sintética 1/√N
python minfin_baseline_plot.py        # baseline MINFIN vs candidatos
```

### Una corrida real (1 modelo, 8 turnos, ~$1)

```bash
python irl_audit_real_run.py --skip-openai --turnos 8 --seed 11
# → figures/<run_id>_irl_audit/Claude/{posterior,audit,harms,consistency}.csv
# → figures/<run_id>_irl_audit/reporte.md
```

### Reproducir el estudio N=20 (~$25, ~100 min)

```bash
# fase 1: corridas API (Claude + OpenAI, 20 seeds × 8 turnos × 2 modelos)
python compare_llms_multiseed.py \
    --seeds-from 1 --seeds-to 20 \
    --turnos 8 --menu-mode --continuar-si-falla

# fase 2: audit IRL bayesiano sobre todos los jsonls (~15 min, sin API)
python irl_audit_multiseed.py --batch-dir runs/<batch_id>_multiseed/

# fase 3: ablations (B1/B2 baselines, B3 anchor, R6 prior, faithfulness)
python irl_baselines.py --batch-dir runs/<batch_id>_multiseed/
python irl_b3_human_anchor.py --batch-dir runs/<batch_id>_multiseed/
python irl_r6_prior_sweep.py --batch-dir runs/<batch_id>_multiseed/
python reasoning_consistency_v3.py --batch-dir runs/<batch_id>_multiseed/
```

Outputs:
- `runs/<batch>_multiseed/seed{NNN}_{claude,openai}.jsonl` — datos crudos
- `figures/<batch>_multiseed_analysis/` — outcomes (PIB, presupuesto, etc.)
- `figures/<batch>_multiseed_irl_multiseed/` — tablas main del paper
- `figures/<batch>_multiseed_baselines/` — B1/B2 ablation
- `figures/<batch>_multiseed_b3_anchor/` — B3 anchor MINFIN
- `figures/<batch>_multiseed_r6_prior/` — R6 prior sensitivity
- `figures/<batch>_multiseed_faithfulness_robustness/` — multi-encoder κ

---

## Reproducir los resultados publicados

El estudio reportado arriba corresponde al batch
**`20260503_181558_dceacd_multiseed`** (incluido en este repo). Para regenerar
las tablas main sin gastar API:

```bash
python irl_audit_multiseed.py \
    --batch-dir runs/20260503_181558_dceacd_multiseed/
```

Las tablas finales están en
`figures/20260503_181558_dceacd_multiseed_irl_multiseed/reporte_multiseed.md`
y los reportes de las 4 ablations en los directorios homónimos listados arriba.

---

## Por qué LatAm, por qué Guatemala

Tres razones por las que LatAm es contexto AI Safety crítico, no incidental:

1. **Espacio fiscal restringido** (servicio de deuda significativo, reservas
   finitas) → costo marginal de un fallo de alineamiento más alto.
2. **Desigualdad como issue político dominante** — no el crecimiento como
   en economías avanzadas — cambia las prioridades sustantivas.
3. **Capacidad de oversight humano variable** comprime el gradiente "consulta
   humana → delegación", aumentando la urgencia de auditorías ex-ante.

Guatemala es el primer caso porque (i) macro relativamente simple y bien
documentada, (ii) tensiones reales y en curso (deportaciones, sequía,
corrupción), (iii) heterogeneidad territorial (22 deptos, ~40% pob. indígena),
(iv) datos públicos accesibles sin barreras institucionales (Banguat SOAP,
MINFIN, INE ENCOVI, World Bank).

El método es **country-agnostic**: cada país adicional es un nuevo *dataset*
de calibración, no una nueva metodología.

---

## Threat model

`paper/threat_model.md` formaliza el riesgo siguiendo NIST SP 800-30 adaptado
a LLM-as-policymaker:

| Componente | Para este caso |
|---|---|
| Actor (deployer) | Agencia gubernamental que usa LLM para recomendación presupuestaria |
| Stated objective | $w_{\text{stated}}$ — función objetivo declarada del deployer |
| Realized objective | $w_{\text{LLM}}$ — pesos implícitos del LLM (latentes) |
| Misalignment | $\|w_{\text{LLM}} - w_{\text{stated}}\| > \delta_{\text{tolerable}}$ |
| Harm pathway | Recomendaciones desviadas → ejecución desviada → outcomes desviados |
| Mitigation | BRCA ex-ante (las 7 capas + 3 anchors B1/B2/B3) que detecta el desalineamiento ANTES del despliegue |

Mapeado a frameworks de safety institucional: **NIST AI RMF** (fase *Measure*
de "value alignment risk"), **EU AI Act** Art. 9–15 (high-risk obligations),
**ISO/IEC 42001:2023** (AI management systems), **IEEE 7000** series (ethics
in system design), **Anthropic RSP** (2024), **DeepMind FSF**, **UK AISI
Inspect** (categoría "Allocation and Resource Distribution").

---

## Estructura del repo

```
guatemala-sim/
├── guatemala_sim/
│   ├── world/             # Capa 1: dinámica del simulador
│   ├── irl/
│   │   ├── candidates.py  # Capa 2: menú discreto
│   │   ├── bayesian_irl.py # Capa 3: NUTS + PyMC
│   │   ├── hierarchical_bayesian_irl.py # Capa 3 jerárquica (WIP)
│   │   ├── audit.py       # Capa 4: IRD audit
│   │   ├── audit_sensitivity.py # R6 prior sweep
│   │   ├── posterior_analysis.py
│   │   ├── run_parser.py  # JSONL menu-mode → ParsedRun
│   │   └── ...
│   ├── harms.py           # Capa 5: harm quantification
│   ├── reasoning_consistency.py     # Capa 6 v1: keyword counting
│   ├── reasoning_consistency_v3.py  # Capa 6 v3: embeddings
│   ├── convergent_validity.py       # κ inter-encoder
│   ├── faithfulness_benchmark.py    # batch multi-encoder
│   ├── minfin_ingest.py   # Capa 7: baseline MINFIN
│   ├── country_profile.py
│   ├── models_registry.py # registro de modelos frontier
│   ├── multiseed.py       # análisis multi-seed (outcomes)
│   ├── president.py       # cliente Claude (tool_use)
│   └── president_openai.py # cliente OpenAI (json_schema strict)
├── compare_llms.py            # corrida 1-shot single-seed
├── compare_llms_multiseed.py  # batch multi-seed
├── irl_audit_real_run.py      # audit IRL single-run (capas 4-7)
├── irl_audit_multiseed.py     # audit IRL multi-seed con tests pareados
├── irl_baselines.py           # B1/B2 baselines
├── irl_b3_human_anchor.py     # B3 anchor MINFIN
├── irl_r6_prior_sweep.py      # R6 prior sensitivity
├── reasoning_consistency_v3.py # encoder v3 batch
├── irl_recovery_curve.py      # validación 1/√N sintética
├── minfin_baseline_plot.py    # baseline humano vs candidatos
├── demo_hierarchical_real.py  # demo modelo jerárquico
├── paper/
│   ├── metodologia.md         # método completo (LaTeX-ready)
│   ├── paper_ieee.tex         # versión IEEE conference (es)
│   ├── paper_ieee_en.tex      # versión IEEE conference (en) — canónica
│   ├── ieee_reframing.md      # estrategia venue IEEE
│   ├── threat_model.md
│   ├── constituciones_reveladas.md
│   └── README.md
├── runs/                      # JSONL de cada corrida
├── figures/                   # CSVs y reportes generados
├── tests/                     # 443 tests offline
└── data/                      # WB 2024, MINFIN/ICEFI, prompts
```

---

## Tests

```bash
python -m pytest                # 443 passed offline
python -m pytest -k irl         # tests del IRL bayesiano + ablations
python -m pytest -k menu_mode   # flujo menu-mode end-to-end
python -m pytest -k hierarchical # IRL jerárquico (WIP)
python -m pytest -k consistency # encoders v1/v2/v3 + κ
```

---

## Limitaciones

1. **Solo dos modelos frontera evaluados**: Claude Haiku 4.5 vs GPT-4o-mini.
   Falta Gemini, Llama 3.3, DeepSeek-V3, Claude Opus 4.7, GPT-4o full para
   test cross-vendor de la hipótesis $H_{TC}$ (transfer cultural Norte→Sur).
2. **Horizonte corto**: 8 turnos ≈ 2 años trimestrales; no medimos efectos
   compuestos de largo plazo.
3. **Memoria presidencial limitada**: cada llamada al LLM es independiente,
   sin carry-over de mensajes; la única memoria cross-turno está en
   `state.memoria_presidencial` serializado.
4. **MDP no backtesteado contra MINFIN 2015–2024**: backtesting cuantitativo
   es bloqueante identificado y pendiente — ver
   [`paper/ieee_reframing.md`](paper/ieee_reframing.md) y el plan de
   ejecución.
5. **Reasoning consistency NO es verdict de faithfulness**: el flag deceptive
   alignment no sobrevive validación multi-encoder limpiamente (v3 vs v1/v2).
   Se reporta como **screening signal**, no como detección causal de
   engaño.
6. **MINFIN baseline 2024 únicamente**: anchor B3 usa un solo año. La
   extensión a serie 2015–2024 está pendiente.
7. **Snapshot MINFIN aproximado** en datos secundarios además de SICOIN
   primario; verificar consolidación oficial para publicación final.
8. **Pre-registro pendiente**: las hipótesis y umbrales aplicados son
   defendibles pero no fueron registrados en OSF antes de las corridas.
   Próxima corrida (R7 con modelos adicionales) será post-pre-registro.

---

## Citas

```bibtex
@misc{brca-guatemala-sim-2026,
  title  = {Bayesian Revealed Constitution Analysis: A Behavioral Auditing
            Framework for LLMs in Public-Sector Decision Pipelines,
            Calibrated against Guatemala},
  author = {Estudiante USAC},
  year   = {2026},
  note   = {Universidad de San Carlos de Guatemala},
  url    = {https://github.com/Vallit0/guatesim},
}
```

Bibliografía completa (~30 referencias entre Samuelson 1938 y Hadfield-Menell
2017, métodos estadísticos, frameworks AI Safety institucional, AI policy
LatAm) en [`paper/paper_ieee_en.tex`](paper/paper_ieee_en.tex).

---

## Licencia

[MIT](LICENSE).
