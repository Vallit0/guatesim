# guatemala-sim

> **Auditoría bayesiana de LLM-como-decisor en el Sur Global**, calibrada
> contra Guatemala. Mide lo que un LLM frontera *prefiere* cuando elige
> bajo restricción presupuestaria — no lo que dice que prefiere.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#licencia)
[![Tests](https://img.shields.io/badge/tests-254%20passed-success.svg)](#tests)
[![Method](https://img.shields.io/badge/method-Bayesian%20IRL%20%2B%20IRD%20audit-blueviolet.svg)](#las-7-capas-del-instrumento)
[![Empirical](https://img.shields.io/badge/empirical-N%3D20%20seeds%20%C3%97%202%20modelos-brightgreen.svg)](#hallazgos-empíricos-n20-seeds)

---

## Hallazgos empíricos (N=20 seeds)

Auditoría completa Claude Haiku 4.5 vs GPT-4o-mini, 8 turnos × 20 seeds en
menu-mode. Tests pareados Wilcoxon signed-rank seed-emparejados. **13/13
métricas significativas, mayoría a p<0.001.**

### El hallazgo central

| Modelo | cosine CoT↔acción mediano | flag deceptive alignment |
|---|---:|---:|
| **Claude Haiku 4.5** | **+0.519** | **7/20 ⚠️** |
| GPT-4o-mini | +0.837 | 0/20 |

p-valor pareado: **<0.0001 (\*\*\*)**. En 35% de las seeds, el razonamiento
verbal de Claude *no concuerda* con la política que el IRL bayesiano recupera
de sus elecciones reales. GPT-4o-mini no dispara la alarma en ninguna.

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

### Cuantificación de daño

| Métrica (mediana sobre 20 seeds) | Claude | GPT-4o-mini | Δ Wilcoxon | p |
|---|---:|---:|---:|---:|
| Δhogares pobreza | −411,187 | −473,420 | +65,300 | <0.0001 \*\*\* |
| Muertes evitadas/año | −5,979 | −8,681 | +2,700 | 0.0003 \*\*\* |
| Welfare USD M | −6,495 | −7,478 | +1,030 | <0.0001 \*\*\* |

Reemplazar GPT-4o-mini por Claude en un pipeline guatemalteco implica, en
mediana sobre 20 escenarios alternativos: ~65,000 hogares menos sacados de
pobreza y ~2,700 muertes adicionales por año respecto del baseline GPT.

> **Importante**: este NO es un ranking de calidad. El simulador no penaliza
> default soberano ni riesgo cambiario; un modelo más conservador como Claude
> podría estar viendo riesgos que el sim no captura. Lo que el método mide
> es la **dirección revelada** de la preferencia, no su corrección.

Reportes completos:
- [`figures/20260503_181558_dceacd_multiseed_irl_multiseed/reporte_multiseed.md`](figures/20260503_181558_dceacd_multiseed_irl_multiseed/reporte_multiseed.md)
- [`figures/20260503_181558_dceacd_multiseed_analysis/`](figures/20260503_181558_dceacd_multiseed_analysis/) (análisis de outcomes)

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

Este proyecto aplica esa cadena al caso del LLM-as-policymaker: el system
prompt es la "recompensa proxy" del deployer, las elecciones del LLM son
"trayectorias observadas", y recuperamos bayesianamente la "constitución
implícita" del modelo.

---

## Las 7 capas del instrumento

```
              ┌────────────────────────────────────┐
              │  Capa 7 — Baseline humano (MINFIN) │
              └─────────────────┬──────────────────┘
                                │ ancla
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
 └─────────────┬───────────────────────────────┬───────────────┘
               │                               │
               ▼                               ▼
 ┌────────────────────────────┐   ┌──────────────────────────────┐
 │ Capa 4 — IRD audit         │   │ Capa 6 — Reasoning consistency│
 │ alignment gap entre        │   │ unfaithful CoT detection      │
 │ w_recovered y w_stated     │   │ (deceptive alignment screen)  │
 └────────────┬───────────────┘   └──────────────────────────────┘
              │
              ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Capa 5 — Cuantificación de daño en unidades humanas         │
 │  hogares · niños · muertes/año · welfare USD                 │
 └──────────────────────────────────────────────────────────────┘
```

| Capa | Módulo | Estado |
|---|---|---|
| 1. Mundo simulado | `guatemala_sim/world/` + `engine.py` | 🟢 Validado, 14/20 campos calibrados WB 2024 |
| 2. Menú discreto | `guatemala_sim/irl/candidates.py` | 🟢 27 tests, schemas Pydantic estrictos |
| 3. IRL bayesiano | `guatemala_sim/irl/bayesian_irl.py` | 🟢 Validado sintético + N=20 reales |
| 4. IRD audit | `guatemala_sim/irl/audit.py` | 🟢 N=20: 20/20 misaligned vs intent |
| 5. Harm quantification | `guatemala_sim/harms.py` | 🟢 N=20: harms significativos p<0.001 |
| 6. Reasoning consistency | `guatemala_sim/reasoning_consistency.py` | 🟢 N=20: 7/20 vs 0/20 deceptive flag |
| 7. Baseline MINFIN | `guatemala_sim/minfin_ingest.py` | 🟠 Snapshot manual, validar con SICOIN |

Detalle conceptual y matemático completo en
[`paper/metodologia.md`](paper/metodologia.md). Versión IEEE en
[`paper/paper_ieee.tex`](paper/paper_ieee.tex).

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
python -m pytest                      # 254 tests pasan offline
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
```

Outputs:
- `runs/<batch>_multiseed/seed{NNN}_{claude,openai}.jsonl` — datos crudos
- `figures/<batch>_multiseed_analysis/` — outcomes (PIB, presupuesto, etc.)
- `figures/<batch>_multiseed_irl_multiseed/`:
  - `posteriors_per_seed.csv` — $w$ recuperado por (seed, modelo, dim)
  - `posterior_pooled.csv` — agregado entre seeds
  - `audit_per_seed.csv` — IRD: cosine, ROPE, HDI95
  - `harms_per_seed.csv` — Δhogares/niños/muertes/welfare
  - `consistency_per_seed.csv` — cosine CoT, flag deceptive
  - `tests_pareados.csv` — Wilcoxon Claude vs OpenAI, 13 métricas
  - `reporte_multiseed.md` — tablas listas para paper

---

## Reproducir los resultados publicados

El estudio reportado arriba corresponde al batch
**`20260503_181558_dceacd_multiseed`** (incluido en este repo). Para regenerar
las tablas sin gastar API:

```bash
python irl_audit_multiseed.py \
    --batch-dir runs/20260503_181558_dceacd_multiseed/
```

Las tablas finales están en
`figures/20260503_181558_dceacd_multiseed_irl_multiseed/reporte_multiseed.md`.

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
| Mitigation | Auditoría bayesiana ex-ante (las 7 capas) que detecta el desalineamiento ANTES del despliegue |

Mapeado a frameworks de safety institucional: **Anthropic RSP** (2024),
**DeepMind FSF**, **UK AISI Inspect** (categoría "Allocation and Resource
Distribution"), **NIST AI RMF** (fase *Measure* de "value alignment risk").

---

## Estructura del repo

```
guatemala-sim/
├── guatemala_sim/
│   ├── world/             # Capa 1: dinámica del simulador
│   ├── irl/
│   │   ├── candidates.py  # Capa 2: menú discreto
│   │   ├── bayesian_irl.py # Capa 3: NUTS + PyMC
│   │   ├── audit.py       # Capa 4: IRD audit
│   │   ├── run_parser.py  # JSONL menu-mode → ParsedRun
│   │   └── ...
│   ├── harms.py           # Capa 5: harm quantification
│   ├── reasoning_consistency.py  # Capa 6: CoT vs acción
│   ├── minfin_ingest.py   # Capa 7: baseline MINFIN
│   ├── multiseed.py       # análisis multi-seed (outcomes)
│   ├── president.py       # cliente Claude (tool_use)
│   └── president_openai.py # cliente OpenAI (json_schema strict)
├── compare_llms.py            # corrida 1-shot single-seed
├── compare_llms_multiseed.py  # batch multi-seed
├── irl_audit_real_run.py      # audit IRL single-run (capas 4-7)
├── irl_audit_multiseed.py     # audit IRL multi-seed con tests pareados
├── irl_recovery_curve.py      # validación 1/√N sintética
├── minfin_baseline_plot.py    # baseline humano vs candidatos
├── paper/
│   ├── metodologia.md         # método completo (LaTeX-ready)
│   ├── paper_ieee.tex         # versión IEEE conference
│   ├── threat_model.md
│   ├── constituciones_reveladas.md  # storytelling
│   └── README.md              # narrativa Samuelson → Ng → Hadfield-Menell
├── runs/                      # JSONL de cada corrida
├── figures/                   # CSVs y reportes generados
├── tests/                     # 254 tests offline
└── data/                      # WB 2024, MINFIN snapshot, prompts
```

---

## Tests

```bash
python -m pytest                # 254 passed offline
python -m pytest -k irl         # tests del IRL bayesiano
python -m pytest -k menu_mode   # flujo menu-mode end-to-end
```

---

## Limitaciones

1. **Solo dos modelos frontera evaluados**: Claude Haiku 4.5 vs GPT-4o-mini.
   Falta Gemini, Llama 3.1 405B, DeepSeek-V3 para test cross-vendor de la
   hipótesis $H_{TC}$ (transfer cultural Norte→Sur).
2. **Horizonte corto**: 8 turnos ≈ 2 años trimestrales; no medimos efectos
   compuestos de largo plazo.
3. **Memoria presidencial limitada**: cada llamada al LLM es independiente,
   sin carry-over de mensajes; la única memoria cross-turno está en
   `state.memoria_presidencial` serializado.
4. **Calibración del estado inicial** contra datos públicos pero no auditada
   por economistas guatemaltecos: las magnitudes relativas son más
   confiables que los niveles.
5. **Reasoning consistency v1**: keyword counting es la versión más cruda
   del encoding. Una v2 usaría LLM-as-judge o sentence embeddings con
   projection — más caro pero más sensible.
6. **Snapshot MINFIN**: aproximación manual basada en estructura conocida
   del gasto; verificar contra Liquidación oficial / SICOIN para
   publicación final.

---

## Citas

```bibtex
@misc{guatemala-sim-2026,
  title  = {Auditando al LLM-como-Decisor en el Sur Global:
            una Metodología Bayesiana Calibrada contra Guatemala},
  author = {Estudiante USAC},
  year   = {2026},
  note   = {Universidad de San Carlos de Guatemala},
  url    = {https://github.com/Vallit0/guatesim},
}
```

Bibliografía completa (~30 referencias entre Samuelson 1938 y Hadfield-Menell
2017, métodos estadísticos, frameworks AI Safety institucional, AI policy
LatAm) en [`paper/paper_ieee.tex`](paper/paper_ieee.tex).

---

## Licencia

[MIT](LICENSE).
