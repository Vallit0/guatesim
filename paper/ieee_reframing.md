# Reframing IEEE — BRCA como Instrumento de Auditoría de Ingeniería

> **Documento de estrategia**, no de paper. Decide venue, framing, título, énfasis
> editorial y secciones a añadir/suavizar para mover el trabajo de "ML-research
> paper" a "engineering-grade audit instrument paper".

Fecha: 2026-05-11 · Estado: borrador estratégico · Base empírica: batch
`20260503_181558_dceacd_multiseed` (N=20 seeds × 2 modelos × 8 turnos) +
ablations B1/B2/B3 + R6 prior sensitivity + faithfulness robustness multi-encoder.

---

## 1. Por qué reframear a IEEE

NeurIPS, AAAI y AIES juzgan **novedad ML + rigor estadístico + generalización**.
IEEE — particularmente SaTML, TAI, Computer y el bloque IEEE Standards — juzga
**instrumentación correcta, reproducibilidad operacional, conformidad con
estándares, despliegue defendible**. Para BRCA esto es ventaja, no concesión:
el repo ya tiene 443 tests, semillas fijas, pre-registros pendientes, y el
threat model mapeado a NIST AI RMF. Es más "instrumento" que "modelo nuevo".

Tradeoff a aceptar: IEEE pesa menos en el ranking de prestigio ML puro, pero
**(a)** acepta el ángulo Global South con menor fricción, **(b)** tolera mejor
los 14–16 páginas del paper actual sin pedir colapsar a 8, **(c)** valora la
ingeniería del repo como contribución de primera clase, **(d)** abre puerta a
citas desde la comunidad de safety standards (ISO/IEC 42001, IEEE 7000 series).

---

## 2. Venue target

### Primary — IEEE SaTML 2027 (Secure and Trustworthy Machine Learning)

- **Por qué fit**: scope explícitamente cubre "auditing", "evaluation of
  deployed ML systems", "trustworthiness verification". BRCA es literalmente
  un audit framework para deployed LLM systems.
- **Deadline esperado**: octubre 2026 (basado en SaTML 2025/2026).
- **Páginas**: ~10 + refs, dos columnas IEEEtran. El draft actual de
  `paper_ieee_en.tex` (14–16pp estimadas) requiere trim moderado.
- **Audiencia**: ML security + safety + adversarial — el frame
  "behavioral auditing" + "revealed constitution" pega.

### Secondary — IEEE Transactions on Artificial Intelligence (TAI)

- Journal, no conferencia. Submission rolling.
- Permite 16–20 pp, ideal para versión completa con apéndices teóricos
  (Definición 1, Definición 2, Prop 1, Prop 2 + pruebas).
- Tiempo de revisión: ~6–9 meses. Combinable con submission paralelo a SaTML
  como "extended journal version" si el conference acepta.

### Tertiary regional — IEEE CLEI o IEEE Latin America Conference (LATINCOM)

- Si el ángulo Global South requiere venue regional para legitimidad.
- Aceptan secciones en español.
- Audiencia LatAm directa pero menor visibilidad internacional.

### Descartar para IEEE

- IEEE S&P / EuroS&P — demasiado security-narrow, BRCA no es exploit research.
- IEEE Big Data — fit débil, BRCA no es Big Data en sentido estricto.
- IEEE Computer magazine — demasiado broad, peor fit técnico que SaTML.

**Recomendación**: SaTML 2027 como primary, TAI como journal companion. Paper
IEEE en draft (`paper_ieee_en.tex`) ya está en formato IEEEtran — la
infraestructura está.

---

## 3. Título: tres opciones a IEEE-flavor

El título actual está duplicado: "A Behavioral Auditing Bayesian Revealed
Constitution Analysis: A Behavioral Auditing Framework..." — "Behavioral
Auditing" aparece dos veces. Hay que colapsarlo. Tres alternativas
IEEE-friendly:

**Opción A (engineering-frame, corto):**
> *BRCA: A Bayesian Behavioral Auditing Instrument for LLM Policy Recommenders*

Ventaja: nombre del método al frente, "Instrument" señala framing IEEE,
8 palabras. Desventaja: pierde el ángulo Global South.

**Opción B (sistema-frame, dual-axis):**
> *Bayesian Revealed Constitution Analysis: An Engineering-Grade Audit
> Framework for LLMs in Public-Sector Decision Pipelines*

Ventaja: "Engineering-Grade" señala IEEE directamente, "Public-Sector Decision
Pipelines" señala el deployment context. Desventaja: 16 palabras, largo.

**Opción C (standards-frame, el más IEEE-tradicional):**
> *Behavioral Auditing of Large Language Models Under Compositional Constraints:
> A Bayesian Instrument Aligned with NIST AI RMF*

Ventaja: invoca un standard explícitamente (NIST AI RMF) — IEEE reviewers
adoran esto. Desventaja: pierde "Revealed Constitution Analysis" del nombre.

**Recomendación**: **Opción B**. Conserva el nombre BRCA, marca framing
ingenieril, y "Public-Sector Decision Pipelines" comunica deployment context
sin saturar con "Global South" en el título (eso queda para abstract).

---

## 4. Abstract IEEE — propuesta (~250 palabras)

> Frontier large language models are increasingly deployed as decision
> support in public-sector budget recommendation, yet existing AI safety
> evaluation suites focus on capabilities and toxicity, not on the latent
> normative preferences these models impose when acting as constrained
> decision-makers. We present **BRCA (Bayesian Revealed Constitution
> Analysis)**, a black-box behavioral auditing instrument that recovers
> and validates the latent reward structure of an LLM acting as executive
> recommender from its choices over a discrete menu of compositional
> budget allocations under aggregate constraint. The instrument integrates
> seven instrumented layers — a calibrated macroeconomic environment, a
> discrete candidate menu, Bayesian Inverse Reinforcement Learning,
> Inverse Reward Design audit, harm quantification under explicit
> simulator assumptions, multi-encoder reasoning–action coherence proxy,
> and a triad of normative baselines — and is delivered as a
> 443-test, version-controlled reproducible pipeline with pre-registered
> analysis. Synthetic identifiability is validated at the empirical
> $1/\sqrt{N}$ convergence rate over six reward dimensions. We
> instantiate the framework on Guatemala as a calibrated case and audit
> Claude Haiku 4.5 versus GPT-4o-mini over twenty seeds of eight
> quarterly turns under identical shocks. Both models exhibit systematic
> misalignment with declared deployer intent across 20/20 seeds; against
> a constrained-optimum oracle they fall strictly within an interpretable
> regret band; against a human-process anchor instantiated as the
> Guatemalan MINFIN 2024 executed budget the ordering reverses, ruling
> out single-axis model rankings. Eleven of thirteen pre-specified
> paired comparisons survive Holm–Bonferroni. The instrument is
> country-agnostic; calibration data, not method, varies across
> deployments. We map each audit layer to NIST AI RMF *Measure*
> functions and discuss residual instrumentation risk.

Notas:
- "Audit instrument" se repite 3 veces — consistente con framing IEEE.
- "443-test, version-controlled reproducible pipeline" — IEEE-reviewer-bait.
- Cierra con NIST AI RMF mapping — señal de standards-awareness.
- Quita "Global South" del front-matter; aparece en §1 Introduction.
- Adopta el lenguaje cauto del draft `paper_ieee_en.tex` ("screening signal,
  not a faithfulness verdict" para el flag deceptive alignment).

---

## 5. Estructura de secciones — diff vs draft actual

El draft `paper_ieee_en.tex` ya está cerca. Cambios para el reframing IEEE:

| Sección actual | Acción | Nueva forma IEEE |
|---|---|---|
| §1 Introduction | **Comprimir** ángulo Global South a 1 párrafo | Líder con "deployment gap" → "auditing instrument needed" |
| §2 Related Work | **Reorganizar** | Subsecciones: (a) LLM evaluation suites (Perez, Santurkar, Durmus, Anthropic Evals), (b) AI auditing frameworks (NIST AI RMF, EU AI Act, AISI, ISO/IEC 42001), (c) IRL/IRD literature, (d) revealed preferences económica como **decorativa** o se quita |
| §3 Method | **Mantener + añadir** | + §3.5 nuevo: "Standards mapping" — tabla cada capa → función NIST RMF + cláusula EU AI Act + IEEE 7000 principle |
| §4 Synthetic validation | **Mantener** verbatim | Es el sweet-spot, ya está bien |
| §5 Empirical study | **Mantener + ampliar** | Subsecciones por ablation: §5.1 main N=20, §5.2 B1/B2/B3, §5.3 R6 prior sensitivity, §5.4 faithfulness multi-encoder |
| §6 Threat model | **Mover a §2.x** | Como subsección de Related/Background con framing NIST |
| §7 Discussion | **Añadir** | + subsección "Failure modes and instrumentation risk" (FMEA-style) |
| (nuevo) §8 | **Añadir** | "Reproducibility checklist" estilo IEEE/ACM artifact evaluation: dataset, env, seeds, dependencies, replication budget |
| §9 Limitations | **Mantener** | Honesto, IEEE valora la honestidad |
| §10 Future work | **Cortar** | Min, IEEE no le interesa "future work" largo |

---

## 6. Énfasis editorial — qué subir, qué bajar

### Subir (IEEE prefiere):
- **Reproducibilidad operacional**: 443 tests, seeds fijas, batch ID
  determinista (`20260503_181558_dceacd_multiseed`), versionado de prompts,
  presupuesto API documentado.
- **Instrumentación**: cada una de las 7 capas como módulo testeable
  independientemente, con interfaces tipadas (Pydantic schemas) e
  invariantes verificables.
- **Standards mapping**: tabla explícita BRCA-layer × NIST-AI-RMF-function ×
  EU-AI-Act-Article × ISO/IEC-42001-clause. Esto NO existe en el draft
  actual y es la diferencia más grande con NeurIPS.
- **Failure modes**: FMEA (Failure Modes and Effects Analysis) sobre cada
  capa. Esto es vocabulario IEEE-clásico.
- **Deployment scenarios**: ¿en qué situaciones reales una agencia
  gubernamental correría este audit antes de delegar a un LLM? Casos
  concretos (presupuesto MINFIN, recomendación de gasto social, asignación
  de transferencias condicionadas).
- **Engineering-grade triple condición de auditoría**: replicabilidad
  (ICC) ∧ sensibilidad (cosine) ∧ especificidad (regret to B1) — todas
  con umbrales pre-registrados.

### Bajar (IEEE no premia tanto):
- Las pruebas formales de identificabilidad (Prop 1, Prop 2): **mover a
  apéndice**, en cuerpo principal solo enunciado + intuición.
- Conexión con revealed preferences económica (Samuelson, Afriat): **o
  formalizar GARP completo o quitar**. Decoración no se acepta en IEEE.
- Generalización cross-country: **mantener pero suavizar**. El frame
  IEEE acepta "we demonstrate on one calibrated case; the instrument
  is by-construction reusable" sin exigir cross-country en el paper.
- Discusión filosófica/programática de "AI Safety from the Global South":
  mover a §10 broader impact, no en intro.

### Quitar:
- Cualquier reclamo de "first" en sentido sustantivo amplio. IEEE pesa
  más "demonstrably correct" que "first".
- Referencias a venues alternativos (Khipu, LatinX in AI) — irrelevantes
  para reviewer IEEE.

---

## 7. Tres tablas nuevas requeridas para IEEE-fit

### Tabla 1 — Standards mapping

| BRCA layer | NIST AI RMF function | EU AI Act article | ISO/IEC 42001 clause | IEEE 7000 principle |
|---|---|---|---|---|
| 1. Calibrated environment | Measure 2.7 (representativeness) | Art. 10 (data quality) | 8.3 (data management) | 5.3 (values traceability) |
| 2. Discrete menu | Measure 2.1 (test design) | Art. 9 (risk management) | 6.1 (risk assessment) | 5.4 (transparency) |
| 3. Bayesian IRL | Measure 2.3 (explainability) | Art. 13 (transparency) | 8.2 (operational planning) | 5.5 (accountability) |
| 4. IRD audit | Measure 2.5 (validity & reliability) | Art. 15 (accuracy) | 6.2 (objectives) | 5.6 (effectiveness) |
| 5. Harm quantification | Measure 2.6 (safety) | Art. 9.2 (harm assessment) | 6.1.3 (risk evaluation) | 5.7 (well-being) |
| 6. Reasoning consistency | Measure 2.8 (fairness/bias) | Art. 13.3 (instructions for use) | 8.4 (operational controls) | 5.8 (data agency) |
| 7. Human-process anchor | Measure 4 (track performance) | Art. 14 (human oversight) | 9.1 (performance evaluation) | 5.9 (effectiveness review) |

Esta tabla es **la pieza más rentable** del reframing IEEE. 0.5 días de trabajo,
señaliza standards-awareness, contesta a priori la pregunta del reviewer "how
does this connect to existing audit frameworks?".

### Tabla 2 — Failure modes (FMEA-style)

| Layer | Failure mode | Detection mechanism | Mitigation in BRCA |
|---|---|---|---|
| 1 | MDP parameters miscalibrated | Backtesting MINFIN 2015–2024, MAPE threshold | Sensitivity sweep ±20% on key params |
| 2 | Menu candidates not spanning policy space | B1 constrained optimum coverage check | 5 canonical candidates pre-registered |
| 3 | NUTS non-convergence (R̂ > 1.01) | Automated R̂ + ESS check | Reject and re-sample with different seed |
| 4 | Audit false positive (intent recovered but spurious) | Synthetic 1/√N recovery | Validated up to N=80 trajectories |
| 5 | Harm magnitudes outside simulator validity | Sensitivity to elasticity assumption | Reported as bounded under instrumentation |
| 6 | Reasoning encoder spurious | Multi-encoder κ check (v1/v2/v3) | Reported as screening signal only |
| 7 | Anchor data outdated | Source date logged in `data/SOURCES.md` | Re-pull on `data/SOURCES.md` change |

### Tabla 3 — Reproducibility checklist (IEEE/ACM artifact eval style)

| Item | Status | Evidence |
|---|---|---|
| Source code public | ✓ | `github.com/Vallit0/guatesim` |
| Dependencies pinned | ✓ | `pyproject.toml`, Python 3.11.9 |
| Random seeds fixed | ✓ | seeds 1–20, batch ID timestamp+hash |
| Test suite | ✓ | 443 tests, `pytest` offline |
| Raw data archived | ✓ | `runs/20260503_181558_dceacd_multiseed/` |
| Compute budget documented | ✓ | ~$25 USD, ~100 min wall-clock |
| Pre-registration | Pending | OSF DOI to be assigned |
| Standards mapping | ✓ | Table 1 |
| Replication script | ✓ | `irl_audit_multiseed.py --batch-dir <...>` |

---

## 8. Lo que ya está bien y NO hay que tocar

- Validación sintética 1/√N — pieza fuerte, IEEE la va a amar.
- Tabla de R6 prior sensitivity (cos 0.99 al cambiar σ entre 0.5 y 2.0,
  0/40 reclassificaciones) — exactamente el tipo de robustness check
  que IEEE espera.
- Tabla B1/B2/B3 — el hallazgo de que el ordering Claude vs GPT-4o-mini
  **se invierte** entre B1 (stated reward) y B3 (human process) es la
  joya empírica del paper. IEEE valora hallazgos no-mono-axiales.
- 443 tests + seeds fijas + replication script — ya estás IEEE-ready
  en reproducibilidad.

---

## 9. Plan de trabajo para alinear a IEEE — 7–10 días

| Día | Tarea | Output |
|---|---|---|
| 1 | Colapsar título, escribir abstract IEEE | Versión nueva del header del `paper_ieee_en.tex` |
| 2 | Construir Tabla 1 (Standards mapping) | Nueva subsección §3.5 |
| 3 | Construir Tabla 2 (FMEA) | Nueva subsección §7.1 |
| 4 | Construir Tabla 3 (Reproducibility checklist) | Nueva sección §8 |
| 5 | Reorganizar Related Work (§2) | §2 reescrita |
| 6 | Mover pruebas Prop 1/2 a apéndice, mantener intuición | §3 trim + Appendix A |
| 7 | Decidir formal-or-out sobre revealed preferences económica | §2.x ajustada |
| 8 | Pasada de coherencia + compilación pdflatex | PDF v1 |
| 9 | Review interno (compañero USAC) | Notas |
| 10 | Edits finales + verificación page count | Submission-ready |

---

## 10. Riesgos del reframing

1. **Cabeza-mitad**: si lo extiendes a 14pp para journal pero el conference
   pide 10pp, mantener dos versiones (extended vs conference) sincronizadas
   es costo continuo. Mitigación: usar `\IEEEcompsoctitleabstractindextext`
   y comments LaTeX para alternar.
2. **Pérdida de framing Global South**: si bajas mucho ese ángulo en intro,
   pierdes la diferenciación. Mitigación: mantenerlo como §1.1 (una página)
   y como motivación del caso Guatemala, no como contribución principal.
3. **Reviewer IEEE pide ablation cross-language**: el draft actual usa solo
   ES + EN en prompts. Si el reviewer pide PT, requiere corrida adicional.
   Mitigación: pre-anunciar en limitations.
4. **SaTML 2027 puede no abrir tracks de audit**: verificar CFP cuando salga
   (~jul-2026). Plan B: TAI journal directo.

---

## 11. Decisión que el usuario tiene que tomar

Tres caminos posibles, deben elegirse antes de invertir los 7–10 días:

| Camino | Esfuerzo | Riesgo | Ganancia |
|---|---|---|---|
| **A. Reframe completo IEEE** (este documento) | 7–10 días | Pierde tracción NeurIPS | Submission SaTML 2027 viable |
| **B. Dual-track**: NeurIPS workshop primero (Sep 2026), IEEE TAI después | 10 sem | Doble trabajo, pero amortizado | Dos publicaciones |
| **C. Skip IEEE**, ir AAAI 2027 main | 0 días extras | Reviewer pool más duro | Prestigio ML mayor |

**Recomendación**: camino B. NeurIPS workshop submission usa el draft actual
con trim a 8pp; IEEE TAI absorbe la versión extendida con las tres tablas
nuevas. Costo extra real ~5 días sobre el plan de NeurIPS workshop, ganancia
es una publicación adicional con audiencia complementaria.
