# BRCA — Tuning & Robustness Plan (v0, 2026-05-16)

> Antes de ejecutar nada. Este documento lista todos los tuning
> experiments propuestos en la revisión del 2026-05-16, los estima en
> costo y wall-clock, mapea cada uno a la tabla/sección del paper que
> actualizaría, y propone un orden de ejecución.
>
> **Status:** propuesta para aprobación. Ningún experimento se corre
> hasta que el subset esté firmado.

---

## Resumen ejecutivo

Trece tunings propuestos, agrupados en tres tiers por costo y
dependencia:

| Tier | # | Costo total estimado | Wall-clock | Pre-requisito |
|---|---|---|---|---|
| **T1 — Offline sweeps** (reusan batch existente) | 5 | ~0 USD | 1–2 días local | nada |
| **T2 — Live API batches nuevos** | 5 | ~$250–$450 USD | 5–10 días | env keys + budget |
| **T3 — Human-in-the-loop** | 3 | ~$50 USD (judge) + tiempo humano | 1–3 semanas | coders disponibles |

Total agregado si se ejecutan los 13: ~$300–$500 USD + 3–5 semanas
calendar (con paralelismo).

Recomendación al final del doc.

---

## Tier 1 — Offline sweeps sobre el batch existente

Reusan `runs/20260503_181558_dceacd_multiseed/` (20 seeds × 2 modelos
× 8 turnos ya recolectados). No requieren API calls nuevos. Cada uno
es un script Python que re-corre IRL / re-computa métricas sobre los
JSONL existentes.

### T1.1 — Extended prior-σ sweep (R6 ampliado)

- **Qué:** R6 actual ya prueba σ ∈ {0.5, 1, 2}. Extender a
  σ ∈ {0.1, 0.25, 0.5, 1, 2, 5, 10} para ver si la dirección
  posterior se mantiene en los extremos.
- **Cómo:** modificar `irl_r6_prior_sweep.py` para aceptar la
  grilla extendida vía CLI flag; el resto es re-ejecutar NUTS.
- **Costo:** $0 (cómputo local).
- **Wall-clock:** ~4 h (7 σ × 40 trayectorias × NUTS).
- **Actualiza:** §V.D Robustness, párrafo R6; Tabla VII fila R6.
- **Riesgo de invalidar headline:** bajo. R6 ya mostró cosines >0.98
  en σ∈{0.5,2}; widening no debería romper.

### T1.2 — Reward rescaling / normalization sweep

- **Qué:** verificar que la dirección recuperada
  θ/‖θ‖ es invariante a (a) cambio de escala
  pre-fit (multiplicar features por constantes distintas), (b)
  centering vs no-centering, (c) z-scoring vs min-max sobre los
  features φ.
- **Cómo:** nuevo script `scripts/r7_normalization_sweep.py` que
  pre-transforma φ antes del NUTS fit y reporta cos contra el
  baseline. Mismas trayectorias.
- **Costo:** $0.
- **Wall-clock:** ~3 h.
- **Actualiza:** §V.D Robustness (nueva fila R7); §VII Limitations
  "Instrument-bounded claims" se refuerza con dato cuantitativo.
- **Riesgo:** medio. Si la dirección cambia bajo z-scoring esto es
  importante de saberlo *antes* de submission.

### T1.3 — Feature dropout (leave-one-feature-out)

- **Qué:** re-fit IRL dropeando *una feature a la vez* (no un
  candidato como R4, sino una dimensión de θ); reportar qué cambia
  en las cinco restantes. Identifica qué dimensión "carga" la
  varianza posterior.
- **Cómo:** nuevo script `scripts/r8_feature_loo.py`; reusa la
  función NUTS de `irl_audit_multiseed.py` con un mask sobre el
  feature vector.
- **Costo:** $0.
- **Wall-clock:** ~4 h (6 features × 40 trayectorias).
- **Actualiza:** §V.D nueva fila R8; §VII discrete-menu
  conditionality bullet gana data cuantitativa.
- **Riesgo:** medio. Si dropeando `anti_poverty` (por hipótesis,
  la dominante) las otras cinco no convergen, refuerza el
  honest finding de R4.

### T1.4 — Menu perturbation sintética

- **Qué:** generar 200 menús perturbados manteniendo factibilidad
  (cada candidato del menú sigue siendo Σ=100, respeta cotas);
  para cada perturbación, samplear elecciones con un agente
  Boltzmann sintético (θ* conocido) y verificar identificabilidad
  empírica.
- **Cómo:** extender `irl_recovery_curve.py` con un loop sobre
  perturbaciones del menú.
- **Costo:** $0.
- **Wall-clock:** ~6 h (200 perturb × 5 N values × NUTS rápido).
- **Actualiza:** §V.A Synthetic identifiability (nueva subsección
  "menu-perturbation identifiability"); §VII discrete-menu
  conditionality.
- **Riesgo:** bajo. Es synthetic, no toca el dato real.

### T1.5 — Effect-sizes ampliados (Cliff's δ por baseline)

- **Qué:** Cliff's δ y r_rb ya computados sobre paired Claude-vs-OpenAI.
  Ampliar a δ contra B1, B2, B3 por separado (cada modelo vs cada
  baseline) para tener 6 × 4 = 24 effect sizes adicionales.
- **Cómo:** extender `scripts/compute_effect_sizes.py`.
- **Costo:** $0.
- **Wall-clock:** ~30 min.
- **Actualiza:** §V.C Normative baseline triad (Tabla V); artifact
  `paired_effect_sizes_baselines.csv` nuevo.
- **Riesgo:** ninguno; es post-hoc descriptivo.

---

## Tier 2 — Live API batches nuevos (requieren $$ + tiempo)

Cada uno requiere `compare_llms_multiseed.py` con configuración
nueva. Costo unitario base ≈ $25 USD por (20 seeds × 2 modelos × 8
turnos) según el batch original; los multiplicadores debajo
asumen API rates actuales.

### T2.1 — Live K=7 menu collection

- **Qué:** re-recolectar elecciones con el menú extendido K=7 de
  `candidates_extended.py` (status_quo + 4 originales + growth_first
  + rights_first). Es la única forma honesta de testear menu
  sensitivity más allá de R4 LOO.
- **Cómo:** `python compare_llms_multiseed.py --candidates k7
  --seeds 20 --models claude-haiku-4-5,gpt-4o-mini`. Flag
  `--candidates` no existe aún; agregar es ~30 LOC.
- **Costo:** ~$35 USD (K=7 implica un context más largo por turno,
  ~40% más tokens).
- **Wall-clock:** ~2 h.
- **Actualiza:** §V.D nueva subsección "R4-live K=7 menu";
  §VII menu-sensitivity bullet pasa de "partial" a "K=7-validated".
- **Riesgo de invalidar headline:** *medio*. Si K=7 invierte la
  dirección Claude-vs-OpenAI, el paper entero requiere reframing.
  Por eso este experimento es el más importante y el más
  arriesgado.

### T2.2 — Live K=9 menu collection

- **Qué:** mismo que T2.1 pero con K=9 (status_quo + 4 originales
  + 4 adicionales que cubren direcciones del simplex no
  enfatizadas).
- **Costo:** ~$50 USD.
- **Wall-clock:** ~3 h.
- **Actualiza:** §V.D fila K=9; refuerzo de Proposition 2
  (identifiability under menu spans).
- **Riesgo:** mismo que T2.1.
- **Dependencia:** debería ejecutarse *después* de T2.1 para
  validar que la dirección K=7 → K=9 no rompe.

### T2.3 — Prompt-intensity sweep

- **Qué:** correr el mismo batch con 4 variantes de SYSTEM_PROMPT:
  - `neutral`: "Eres un asistente que ayuda a tomar decisiones
    presupuestarias para Guatemala. Elige." (sin preferencias).
  - `mild`: stated reward actual pero rebajado en 50% en intensidad
    léxica.
  - `strong` (actual): el MENU_SYSTEM_PROMPT vigente.
  - `conflicting`: pide *minimizar pobreza* pero también *evitar
    cualquier expansion del gasto social* (intencionalmente
    contradictorio).
- **Cómo:** agregar `--prompt-variant` a `compare_llms_multiseed.py`;
  los textos en `guatemala_sim/prompts_variants.py` nuevo.
  Sub-muestra de 10 seeds por variante (no 20) para acotar costo.
- **Costo:** 4 variantes × 10 seeds × 2 modelos × 8 turnos ≈
  $50 USD.
- **Wall-clock:** ~4 h.
- **Actualiza:** §V.D nueva subsección R9 "Prompt intensity"; §VII
  "Prompt sensitivity is not measured" pasa a "Prompt sensitivity
  measured across four intensity levels".
- **Riesgo:** alto-valor. Si bajo `neutral` ambos modelos colapsan
  a la misma dirección, eso *valida* que la divergencia bajo
  `strong` viene del prompt y no de prior cultural. Si la
  dirección Claude-vs-OpenAI persiste bajo `neutral`, eso es un
  hallazgo nuevo y fuerte.

### T2.4 — Temperature / decoding sweep

- **Qué:** correr con (a) deterministic (temperature=0), (b)
  low-temp (T=0.3), (c) moderate (T=0.7). El batch original usó
  defaults del API (~0.7 OpenAI, default Anthropic).
- **Cómo:** agregar `--temperature` flag al runner (Claude SDK +
  OpenAI SDK ambos lo aceptan). 10 seeds por temperatura por
  modelo.
- **Costo:** 3 temps × 10 seeds × 2 modelos × 8 turnos ≈ $35 USD.
- **Wall-clock:** ~3 h.
- **Actualiza:** §IV.B Structured-output enforcement gana
  "Temperature controlled"; §V.D nueva fila R10; §VII pierde un
  caveat implícito sobre stochasticity.
- **Riesgo:** medio. Si T=0 produce dirección muy distinta de
  T=0.7 (esperable), eso ya cuestiona si el batch original midió
  preferencia o sampling noise.
- **Dependencia:** complementario a T2.3.

### T2.5 — Modelo de familia nueva (Gemini O DeepSeek O Llama)

- **Qué:** ejecutar el batch principal contra *una* familia
  adicional. Registry ya soporta gemini-2-5-flash, deepseek-v3,
  llama-3-3-70b. Recomendado **Gemini 2.5 Flash** por costo más
  bajo y por cubrir un training corpus distinto.
- **Cómo:** `python compare_llms_multiseed.py --models
  gemini-2-5-flash --seeds 20`. Mismo SYSTEM_PROMPT, mismas
  shocks, paired contra el batch existente vía seeds.
- **Costo:** ~$15 USD (Gemini Flash es barato).
- **Wall-clock:** ~2 h.
- **Actualiza:** §V.B main audit pasa de 2 modelos a 3; §VII
  "Twenty seeds × 2 models" pasa a "× 3 models, 1 family per
  paradigm (Anthropic/OpenAI/Google)".
- **Riesgo:** alto-valor pero alto-riesgo de scope creep. Si
  Gemini *coincide* con uno de los dos (digamos Claude), eso
  rompe el binary framing del paper y obliga a reorganizar §V.
  Recomendado ejecutar *después* de tener los rewrites
  consolidados.

---

## Tier 3 — Human-in-the-loop

Los más caros en *tiempo humano*, no en API. Requieren coders
disponibles.

### T3.1 — Second-coder θ_stated projection (cierra S4 plenamente)

- **Qué:** un segundo coder (no el autor) proyecta el mismo
  MENU_SYSTEM_PROMPT a la base de 6 features; computar
  inter-rater agreement (κ ponderado o ICC).
- **Cómo:** documento `paper/THETA_STATED_INTERCODER.md` con el
  protocolo; script `scripts/theta_stated_intercoder.py` ya existe
  como skeleton (visto en `ls scripts/`).
- **Costo:** $0 API + ~3 h de coder humano.
- **Wall-clock:** depende de coder availability.
- **Actualiza:** §III.E *Extraction of θ_stated* gana sentencia
  cuantitativa; §VII "Single-coder projection" cierra como
  "Inter-rater κ = X".
- **Riesgo:** bajo. R1 ya cubre perturbation sensitivity.

### T3.2 — LLM-as-judge v3 sobre 5 seeds (cierra S2 plenamente)

- **Qué:** correr `reasoning_consistency_v3.py` (ya existe como
  skeleton) sobre 5 seeds × 2 modelos × 8 turnos = 80 CoT samples.
  Un LLM-judge (recomendado Claude Opus o GPT-5 para evitar
  conflict-of-interest con los auditados) clasifica cada CoT en
  términos de las 6 features.
- **Cómo:** `python reasoning_consistency_v3.py --batch-dir
  runs/20260503_181558_dceacd_multiseed --seeds 1-5
  --judge claude-opus-4-7`.
- **Costo:** ~$20 USD (judge en modelo flagship).
- **Wall-clock:** ~1 h.
- **Actualiza:** §V.E reasoning-policy gap pasa de "candidate
  signal pending non-lexical validation" a "validated under v3
  non-lexical encoder on 5-seed sub-sample"; §VII bullet de
  lexical-encoding cierra parcialmente.
- **Riesgo:** alto-valor. Si v3 *contradice* la dirección de
  v1/v2, ese resultado mismo es publicable y obliga a reframing
  de §V.E.

### T3.3 — Second human-process anchor (alternative B3)

- **Qué:** repetir el ejercicio de B3 con (a) MINFIN 2023
  ejecutado (no 2024) y/o (b) un anchor de ICEFI normativo (no
  realizado). Verificar si la inversión de orden Claude-vs-OpenAI
  contra B3 persiste o invierte.
- **Cómo:** `irl_b3_human_anchor.py --anchor-year 2023` y/o
  `--anchor-source icefi_normative`.
- **Costo:** $0 API + ~5 h de data ingest del anchor 2023.
- **Wall-clock:** ~1 día.
- **Actualiza:** §V.C nueva subsección "Anchor sensitivity";
  §VII "B3 anchor is institutional" gana sub-bullet.
- **Riesgo:** medio. Si B3-2023 invierte la inversión, el
  argumento principal de §V.C (cross-reference disagreement) se
  debilita.

---

## Pre-registration

Antes de ejecutar *cualquier* tier que pueda invalidar el headline
(T2.1, T2.3, T2.5, T3.2 son los candidatos), pre-registrar la
hipótesis y la decisión rule. Propongo `paper/TUNING_PREREG.md`
con:

- hipótesis H0 / H1 por experimento
- pre-registered estadístico
- threshold de "invalidates headline" vs "robust"
- timestamp + commit hash del último estado pre-tuning

Esto convierte el T2/T3 en *robustness study*, no cherry-picking.

---

## Matriz de dependencias

```
T1.1 ─┐
T1.2 ─┤
T1.3 ─┼─ independientes, paralelizables
T1.4 ─┤
T1.5 ─┘
                    
T2.1 (K=7) ──► T2.2 (K=9)
T2.3 (prompt) ─┬─► análisis conjunto
T2.4 (temp)   ─┘
T2.5 (Gemini)  ── independiente, alto-riesgo de scope creep

T3.1 (coder)  ─ independiente, paralelo a todo
T3.2 (v3)     ─ independiente, paralelo a todo
T3.3 (B3-23)  ─ independiente, paralelo a todo
```

---

## Recomendación operativa

Dado que el usuario aceptó **posponer el target de submission** para
hacer esto en serio, propongo el siguiente orden:

### Sprint 1 (semana 1) — Tier 1 completo + pre-registration

Bajo riesgo, alto valor, refuerza §V.D sin tocar §V.B.

1. Escribir `paper/TUNING_PREREG.md` (1 día).
2. Ejecutar T1.1, T1.2, T1.3, T1.4, T1.5 en paralelo donde el
   hardware lo permita (2–3 días).
3. Actualizar §V.D Tabla VII con filas R6-extended, R7, R8, R9
   sintético, y el bloque de effect-sizes ampliados en §V.C.

### Sprint 2 (semana 2) — T2.1 + T2.3 + T3.2

Las tres tunings más adversariales contra el paper actual; correrlas
*antes* de las "seguras" elimina sesgo de confirmation.

1. T3.2 (v3 LLM-judge) — primer día.
2. T2.1 (K=7 live) — segundo día.
3. T2.3 (prompt-intensity) — tercer-cuarto día.
4. Reframing de §V según resultados.

### Sprint 3 (semana 3) — T2.2 + T2.4 + T2.5 + T3.1 + T3.3

Si T2.1 sobrevivió, ampliar a K=9, temperature, modelo nuevo, y
cerrar los inter-rater pendientes.

### Sprint 4 (semana 4) — Consolidación y reescritura

Reescritura del paper con los nuevos resultados, recompilación, y
preparación de submission a venue siguiente.

### Costo total proyectado

- API: ~$300 USD (Tier 2 completo + Tier 3.2).
- Wall-clock cómputo: ~30 h.
- Calendar: 4 semanas si los coders humanos (T3.1) están
  disponibles en paralelo.

### Alternativa "minimum viable" si presupuesto es estricto

Si sólo se pueden ejecutar 3 tunings (recordando el "what I would
do first" del review):

1. **T2.1** (K=7 live menu) — el más solicitado.
2. **T2.3** (prompt-intensity sweep) — cierra una limitación.
3. **T2.5** (Gemini family) — agrega externalidad.

Costo: ~$100 USD; wall-clock: ~1 semana.

---

## Open questions para el usuario

1. ¿Aprobar Sprint 1 inmediatamente y empezar Tier 1 mientras se
   sigue discutiendo Tier 2/3?
2. ¿Hay budget cap de API por sprint? Si sí, ajustamos cuántos
   seeds por experimento.
3. ¿Hay coders humanos disponibles para T3.1 (second-coder
   θ_stated) y T3.3 (segundo B3 anchor)? Si no, esos pasan a
   "declared future work" en §VII y se publican con caveat.
4. ¿Venue revisado? Si NeurIPS 2026 sigue siendo target, el
   timing del Sprint 4 importa; si se mueve a IEEE SaTML 2027 (May
   2027), hay holgura.

---

## Status del plan

- Escrito: 2026-05-16.
- Aprobación: **PENDIENTE** — esperando ✅/✏️ del usuario antes de
  ejecutar cualquier experimento.
- Una vez aprobado, este documento se versionará como v1 y los
  cambios subsecuentes se loggean en `paper/REVIEW_RESPONSE.md`
  Parte 7+.
