# BRCA — Review Response Log

> **Documento operativo**, no narrativa. Para cada cambio aplicado a
> `paper_ieee.tex` durante el reframing a IEEE SaTML / CONESCAPAN, registra
> qué se cambió, dónde, y con qué evidencia.

Fecha: 2026-05-11
Empirical base: batch `20260503_181558_dceacd_multiseed` (N=20 seeds × 2
modelos × 8 turnos) + ablations B1/B2/B3 + R1–R4, R6.

---

## Parte 1 — Reframing IEEE (de `paper/ieee_reframing.md`)

El documento de estrategia `paper/ieee_reframing.md` definió siete movidas:
título Opción B, abstract IEEE-flavor, tres tablas nuevas (standards
mapping, FMEA, reproducibility checklist), reorganización de §2, énfasis
ingenieril en §3, subsecciones por ablación en §5, y movimiento de pruebas
formales a apéndice. Cada movida se aplicó como sigue:

| Movida | Implementación | Locación en `paper_ieee.tex` |
|---|---|---|
| Título Opción B | *Bayesian Revealed Constitution Analysis: An Engineering-Grade Audit Framework for LLMs in Public-Sector Decision Pipelines* | `\title{...}` |
| Abstract IEEE | ~250 palabras; "instrument" 3×; standards mapping en oración 2; cierra con FMEA | `\begin{abstract}...` |
| Tabla 1 — Standards mapping | NIST AI RMF × EU AI Act × ISO/IEC 42001 × IEEE 7000 por capa | `\label{tab:standards}` |
| Tabla 2 — FMEA | Failure mode + detection + mitigation por capa | `\label{tab:fmea}` |
| Tabla 3 — Reproducibility checklist | IEEE/ACM artifact-eval style | `\label{tab:checklist}` |
| §2 reorganizada | Subsecciones (a) eval suites (b) auditing frameworks/standards (c) IRL/IRD (d) faithfulness (e) threat model NIST | §II.A–E |
| §5 subseccionada | §V.A synth, §V.B main, §V.C baselines, §V.D robustness, §V.E RPC, §V.F downstream | §V |
| §3.9 standards mapping | Renombrada "cross-cutting view"; no es capa | `\label{sec:standards}` |
| Prop 1/2 → apéndice | Sólo enunciado + intuición en cuerpo | Appendix~A |
| Threat model → §2.E | Bajo framing NIST SP 800-30 + RMF | `\label{sec:threat}` |
| Reproducibility checklist | Nueva §VII | `\label{sec:repro}` |

**Removido** del draft anterior: referencias a Khipu/LatinX, claims de
"first" en sentido sustantivo amplio, ángulo Global South como
contribución principal (movido a calibración context en §I).

---

## Parte 2 — CONESCAPAN review (2026-05-11)

Veredicto del reviewer: **accept con revisiones**. Issues mapeados a cambios
concretos en el texto:

### Bloqueantes

#### C1 — Anonimato (línea 1, "Author Name") + email identificable

**Status:** cerrado.
**Cambio:** placeholder visible `[Author name to be inserted at submission]`,
email removido. Bloque de autor incluye un comentario LaTeX explicando que
CONESCAPAN es single-blind y una variante double-blind comentada para reuso.
**Locación:** `\author{...}` block (líneas ~49–60).

#### C2 — Referencias huérfanas `[17]` y `[18]`

**Status:** cerrado.
**Cambio:** `\bibitem{anthropic_rsp}` y `\bibitem{deepmind_fsf}` removidos del
bloque `\begin{thebibliography}`.
**Verificación:** grep `\cite{anthropic_rsp\|deepmind_fsf}` no encuentra
ocurrencias en `paper_ieee.tex`.

### Sustantivos

#### S1 — "Frontier LLMs" vs. modelos auditados (Haiku/4o-mini son tier deployment)

**Status:** cerrado por opción (a) — reframing.
**Cambio:** "frontier large language models" → "deployment-tier large language
models" en abstract; nuevo párrafo dedicado en §I explicando por qué auditamos
el tier small/cost-efficient (es lo que LDCs pueden costear a escala
ministerial); flagship-tier listado en Limitations como tier ausente.
**Locación:** abstract (líneas ~60–98); §I párrafo 2 (líneas ~134–147);
Limitations §"Deployment-tier coverage only".

#### S2 — Encoding lexical del CoT amenaza el claim (c)

**Status:** parcialmente cerrado por reformulación; v3/v4 listados como
required follow-ups.
**Cambio:** §V.E renombrada *Reasoning--policy consistency: candidate signal*.
Lenguaje cambiado de "robust screening signal" a "candidate signal pending
non-lexical validation". Confounder de registro lexical (GPT-4o-mini puede
verbalizar más cerca del lexicon de anchor phrases) identificado
explícitamente en el cuerpo. v3 (LLM-as-judge) y v4 (blind human coding)
sobre 5-seed subset documentados como required, no optional. Discusión §VI.A
y Conclusion suavizadas a "candidate cross-model gap... directionally
consistent... pending non-lexical validation".
**Locación:** `\label{sec:rpc-gap}`; `\subsection{What BRCA proved}`; §VIII
Conclusion.
**Acción pendiente:** correr v3 sobre 5 seeds (sub-muestra) antes del envío.

#### S3 — Asimetría enforcement (tool_use + retries vs json_schema strict)

**Status:** cerrado por reconocimiento en Limitations.
**Cambio:** nuevo bullet *Asymmetric structured-output enforcement* en
§Limitations: documenta que Claude vía tool-use con hasta 3 retries vs
GPT-4o-mini vía server-side constrained decoding es una diferencia
no-controlada que puede shape el CoT y por extensión §V.E. Matched-enforcement
experiments listados como follow-up.
**Locación:** §Limitations.

#### S4 — Extracción de θ_stated no documentada en cuerpo

**Status:** cerrado.
**Cambio:** nuevo bloque *Extraction of θ_stated* dentro de §III.E. Documenta:
(a) proyección manual determinista, no LLM-as-judge; (b) prompt verbatim que
se proyecta; (c) reglas de salencia (dominant/secondary/tertiary); (d)
release del MENU\_SYSTEM\_PROMPT en `guatemala_sim/prompts.py`; (e) limitación
de single-coder, segundo-coder inter-rater como follow-up.
**Locación:** §III.E *IRD audit*, bloque "Extraction of θ_stated".

#### S5 — MINFIN 2024 como "human-process anchor" mezcla normativo con político

**Status:** cerrado.
**Cambio:** caveat añadida en §III.H (baselines): B3 es *revealed
institutional-behavior anchor*, no preferencia humana idealizada;
"closeness to B3" se debe leer como "closeness to existing institutional
process". Re-iterado en Limitations §"B3 anchor is institutional, not
normative" y en Conclusion.
**Locación:** §III.H; §Limitations; §Conclusion.

#### S6 — R4 LOO sobre human_development rompe parcialmente

**Status:** cerrado por sentencia explícita de identificabilidad.
**Cambio:** frase añadida al final del bloque R4: *"the recovered
\texttt{anti\_poverty} dimension is identified principally by the contrast
\texttt{human\_dev} vs.\ rest; widening $K$ (live $K{=}7$ or $K{=}9$
collections) is a prerequisite for any claim of independent identification
of the six dimensions in isolation."*
**Locación:** §V.D, bloque R4; reforzado en §Limitations §"Menu sensitivity is partial".

### Menores (M1–M8)

| Item | Status | Locación |
|---|---|---|
| M1 — Eq.\ 1 unidades + origen de 0.20 | cerrado | §III.B, después de Eq.~\ref{eq:gdp}; $0.20 = 2/9$ uniform threshold |
| M2 — Prop 1 estimador (MAP vs posterior mean) | cerrado | Appendix~A; Prop 1 reescrita para incluir ambos |
| M3 — Standards mapping cross-cutting, no layer | cerrado | §III.I renombrada *Standards mapping (cross-cutting view)* |
| M4 — Normalización post-`\|\theta\|`=1 explícita | cerrado | §III.D, bloque "Normalization" |
| M5 — BRCA name vs BRCA1/BRCA2 | cerrado | §I, footnote en `\textbf{BRCA}` |
| M6 — Tabla VI disclaimer in-table | cerrado | `\label{tab:harms}`, caption con "NOT A POLICY FORECAST" en bold |
| M7 — SYSTEM\_PROMPT documentado | cerrado | §IV.A, path a `guatemala_sim/prompts.py::MENU_SYSTEM_PROMPT` |
| M8 — Effect sizes para paired tests | cerrado, ver Parte 3 | §V.B bloque "Effect sizes"; §Limitations §"Twenty seeds" |

### Estratégicas

| Sugerencia | Status |
|---|---|
| Gancho operacional en §I.A | cerrado: nuevo párrafo de apertura con la escena "Latin American ministry of finance evaluating two LLM vendors" |
| Standards mapping prominente en abstract | cerrado: movido a la oración 2 del abstract |
| Figura del pipeline (TikZ) | pendiente; opcional |

---

## Parte 3 — Verificación empírica de effect sizes (M8)

El reviewer pidió Cliff's δ o similar además de p-values para los 13 paired
tests pre-registrados. La primera versión del texto contenía estimaciones
plausibles ("8 large / 2 medium") pero no verificadas. Las cifras se
recomputaron sobre el batch real y el texto se ajustó a los valores
observados.

### Script reproducible

`scripts/compute_effect_sizes.py` (run from repo root):

```
python scripts/compute_effect_sizes.py
```

Inputs (todos versionados):

- `figures/.../audit_per_seed.csv` (cosine_irl, w_norm, chosen_entropy)
- `figures/.../harms_per_seed.csv` (delta_hogares, muertes_anuales, welfare_usd_mm)
- `figures/.../consistency_per_seed.csv` (cosine_cot)
- `figures/.../posteriors_per_seed.csv` (las seis dimensiones)
- `figures/.../tests_pareados.csv` (catálogo de los 13 tests + p-values)

Output: `figures/.../paired_effect_sizes.csv` con columnas
`metric, n_pairs, median_diff_claude_minus_openai, cliffs_delta,
rank_biserial_r_rb, magnitude_cliffs, pvalue_wilcoxon`.

### Definiciones

- **Paired Cliff's δ** (sign-dominance): `δ = (#pos − #neg) / n`, con
  `diff = θ_claude − θ_openai` por seed; ties exactos ignorados.
  Convención de magnitud (Romano et al.):
  `|δ| < 0.147` negligible · `< 0.33` small · `< 0.474` medium · `≥ 0.474` large.
- **Matched-pairs rank-biserial r_rb**: `(W₊ − W₋) / (W₊ + W₋)` donde
  `W₊` / `W₋` son sumas de ranks de las diferencias positivas / negativas.

### Resultados (de `paired_effect_sizes.csv`)

| Metric | $p$ Wilcoxon | Cliff's $\delta$ | $r_{\text{rb}}$ | Magnitud |
|---|---|---|---|---|
| `delta_hogares` | $1.9 \times 10^{-6}$ | $+1.00$ | $+1.00$ | large |
| `welfare_usd_mm` | $1.9 \times 10^{-6}$ | $+1.00$ | $+1.00$ | large |
| `cosine_cot` | $1.9 \times 10^{-6}$ | $-1.00$ | $-1.00$ | large |
| `w_norm` | $5.7 \times 10^{-6}$ | $-0.90$ | $-0.98$ | large |
| `w[anti_pobreza]` | $3.8 \times 10^{-6}$ | $-0.90$ | $-0.99$ | large |
| `w[pro_crecimiento]` | $4.8 \times 10^{-5}$ | $-0.90$ | $-0.92$ | large |
| `muertes_anuales` | $2.9 \times 10^{-4}$ | $+0.85$ | $+1.00$ | large |
| `w[pro_confianza]` | $1.7 \times 10^{-3}$ | $-0.70$ | $-0.76$ | large |
| `w[anti_inflation_dev]` | $8.3 \times 10^{-3}$ | $+0.60$ | $+0.66$ | large |
| `cosine_irl` | $1.7 \times 10^{-2}$ | $-0.60$ | $-0.60$ | large |
| `w[anti_deuda]` | $3.3 \times 10^{-2}$ | $+0.50$ | $+0.54$ | large |
| `chosen_entropy` | $0.24$ | $+0.25$ | $+0.34$ | small |
| `w[pro_aprobacion]` | $0.45$ | $-0.20$ | $-0.20$ | small |

### Hallazgos derivados (todos basados en datos reales)

1. **Las 11 rejections son TODAS "large"** por Cliff's δ.
   Siete con $|\delta| \ge 0.85$, cuatro con $0.5 \le |\delta| < 0.85$.
2. **Las 2 no-rejections son "small"** ($\delta \in \{+0.25, -0.20\}$),
   alineado con la falta de poder y con la interpretación de §V.B sobre la
   insensibilidad mecánica de `chosen_entropy` y la dispersión natural de
   `w_pro_approval`.
3. **`w_pro_crecimiento` matiz**: $|\delta| = 0.90$ (sign-dominance: Claude
   por debajo de GPT-4o-mini en 19/20 seeds) pero la magnitud absoluta del
   median diff ($-0.10$) está cerca de la dispersión posterior por-seed
   ($\pm 0.09$). Es robusto en dirección, modesto en magnitud — el caveat del
   reviewer queda registrado con precisión.

### Cambios en `paper_ieee.tex`

- §V.B *Effect sizes* reescrita con cifras reales:
  *"Of the eleven rejected tests, all eleven satisfy $|\delta_{\text{Cliff}}| \ge 0.5$ (large); seven satisfy $|\delta| \ge 0.85$."*
- §Limitations §*Twenty seeds* alineada: *"All eleven rejections have large
  sign-dominance ($|\delta| \ge 0.5$); we flag $w_{\text{pro\_growth}}$ as a
  direction-robust but small-magnitude disagreement at this seed budget."*
- Referencia al artifact `paired_effect_sizes.csv` añadida en ambos sitios.

---

## Parte 4 — Items abiertos antes del envío

| # | Item | Esfuerzo | Bloqueante para envío |
|---|---|---|---|
| 1 | v3 LLM-as-judge sobre 5 seeds (cierra S2 plenamente) | medio (1 día) | no, pero alto-valor adversarial |
| 2 | Figura del pipeline TikZ (sugerencia estratégica 2) | bajo (medio día) | no |
| 3 | Compilación pdflatex + page-count (target ~10pp SaTML / sin límite duro CONESCAPAN) | trivial (Overleaf) | sí |
| 4 | OSF pre-registration DOI (checklist row "pending → yes") | bajo | depende del venue |
| 5 | Insertar nombre real del autor + decidir blind/no-blind del venue final | trivial | sí |
| 6 | Second-coder θ_stated re-projection (sub-bullet de S4) | medio | no |

---

## Parte 5 — Artefactos creados / modificados en esta sesión

**Nuevos:**

- `scripts/compute_effect_sizes.py` — script reproducible para Cliff's δ y
  rank-biserial r_rb sobre los 13 paired tests.
- `figures/20260503_181558_dceacd_multiseed_irl_multiseed/paired_effect_sizes.csv`
  — output canónico citado desde el paper.
- `paper/REVIEW_RESPONSE.md` — este documento.

**Reescritos / modificados:**

- `paper/paper_ieee.tex` — rewrite completo de Spanish-conference draft
  (~6pp) a English IEEE engineering-instrument paper (~10pp target), con
  todos los cambios del reframing (Parte 1) y los del review (Parte 2 +
  Parte 3).

**Versionado anterior preservado:**

- `paper/paper_ieee_en.tex` — versión canónica anterior intacta;
  `paper/paper_ieee.tex` ahora la supersede para el path IEEE
  SaTML / CONESCAPAN.

---

## Procedencia de las cifras citadas en el paper

Para auditoría posterior, cada cifra-clave del cuerpo proviene de un
artefacto versionado:

| Cifra | Fuente |
|---|---|
| $\hat R{=}1.000$, ESS-bulk min $>4500$ | `audit_per_seed.csv` (cols `rhat_max`, `ess_bulk_min`) |
| Posterior pooled per dimension (Table III) | `posterior_pooled.csv` |
| Median cosines `+0.689` / `+0.725` | `audit_per_seed.csv` mediana sobre 20 seeds |
| Median $L_1$ vs MINFIN `59.1` / `60.6` | `figures/.../b3_anchor/per_seed.csv` |
| Baseline regret y lift | `figures/.../baselines/baselines_per_seed.csv` |
| R6 cosines vs $\sigma{=}1$ ref | `figures/.../r6_prior/cos_to_sigma1.csv` |
| R4 LOO medians | `figures/.../sensitivity/r4_leave_one_out.csv` |
| R2 threshold sweep | `figures/.../sensitivity/r2_threshold_sweep.csv` |
| R3 dual encoding | `figures/.../sensitivity/r3_dual_encoding.csv` |
| Faithfulness multi-encoder | `figures/.../faithfulness_robustness/consistency_multi_encoder.csv` |
| Cliff's $\delta$ y $r_{\text{rb}}$ (Parte 3) | `figures/.../paired_effect_sizes.csv` |
| Harm magnitudes (Table VI) | `harms_per_seed.csv` mediana sobre 20 seeds |
| Synthetic recovery slope $-0.498 \pm 0.014$ | corrida sintética separada, archivada en `figures/irl_recovery/` |

Cualquier reviewer puede reproducir las cifras corriendo
`python scripts/compute_effect_sizes.py` y los scripts de robustness ya
documentados en Appendix~B del paper.

---

## Parte 6 — Claim-scope softening (2026-05-16)

Round adicional de revisión propuso suavizar el lenguaje normativo del
paper para que las afirmaciones queden estrictamente al nivel de
medición (no de intent inference). Aplicado en `paper_ieee.tex`:

### Reemplazos de framing

| Original | Reemplazo | Locación |
|---|---|---|
| "systematic misalignment with the declared deployer intent" | "persistent divergence from the declared deployer intent under the BRCA instrumentation and calibrated menu" / "consistently diverge from the stated reward, with recovered weights outside the pre-registered ROPE" | abstract; §I.A.iii; §V.B; §VI.A; §VIII |
| "recovers and validates the latent reward structure of an LLM" | "estimates a posterior over reward weights from structured choices under a fixed menu and interprets the result as a recovered preference profile conditional on the model, feature basis, and calibration layer" | abstract; §I; §III.D (nuevo párrafo de apertura) |
| "country-agnostic by construction" | "method-agnostic apart from calibration data, simulator, and human-process anchor" | abstract; §I; §VIII. Mantenida la frase "B1 and B2 are country-agnostic" en §III.H porque ahí describe las dos baselines específicas, no el instrumento |
| "Harm quantification" (título §III.F) | "Simulator-translated downstream illustration" | §III.F |
| "Existing AI evaluation suites... do not recover the latent normative preferences..." | "Existing AI evaluation suites are useful for capability, toxicity, and steerability, but do not directly recover the latent reward structure implied by a sequence of structured policy choices" | abstract; §I (escena ministerio) |
| RPC framing: "lexical screening proxy, not a faithfulness verdict" + Lanham cite | Reformulada como "screening signal" con caveat explícito: "the lexical nature of the proxy means it cannot be interpreted as evidence of deceptive alignment in the sense of [hubinger2024]" | §III.G |

### Nuevos bullets en Limitations (§VII)

1. **Discrete-menu and action-space conditionality.** Bayesian IRL
   recovers preferences sólo relativo al action space observado y al
   feature basis; no representa una utility function universal oculta.
   R4 acota menu-sensitivity para *nuestro* menú pero no certifica el
   resultado contra menús estructuralmente distintos.
2. **External validity intentionally limited.** Dos modelos, una
   calibración de país; cross-model y cross-setting generalization es
   open question, no claim del paper.
3. **Simulator-translated metrics are illustrative.** Bullet
   re-escrito (ex "Harm layer is simulator-dependent") para enfatizar
   que las cifras de §V.F son ilustrativas y deben re-estimarse bajo
   la calibración propia del auditor antes de cualquier uso downstream.

### Referencias

Las citas sugeridas por el reviewer (Ramachandran-Amir 2007, Ziebart
2008, Lanham 2023, NIST AI RMF, EU AI Act, IEEE 7000, ISO/IEC 42001)
ya estaban en `\begin{thebibliography}`. Se reforzó la cita explícita
de Ramachandran-Amir / Ziebart / McFadden en el nuevo párrafo de
apertura de §III.D, donde antes el contexto sólo aparecía vía la
ecuación de Boltzmann. No se inventaron bibitems nuevos; las
"best next references to look up" del reviewer son sugerencias
temáticas sin citaciones concretas y se omitieron para no introducir
referencias no verificadas.

### Status

Cerrado en `paper_ieee.tex`. Pendiente recompilar pdflatex para
verificar wrap/overflow después de los cambios en abstract y §VII.

---

## Parte 7 — Ejecución del programa de tuning (2026-05-26)

### Nota de archivo canónico (IMPORTANTE)

Auditando el repo el 2026-05-26 se encontró que **el draft IEEE en
inglés con la estructura §V.A–F, las tablas R1–R6, FMEA y standards
mapping vive en `paper/paper_ieee_en.tex`**, no en `paper_ieee.tex`.
Las Partes 1–6 de este documento hablan de "`paper_ieee.tex`" como el
draft inglés, pero el archivo con ese contenido es `paper_ieee_en.tex`
(71 KB); el `paper_ieee.tex` actual (37 KB) es una versión paralela en
español. Las ediciones de tuning de la Parte 7 se aplicaron a
**`paper_ieee_en.tex`** (el archivo que realmente contiene R6 y θ_stated).
**Decisión del usuario (2026-05-27):** el target de submission es
**`paper/paper_ieee_en.tex`** (inglés). Es el archivo canónico; todas
las ediciones del programa de tuning se aplican ahí. El
`paper/paper_ieee.tex` (español) queda como versión paralela/local, no
se mantiene en sync (candidato a archivar; no se borró sin instrucción
explícita).

### Resultados de los tunings ejecutados

Detalle completo con (a) cifras, (b) regla pre-registrada, (c) veredicto,
(d) revisión implicada en `paper/TUNING_PREREG.md` §RESULTS. Resumen:

| Tuning | Veredicto | Efecto en el paper |
|---|---|---|
| T1.1 prior-σ extendido | ⚠ invalidación de dirección (σ≥5) / veredicto robusto | R6 reescrito: dirección acotada a σ≤2, degradación σ≥5 reportada, misalignment 40/40 robusto en toda la grilla |
| T1.4 perturbación menú | PASS (L2 slope −0.493) | refuerza §V.A identificabilidad |
| T1.5 effect sizes baselines | descriptivo | enriquece Tabla V |
| T2.1 K=7 live | H1/H3 robusto, H2 parcial | confirmatorio §V.D; falta K=7 baselines para cerrar H2 |
| T3.1 segundo coder θ_stated | ⚠ invalidación (κ min −0.20) | nuevo bullet en §VII: θ_stated coder-dependiente; R1 defiende el veredicto binario |

### Cambios aplicados a `paper_ieee_en.tex`

1. **§V R6** — párrafo reescrito con grilla extendida σ∈{0.1…10}; claim
   de dirección acotado a σ≤2; degradación σ≥5 explícita; veredicto
   binario robusto en toda la grilla.
2. **Tabla `tab:robust`** — filas R6 reemplazadas: dirección σ∈[0.1,2]
   (sí), σ=5 y σ=10 (no, fuera de scope), reclasificaciones 0/40 grilla
   completa (sí).
3. **§VII Limitations** — nuevo bullet
   *"θ_stated is a coder-dependent projection"* con κ, cosenos y la
   defensa vía R1.

### Pendiente antes del envío (actualizado)

- Rerun T1.2 (normalización; el dir de salida quedó vacío).
- Correr T1.3 (feature dropout), T2.2 (K=9), T3.2 (v3 judge).
- T2.4 (temperature) corriendo; consolidar análisis IRL al terminar.
- T2.3 (prompt intensity): auditar IRL los batches ya recolectados vs.
  regla pre-registrada.
- T3.3 (B3-2023): **bloqueado por datos** — requiere presupuesto
  ejecutado MINFIN 2023 real + flag `--anchor-year`.
- Holm-Bonferroni sobre la familia p-valor cuando T2.3/T2.4/T3.2/T3.3
  estén completos.
- Recompilar pdflatex y verificar page-count.
