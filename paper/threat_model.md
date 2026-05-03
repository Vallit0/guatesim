# Threat Model: LLM-as-Policymaker en Asignación Presupuestaria

> Documento complementario a *Constituciones Reveladas* — formaliza la
> preocupación de seguridad que motiva el método de auditoría
> bayesiano.

---

## 1. Forma del threat model

Adoptamos la estructura canónica de threat models en seguridad
informática (NIST SP 800-30, *Guide for Conducting Risk Assessments*,
2012) adaptada al contexto de despliegue de LLMs en política pública:

| Componente | Pregunta | Respuesta para este caso |
|---|---|---|
| **Actor (deployer)** | ¿Quién despliega el sistema? | Agencia gubernamental / órgano técnico / consultora pública |
| **Capability (sistema)** | ¿Qué capacidad técnica se delega? | Recomendación o ejecución de asignación presupuestaria entre $K \geq 5$ partidas bajo restricción agregada (suma 100 %) |
| **Stated objective (intent)** | ¿Qué dice el deployer querer optimizar? | "Bienestar agregado", "ODS", "reducción de pobreza", o función formal en doc oficial |
| **Realized objective (latent)** | ¿Qué optimiza efectivamente el sistema? | $w_{\text{LLM}} \in \mathbb{R}^d$ — pesos sobre dimensiones de bienestar implícitos en el entrenamiento (RLHF, Constitutional AI, datos de pre-training) |
| **Misalignment** | ¿Cómo divergen las dos? | $w_{\text{LLM}} \neq w_{\text{stated}}$ por pre-training sobre corpus no representativo del país de despliegue, o por feedback humano de población no-target |
| **Harm pathway** | ¿Cómo se materializa el daño? | Recomendaciones sistemáticamente desviadas → ejecución presupuestaria desviada → outcomes desviados (más pobreza, menos cobertura, etc.) |
| **Harm units** | ¿En qué unidades se mide? | Hogares bajo línea de pobreza, niños fuera de escuela, muertes evitables, USD de welfare loss (ver `guatemala_sim/harms.py`) |
| **Threshold** | ¿Cuándo es "preocupante"? | Differential harm vs baseline humano $\geq$ umbral X (a definir por el deployer; en este paper reportamos rangos sin umbral) |
| **Mitigation** | ¿Qué hace nuestro método? | Auditoría bayesiana ex-ante (`fit_bayesian_irl`) que recupera $\hat{w}$ y su HDI95 antes del despliegue, permitiendo comparación con $w_{\text{stated}}$ |

---

## 2. Escenario concreto de despliegue

> Un Ministerio de Planificación contrata a una consultora para
> automatizar parte del análisis presupuestario. La consultora propone
> usar un LLM frontera con un *system prompt* que incluya la función
> objetivo declarada del Ministerio (ej. "priorizar reducción de
> pobreza extrema y cobertura de servicios básicos"). El LLM produce
> recomendaciones que el equipo técnico revisa y eleva al despacho.

Este escenario **no es hipotético**: en 2024–2026 hay reportes públicos
de uso de LLMs en análisis de política pública en varios gobiernos
latinoamericanos (Argentina, Brasil, Chile, Colombia, México). Lo que
no está estandarizado es la auditoría previa a su uso.

### 2.1. Por qué los métodos existentes no detectan este harm

| Método existente | Por qué no alcanza |
|---|---|
| **Benchmark academic (MMLU, BIG-bench)** | Mide capacidad general, no preferencias en dominio específico |
| **Red-teaming manual** | No detecta sesgos sistemáticos sutiles que solo emergen en agregado |
| **Stated-preference probing (Perez et al. 2022)** | El LLM responde lo que el evaluador quiere oír (sycophancy) |
| **Constitutional AI training** (Bai et al. 2022) | Entrena valores generales, no auditables ex-post para casos específicos |
| **RLHF reward models** | Optimizan preferencias del rater, no de la población de despliegue |

Nuestra contribución llena un hueco específico: **inferencia bayesiana
de preferencias reveladas en dominio cuantitativo bajo restricción
agregada**, con incertidumbre cuantificada (HDI95) y validación
sintética de identificabilidad.

---

## 3. Threat model formal

Sea:
- $\mathcal{A}(s)$ el menú discreto de $K$ asignaciones presupuestarias en estado $s$
- $\phi(s, a) \in \mathbb{R}^d$ vector de outcomes esperados por asignación
- $w_{\text{stated}} \in \mathbb{R}^d$ los pesos de la función objetivo declarada del deployer
- $w_{\text{LLM}} \in \mathbb{R}^d$ los pesos implícitos del LLM, no observables directamente

El LLM samplea acciones desde:
$$\pi_{\text{LLM}}(a \mid s) \propto \exp(w_{\text{LLM}}^\top \phi(s, a))$$

El threat es:

$$\boxed{\| w_{\text{LLM}} - w_{\text{stated}} \| > \delta_{\text{tolerable}}}$$

donde $\delta_{\text{tolerable}}$ es el desalineamiento que el deployer
considera aceptable.

### 3.1. El problema de auditoría

Sin nuestro método, el deployer solo observa **una** decisión por turno
y no puede separar:
- (a) **value misalignment**: $w_{\text{LLM}} \neq w_{\text{stated}}$
- (b) **capability gap**: el LLM no sabe predecir bien $\phi(s, a)$
- (c) **stochastic noise**: variación turn-to-turn del sampler

Nuestra pipeline:
1. Genera $N$ observaciones $(s_t, \mathcal{A}(s_t), a_t)$ controladas
2. Recupera el posterior $P(\hat{w} \mid \text{datos})$ vía
   `fit_bayesian_irl`
3. Reporta $\hat{w}$ con HDI95 por dimensión + diagnostics
4. El deployer compara $\hat{w}$ contra $w_{\text{stated}}$ y decide

---

## 4. Quantificación operativa del harm

`guatemala_sim/harms.py` traduce el desalineamiento $w_{\text{LLM}} \neq
w_{\text{stated}}$ a unidades humanas via las elasticidades calibradas
del simulador:

```python
from guatemala_sim.harms import estimate_trajectory_harm, harm_difference_summary

harm_claude = estimate_trajectory_harm(state_inicial, state_final_claude)
harm_gpt    = estimate_trajectory_harm(state_inicial, state_final_gpt)

print(harm_difference_summary("Claude", harm_claude, "GPT-4o-mini", harm_gpt))
# → "Reemplazar GPT-4o-mini por Claude sobre 8 turnos (~2.0 años) implica
#    XXX,XXX hogares adicionales bajo línea de pobreza, YYY,YYY niños
#    adicionales fuera de escuela, y ZZZ muertes adicionales al año en el
#    equilibrio de cobertura. Welfare delta agregado: USD ±NNN M."
```

Las elasticidades vienen de:
- **Mortalidad ↔ cobertura salud**: Cutler, Deaton & Lleras-Muney
  (2006), *The Determinants of Mortality*, JEP 20(3).
- **Pobreza ↔ crecimiento**: Ravallion (2001), *Growth, Inequality and
  Poverty*, World Development 29(11). Elasticidad $-0.35$ ya en
  `world/macro.py`.

**Aclaración honesta**: estas estimaciones son aproximaciones de orden
de magnitud calibradas con literatura empírica, no proyecciones
vinculantes. El propósito es transformar resultados abstractos en
unidades operativas, no producir cifras de planificación.

---

## 4.bis Hipótesis específica del despliegue Sur Global: transfer cultural Norte→Sur

El threat model formalizado en §3 es **agnóstico al contexto cultural** —
el desalineamiento $\|w_{\text{LLM}} - w_{\text{stated}}\| > \delta$ vale
para cualquier país. Pero el caso de despliegue Sur Global (LatAm en
particular) tiene una hipótesis **adicional** que el método permite testear
empíricamente, y que da urgencia particular al threat model:

> **Hipótesis del transfer cultural (H_TC).** Los pesos $w_{\text{LLM}}$
> recuperados por IRL bayesiano sobre LLMs frontera reflejarán prioridades
> del corpus de entrenamiento (predominantemente US/UK liberal-fiscal),
> sistemáticamente desviadas de las prioridades documentadas de la
> población guatemalteca y de las prioridades declaradas en planes
> nacionales de desarrollo del Sur Global.

**Origen de la hipótesis.** Los modelos frontera son entrenados con
mecanismos calibrados en el Norte Global:

| Mecanismo de calibración | Origen | Sesgo cultural plausible |
|---|---|---|
| Pre-training corpus | Web anglo (~ 60–80 %) + español/portugués peninsulares | Vocabulario fiscal-prudencial centro-norte |
| RLHF rater pool | Anthropic / OpenAI / DeepMind contractors mayoritariamente US | Preferencias políticas medianas US |
| Constitutional AI principles | Redactados en California/Londres | Énfasis liberal-individualista |
| Frontier Safety Frameworks | Anthropic RSP, DeepMind FSF — corporate California/London | Threat models federales US/EU |
| Eval benchmarks | MMLU, BIG-bench, HELM | Preguntas con sesgo cultural anglo |

**Operacionalización testeable.** Si H_TC es cierta, el método de auditoría
debería revelar:

1. Constituciones LLM con $w_{\text{anti\_deuda}}$ alta (prudencia fiscal
   anglo-tradicional) por encima de lo que el system prompt declara.
2. $w_{\text{anti\_pobreza}}$ y $w_{\text{pro\_confianza}}$ por debajo de
   lo que documentos como ENCOVI o Latinobarómetro Guatemala 2024
   priorizan.
3. Cosine similarity bajo entre el $w_{\text{recovered}}$ y un
   $w_{\text{population}}$ codificado de encuestas LatAm de prioridades
   políticas.

**Por qué esta hipótesis es relevante para el threat model.** Si H_TC se
confirma, el riesgo no es "el LLM está mal alineado *en general*" — es "el
LLM está alineado a prioridades de un contexto distinto al de despliegue".
Eso es **transfer learning sin transparencia**: el deployer compra un
modelo asumiendo neutralidad cultural y obtiene un modelo culturalmente
sesgado en una dirección que no eligió.

Esa formulación cambia las recomendaciones operativas:

- No alcanza con auditar contra el system prompt declarado del deployer.
- Hay que auditar contra **las prioridades documentadas de la población
  destinataria** del despliegue.
- La auditoría se vuelve un *requisito de localización cultural* análogo a
  los estándares de localización lingüística.

**Cómo se testea con este testbed.** Codificás dos `w_stated`: uno del
system prompt actual (typical "priorizar reducción de pobreza"), otro
de una agregación de prioridades de Latinobarómetro Guatemala 2024 +
plan K'atun + ENCOVI. Aplicás `audit_llm_alignment` contra ambos y
reportás los dos alignment gaps. Si el gap contra el segundo es
sistemáticamente mayor → evidencia de H_TC.

**Limitación honesta.** Esto requiere un dataset adicional: prioridades
agregadas de la población guatemalteca codificadas en las 6 dimensiones
del IRL. Existe la materia prima (Latinobarómetro, LAPOP AmericasBarometer,
ENCOVI), falta el trabajo de codificación. Es la extensión natural Sprint 3
del proyecto.

---

## 5. Threat model: lo que NO afirma este trabajo

Tres cosas que el método **no resuelve** (importante para no
sobreafirmar):

1. **No previene daños reales**: el sim es una herramienta de
   evaluación pre-despliegue, no un control en producción.
2. **No identifica el mecanismo del desalineamiento**: detecta que
   $\hat{w} \neq w_{\text{stated}}$ pero no por qué (pre-training,
   RLHF, prompt, dataset).
3. **No es transferible automáticamente entre dominios**: $\hat{w}$
   recuperado en presupuesto guatemalteco no es necesariamente $\hat{w}$
   en presupuesto chileno o en asignación de portfolios. Necesita
   recalibración por dominio.

---

## 6. Threat model coverage en el paper

| Sección del paper | Componente del threat model que cubre |
|---|---|
| §1 Introducción | Actor, capability, harm pathway |
| §3 Method (IRL bayesiano) | Mitigation (la auditoría) |
| §4 Synthetic recovery | Validación de identificabilidad |
| §5 Empirical results | Magnitud del desalineamiento observado |
| §5.X Harm quantification | Harm units (hogares, muertes, USD) |
| §6 Discussion | Threshold, limitations |
| §7 Limitations | Lo que el método NO captura (sección 5 acá) |

---

## 7. Mapeo a frameworks de safety institucional

| Framework | Categoría aplicable | Nuestro método cubre |
|---|---|---|
| **Anthropic RSP** (Responsible Scaling Policy 2024) | "Autonomy and decision-making capabilities" — ASL-3 monitoring | Auditoría pre-despliegue de preferencias en decisiones bajo restricción |
| **DeepMind Frontier Safety Framework** (2024) | "Persuasion and Manipulation" + "Autonomous Operation" | Detección de sesgos sistemáticos en outputs estructurados |
| **UK AISI Inspect** | "Bias and Misuse" → "Allocation and Resource Distribution" | Métrica reproducible de revealed preferences en asignación |
| **NIST AI Risk Management Framework** | Map → Measure → Manage de "value alignment risk" | Provee la fase Measure con incertidumbre cuantificada |

---

## 8. Referencias

- Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI
  Feedback*. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073).
- Bowman, S., et al. (2022). *Measuring Progress on Scalable Oversight
  for Large Language Models*.
  [arXiv:2211.03540](https://arxiv.org/abs/2211.03540).
- Casper, S., et al. (2023). *Open Problems and Fundamental Limitations
  of RLHF*. [arXiv:2307.15217](https://arxiv.org/abs/2307.15217).
- Cutler, D., Deaton, A., & Lleras-Muney, A. (2006). *The Determinants
  of Mortality*. JEP 20(3), 97–120.
- Hadfield-Menell, D., et al. (2017). *Inverse Reward Design*. NeurIPS.
  [arXiv:1711.02827](https://arxiv.org/abs/1711.02827).
- NIST (2012). SP 800-30 Rev 1, *Guide for Conducting Risk Assessments*.
- Perez, E., et al. (2022). *Discovering Language Model Behaviors with
  Model-Written Evaluations*.
  [arXiv:2212.09251](https://arxiv.org/abs/2212.09251).
- Ravallion, M. (2001). *Growth, Inequality and Poverty: Looking Beyond
  Averages*. World Development 29(11).
- Russell, S. (2019). *Human Compatible: AI and the Problem of
  Control*. Viking.
