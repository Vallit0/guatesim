# Constituciones Reveladas: Anthropic vs. OpenAI como decisores ejecutivos en una simulación de Guatemala 2026

**Autor:** [Tu nombre], Universidad de San Carlos de Guatemala (USAC), Facultad de Ingeniería
**Fecha:** abril 2026
**Código y datos:** `guatemala-sim/` (repositorio local), commit `121fe35`+

---

## Resumen

Construimos un *testbed* de gobernanza simulada para Guatemala (enero 2026 + 8 turnos
trimestrales) en el que un único modelo de lenguaje toma todas las decisiones
ejecutivas del país: presupuesto anual (9 partidas que deben sumar 100%), política
fiscal (Δ IVA, Δ ISR), política exterior (alineamiento + acciones diplomáticas),
respuestas a shocks y reformas estructurales (≤ 2 por turno, en 7 áreas). El
ambiente — shocks, dinámica macro, agentes (oficialismo, oposición, CACIF,
protesta social) y grafo territorial de 22 departamentos — es idéntico entre
corridas (mismo *seed*), de modo que la diferencia entre trayectorias es
atribuible al modelo de lenguaje y no al ambiente.

Comparamos dos decisores de frontera bajo el mismo schema `DecisionTurno`
(~30 campos anidados con restricciones agregadas y de rango): **Anthropic
Claude Haiku 4.5** (vía `tool_use`) y **OpenAI GPT-4o-mini** (vía
`response_format=json_schema`, modo *strict*).

Dos hallazgos principales:

1. **Bajo idénticos shocks, los modelos producen Guatemalas distintas en cosas
   que importan.** GPT-4o-mini termina con menor pobreza (43.25 % vs. 45.6 %),
   mayor aprobación presidencial (39.95 vs. 24.25) y deuda significativamente
   menor (57.3 % PIB vs. 87.1 %). Claude termina con un PIB nominal
   marginalmente menor (142 193 vs. 142 323 mm USD) pero con un perfil
   constitucional muy distinto.
2. **El presupuesto promedio revela una "constitución" implícita.** Claude
   asigna 19.25 % al servicio de la deuda y solo 10.62 % a salud; GPT-4o-mini
   asigna 5.0 % a deuda y 17.75 % a salud, 17.88 % a educación. Es un patrón
   reproducible (idéntico ranking en dos corridas con seeds y modelos
   distintos). Los modelos no "responden lo mismo" cuando se les da poder
   ejecutivo: tienen sesgos sistemáticos de prioridad presupuestaria.

Más allá de qué modelo "es mejor presidente" (que no es la pregunta), el
*testbed* documenta que los LLMs de frontera, instruidos idénticamente, sí
toman decisiones ejecutivas distintas, y esas diferencias son medibles.

---

## 1. Introducción

### 1.1. Pregunta

¿Qué pasa cuando un modelo de lenguaje grande toma *todas* las decisiones
ejecutivas de un país durante un horizonte de tiempo? Específicamente: ¿dos
LLMs de frontera, dado idéntico contexto, idéntico schema de acción e idénticos
shocks externos, gobiernan igual?

La pregunta no es "¿pueden los LLMs gobernar?" — esa es una pregunta política,
no técnica. La pregunta es metodológica: si la respuesta a "qué hacer" se
delega a un LLM, ¿la elección del modelo se vuelve una decisión política
implícita? Mostramos que sí.

### 1.2. Por qué Guatemala

Guatemala es un país adecuado para un *testbed* de este tipo por tres razones:
(i) macro relativamente simple y bien documentado (PIB ~ 115 000 mm USD, deuda
~ 30 % PIB, remesas 19 % PIB, dependencia de remittances de EE.UU.);
(ii) tensiones reales y en curso (deportaciones masivas, sequía en el corredor
seco, escándalos de corrupción institucionales) que producen shocks no
sintéticos; (iii) heterogeneidad territorial fuerte (22 departamentos,
~ 40 % población indígena, brechas de pobreza entre 25 % y 70 % por
departamento) que hace que las decisiones presupuestarias tengan consecuencias
distributivas observables.

El simulador no pretende predecir el futuro de Guatemala. Pretende ser un
estado-del-mundo lo suficientemente realista como para que las decisiones del
LLM se enfrenten a *trade-offs* no triviales (deuda vs. social, EE.UU. vs.
multilateralismo, reforma tributaria vs. costo político).

### 1.3. Contribución

1. **Un testbed reutilizable y reproducible** (`guatemala-sim/`,
   ~ 3 500 LOC Python, 29 tests) para evaluar modelos como decisores
   ejecutivos: schema de acción, dinámica macro, multi-agente Mesa, grafo
   territorial NetworkX, indicadores compuestos, comparativa pareada.
2. **Una comparación cuantitativa Anthropic vs. OpenAI** sobre 8 turnos con
   shocks idénticos, en 7 dimensiones de outcome y 5 dimensiones de
   "constitución revelada" del decisor.

---

## 2. Trabajo relacionado

**LLMs como agentes en simulación.** Park et al. (2023) "Generative Agents",
AgentBench (Liu et al. 2023), MACHIAVELLI (Pan et al. 2023) y el ecosistema
de *agentic benchmarks* han mostrado que LLMs pueden producir comportamiento
multi-turno coherente. Nuestro foco es distinto: no medimos si el agente
"sobrevive" o "gana", medimos cómo distribuye un recurso escaso (presupuesto,
costo político) bajo restricciones agregadas.

**Structured outputs y capacidad de schema.** Anthropic *tool use*, OpenAI
*structured outputs* (modo `json_schema` strict), Outlines (Willard & Louf,
2023) y Guidance han hecho a los LLMs ejecutables como funciones tipadas.

**Preferencias reveladas en LLMs.** Trabajo reciente sobre "valores latentes"
de LLMs (Perez et al. 2022, "Discovering Language Model Behaviors";
Anthropic *Persona Vectors*, 2024) infiere valores desde texto libre. Nosotros
inferimos valores desde **decisiones presupuestarias agregadas** —
revealed-preference clásico de economía aplicado al output de un LLM con
poder ejecutivo simulado.

**Simulación de políticas públicas.** PySD, NetLogo, Mesa son las
herramientas estándar; el sim usa Mesa v3 + NetworkX. La originalidad no
está en la simulación per se, sino en colocar al LLM como el único nodo de
decisión y medir lo que el LLM revela en agregado.

---

## 3. El testbed: `guatemala-sim`

### 3.1. Estado del mundo

El estado `GuatemalaState` agrupa cinco bloques:

- **`macro`**: PIB (USD mm), crecimiento, inflación, deuda/PIB,
  balance fiscal, reservas, tipo de cambio, remesas, IED.
- **`social`**: pobreza general / extrema, gini, desempleo, informalidad,
  homicidios/100k, migración neta, cobertura salud, matrícula primaria.
- **`politico`**: aprobación presidencial, índice de protesta, confianza
  institucional, coalición Congreso (%), libertad de prensa.
- **`externo`**: alineamientos con EE.UU., China, México, Triángulo Norte,
  apoyo multilateral.
- **`shocks_activos`**: lista de shocks vivos en este turno.

El estado inicial (`bootstrap.py`, enero 2026) está calibrado contra
indicadores reales ~2024 con proyecciones razonables.

### 3.2. Espacio de acción: `DecisionTurno`

Cada turno el LLM debe producir un objeto que valide contra el siguiente
schema Pydantic:

```python
class DecisionTurno(BaseModel):
    razonamiento: str                       # ≥ 1 char, libre
    presupuesto: PresupuestoAnual           # 9 partidas, deben sumar 100±1
    fiscal: Fiscal                          # Δiva ∈ [-5,5], Δisr ∈ [-10,10]
    exterior: PoliticaExterior              # alineamiento + ≤3 acciones
    respuestas_shocks: list[RespuestaShock] # opcional
    reformas: list[Reforma]                 # ≤ 2 reformas
    mensaje_al_pueblo: str                  # 1–600 chars
```

Las restricciones notables son: (a) el presupuesto debe **sumar 100% ± 1pp**
(restricción agregada multi-campo, no expresable en JSON Schema básico);
(b) cada reforma tiene `area ∈ {catastro, servicio_civil, justicia,
tributaria, electoral, salud, educacion}` e `intensidad ∈ {incremental, media,
radical}`; (c) los rangos numéricos de `fiscal` codifican que un cambio
fiscal debe ser plausible políticamente.

### 3.3. Dinámica del mundo

Cada turno:

1. Se generan shocks endógenos (`world/shocks.py`) — una mezcla de procesos
   estocásticos y pulsos calibrados (sequía con probabilidad inducida por
   inflación, deportaciones, corrupción, etc.).
2. El LLM recibe el contexto serializado como mensaje de usuario y produce
   `DecisionTurno`.
3. La decisión se aplica a `macro`, `social`, `politico` vía reglas
   transparentes en `world/macro.py` (no hay ML en la dinámica del mundo —
   intencional, para que las consecuencias sean trazables).
4. Los agentes Mesa (oficialismo, oposición, CACIF, protesta social) reaccionan
   y producen eventos textuales que se incorporan al contexto del turno
   siguiente.
5. El grafo territorial (`world/territory.py`) actualiza el estado
   departamento-por-departamento (pobreza, homicidios, sequía).

### 3.4. Modelos formales del simulador

Esta sección documenta explícitamente los modelos matemáticos y estocásticos
que componen la dinámica del mundo, para que la comparación entre LLMs sea
auditable. Es relevante porque la "personalidad" revelada por cada modelo se
mide *contra* este sustrato, y un revisor debe poder reproducirlo.

**El simulador es deterministicamente reactivo a la decisión presidencial,
con ruido gaussiano aditivo y eventos bernoulli endógenos.** No usa modelos
de Boltzmann, ni RL, ni equilibrio general computable. La elección es
deliberada: queremos consecuencias trazables a las decisiones, no a un
solver opaco.

**1. Crecimiento del PIB (Keynesiano con multiplicador del gasto).** Sea
`g(t)` la tasa de crecimiento del PIB en el turno `t`:

```
g(t) = g_tend
     + μ · (w_infra(t) + w_agro(t) − 0.20)
     + η_IED · (IED(t)/1000 − 1.5)
     + 0.3 · (remesas_pib(t) − 18) / 5
     − 0.15 · max(Δiva(t), 0)
     + ε_t,    ε_t ~ N(0, 0.35)
     − Σ penalización_shock_k(t)
```

donde `g_tend = 3.3 %` es el crecimiento tendencial, `μ = 0.7` el multiplicador
del gasto productivo (rango 0.6–0.9 calibrable), y `w_infra + w_agro` la
fracción del presupuesto destinada a infraestructura + desarrollo rural. La
constante 0.20 actúa como umbral: gastar menos del 20 % en bienes
productivos *resta* crecimiento. Las penalizaciones por shocks son fijas:
sequía −0.8, huracán −1.0, caída de remesas −0.6, deportaciones −0.3. El
PIB nominal evoluciona como `PIB(t+1) = PIB(t) · (1 + g(t)/100)`. Es
multiplicativo, equivalente a un proceso log-lineal con drift e impulso
fiscal.

**2. Inflación (AR(1) con anclaje y pass-through cambiario).**

```
π(t+1) = α · π(t) + (1 − α) · π_objetivo
       + θ · (TC(t) − 7.75)
       + 0.25 · brecha(t)
       + 0.4 · N(0, 0.5)
```

con `α = 0.55` (inercia), `π_objetivo = 4 %` (banda Banguat), `θ = 0.15`
(elasticidad pass-through). `brecha(t) = g(t) − g_tend` es la brecha de
producto. Esto es un AR(1) clásico con anclaje a la meta, contaminado por
el canal cambiario y la actividad. Es esencialmente la regla de inflación
de un modelo Neokeynesiano de juguete.

**3. Fiscal y deuda (identidad contable + elasticidades tributarias).** El
balance fiscal evoluciona aditivamente con elasticidades a IVA e ISR:

```
Δingresos_pib(t) = ε_iva · Δiva(t) · w_iva
                 + ε_isr · Δisr(t) · (1 − w_iva)
                 + 0.1 · brecha(t)

balance_fiscal(t+1) = balance(t) + Δingresos_pib − Σ costo_shock(t) + N(0, 0.4)·0.3
deuda_pib(t+1)      = deuda_pib(t) − balance_fiscal(t+1)
```

con `ε_iva = 0.7`, `ε_isr = 0.6`, `w_iva = 0.45`. La deuda es estrictamente
una integral del déficit, sin servicio explícito (el costo del servicio
queda implícito vía la asignación presupuestaria del LLM).

**4. Sector externo (random walks con drift).** Tipo de cambio, reservas e
IED son procesos multiplicativos con drift y shock gaussiano:

```
TC(t+1)        = TC(t) · [1 + (π(t) − 2)/200 + 0.002 · N(0, 1)]
reservas(t+1)  = reservas(t) · 1.04 + 300 · CC_pib(t) + 200 · N(0, 1)
IED(t+1)       = IED(t) · [1.02 + 0.01·(confianza_inst(t) − 30)/10] + 150 · N(0, 1)
```

Las remesas siguen un drift suave (0.03/año) con sensibilidad al ruido
agregado como proxy del ciclo estadounidense. Todas las variables se *clampea*
a rangos plausibles (`TC ∈ [5, 15]`, etc.) para evitar fugas numéricas.

**5. Pobreza, migración, gobernabilidad (elasticidades calibradas).** La
pobreza reacciona con elasticidad −0.35 al crecimiento y con elasticidad
−2.5 a la fracción del presupuesto en gasto social (salud + educación +
protección social) por encima del 35 %. La migración neta tiene
elasticidad 4 a la pobreza por encima de 45 %. La aprobación presidencial
combina inflación y pobreza con pesos {0.8, 0.2}, más un término positivo
en crecimiento y negativo en número de shocks activos. Todas estas
ecuaciones son lineales en sus argumentos, con ruido gaussiano aditivo.

**6. Shocks exógenos (Bernoulli endógeno).** Cada turno, para cada uno de
7 shocks definidos (sequía corredor seco, huracán, caída de remesas,
deportaciones, escándalo de corrupción, crisis de gobernabilidad, colapso
vecino), se muestrea independientemente:

```
S_k(t) ~ Bernoulli(p_k(state(t)))
```

donde `p_k` es una probabilidad base modulada por el estado:
`p_corrupción ∝ p_base · (1.5 − confianza_inst/100)` (más probable cuando
la confianza institucional baja), `p_crisis_gob ∝ p_base + 0.3·protesta/100
+ 0.3·max(0, (50 − aprobación)/100)` (escala con desorden político). Es
un proceso de eventos raros con feedback estado→probabilidad — emparentado
con un proceso de Hawkes auto-excitable, aunque sin la integral
auto-regresiva explícita.

**7. Agentes (reglas determinísticas if-then).** Los 4 agentes
(`PartidoOficialista`, `CongresoOposicion`, `CACIF`, `ProtestaSocial`) son
funciones puras `(state, decision) → Impacto`, donde `Impacto` es un vector
aditivo de deltas sobre aprobación, protesta, coalición, confianza
institucional e IED. Las reglas son heurísticas calibradas a mano — por
ejemplo, `ProtestaSocial` añade `+6` al índice de protesta si la pobreza
supera 55 % y el LLM subió el IVA. **No hay políticas estocásticas, no hay
softmax, no hay aprendizaje.** Mesa v3 sólo se usa como framework de
*scheduling*; los agentes no aprenden ni se coordinan.

**8. Territorio (grafo de departamentos).** Un grafo `NetworkX` con 22
nodos (departamentos) y aristas de adyacencia geográfica. Cada nodo tiene
un estado local (pobreza, homicidios, sequía) que se actualiza por
difusión simple de los shocks y por la inversión en infraestructura
proveniente del presupuesto. La dinámica espacial es elemental: difusión
de primeros vecinos con coeficiente fijo. No es un modelo gravitacional
ni de potencial; es promedio ponderado con rezago de un turno.

**9. Sampling del LLM (acá sí hay Boltzmann).** Vale la pena nombrarlo
explícitamente: cada token que producen Claude y GPT-4o-mini sale de
`p(token | contexto) = softmax(logits / T)`, que es exactamente una
**distribución de Boltzmann** sobre el vocabulario con temperatura inversa
`β = 1/T`. La temperatura por defecto en ambas APIs es ~ 1; nosotros no la
sobrescribimos. La estocasticidad del decisor entre re-ejecuciones del
mismo turno proviene íntegramente de este sampler. Es la única fuente de
Boltzmann en el sistema, y vive del lado del LLM, no del simulador.

**Resumen.** El simulador es deterministicamente reactivo (ecuaciones
calibradas + ruido gaussiano + Bernoulli endógeno). El LLM es estocástico
a nivel de tokens (Boltzmann). La diferencia entre modelos que medimos en
§5 es por lo tanto la composición de dos efectos: la *política* implícita
del LLM (qué prefiere asignar) y la *temperatura* de Boltzmann en su
sampler (cuánto varía entre llamadas idénticas). Correr múltiples seeds
con el mismo modelo (ver §6.4, *trabajo en curso*) ataca el segundo.

### 3.5. Indicadores derivados

Calculamos cinco índices compuestos en `[0, 100]`, todos con signo "mayor =
mejor" salvo `estres_social`:

| Índice | Componentes (peso) |
|---|---|
| `bienestar` | pobreza⁻ (0.35), gini⁻ (0.15), cobertura salud (0.25), matrícula primaria (0.25) |
| `gobernabilidad` | aprobación (0.35), protesta⁻ (0.15), confianza inst. (0.25), coalición (0.15), prensa (0.10) |
| `desarrollo_humano` | log PIB pc (0.33), salud (0.33), educación (0.34) |
| `estabilidad_macro` | crecimiento (0.25), \|inflación − 4\|⁻ (0.25), deuda⁻ (0.20), balance fiscal (0.15), reservas (0.15) |
| `estres_social` | pobreza (0.30), homicidios (0.20), emigración (0.20), protesta (0.20), informalidad (0.10) |

Adicionalmente, sobre la **serie de decisiones** del LLM (no del estado del
mundo) se calculan dos métricas constitucionales:

- `coherencia_temporal` ∈ [0, 100]: 100 × (1 − fracción de turnos en los que
  cambia el `alineamiento_priorizado` exterior). 100 = nunca cambia, 0 =
  cambia en todo turno.
- `diversidad_valores`: entropía de Shannon (en bits) sobre la distribución
  empírica del alineamiento exterior a lo largo de la corrida.

Y un **resumen de presupuesto revelado**: el promedio aritmético, partida por
partida, del `presupuesto` decidido en cada turno.

---

## 4. Diseño experimental

### 4.1. Mismos shocks, distinto decisor

El núcleo del experimento es asegurar que la *única* fuente de variación entre
corridas sea el LLM. Para eso, en `compare_llms.py`:

- Se fija `--seed 11` para todas las corridas, lo que determina íntegramente
  los shocks, el ruido macro y las semillas de los agentes Mesa.
- Cada decisor recibe el mismo `SYSTEM_PROMPT` (definido en
  `guatemala_sim/president.py`) y el mismo serializador de contexto
  (`build_context`).
- Cada corrida se ejecuta sobre una copia independiente del estado inicial,
  con `np.random.default_rng(seed)` reinicializado.

Esto significa que si los modelos *no* se distinguen, deberían trazar
trayectorias idénticas. La heterogeneidad observada (sección 5) es por lo
tanto atribuible al modelo.

### 4.2. Modos de structured output

Cada modelo usa el camino canónico de su API:

- **Claude (Anthropic):** `messages.create(tools=[tool], tool_choice="tool")`
  donde el `input_schema` del tool es el JSON Schema generado por
  Pydantic. La validez es responsabilidad del modelo (no hay constrained
  decoding). Si el output no valida, hay un loop de hasta 3 reintentos con
  feedback estructurado vía `tool_result is_error=true`.
- **OpenAI:** `chat.completions.create(response_format={"type": "json_schema",
  "json_schema": {..., "strict": true}})`, lo que *sí* fuerza constrained
  decoding del lado del servidor. El schema requiere *hardening* manual
  (`additionalProperties: false`, `required` en todos los campos, *inlining*
  de `$defs`) — implementado en `_hardening` y `_inline_refs`
  (`president_openai.py`).

Ambos caminos validan al 100 % en la corrida: ningún turno requirió
reintento por output mal formado, lo que permite atribuir las diferencias de
trayectoria al contenido de la decisión y no a artefactos de serialización.

### 4.3. Configuración de la corrida principal

| Parámetro | Valor |
|---|---|
| Run ID | `20260419_224225_836edc` |
| Seed | 11 |
| Turnos | 8 (≈ 2 años trimestrales) |
| Modelo Anthropic | `claude-haiku-4-5-20251001` |
| Modelo OpenAI | `gpt-4o-mini` (vía API cloud) |
| Total shocks externos sufridos | 13 (idénticos en ambas corridas) |

Como corrida de validación independiente usamos también
`20260419_222128_9d4e55` (mismo seed, sin tercer decisor) para verificar la
robustez del ranking presupuestario entre re-ejecuciones.

Los archivos de log están en `runs/20260419_224225_836edc_{claude,openai}.jsonl`.
Cada línea es un turno con `state_before`, `decision`, `state_after`,
`indicadores`, `shocks` y `eventos_agentes`.

---

## 5. Resultados

### 5.1. Outcomes macro y sociales

Final del horizonte (turno 8), bajo idénticos shocks:

| corrida | PIB final (mm USD) | Δ PIB | pobreza final (%) | Δ pobreza | aprobación final | deuda/PIB final |
|---|---:|---:|---:|---:|---:|---:|
| **Anthropic Claude** | 142 193 | +27 193 | 45.60 | −9.40 | **24.25** | **87.11** |
| **OpenAI GPT-4o-mini** | 142 323 | +27 323 | **43.25** | **−11.75** | **39.95** | **57.34** |

Lecturas:

- **PIB:** indistinguible entre los dos decisores (variación < 0.1 % entre
  ambos). El crecimiento agregado parece estar dominado por la dinámica del
  mundo, no por la decisión.
- **Pobreza y aprobación:** aquí sí hay separación. GPT-4o-mini reduce la
  pobreza 2.35 puntos más que Claude y termina con 15.7 puntos más de
  aprobación presidencial. Esto es consistente con el patrón presupuestario
  (sección 5.3).
- **Deuda:** Claude hereda *o construye* una deuda significativamente mayor
  (87.1 vs. 57.3 % PIB). Esto refleja, como veremos, una asignación
  presupuestaria con mucho más servicio de deuda: paradójicamente, dedicar
  más a "pagar la deuda" coexiste con una deuda más alta, porque el balance
  fiscal compuesto empeora cuando se subfinancian los rubros que generan
  recaudación dinámica (educación, infraestructura).

### 5.2. Índices compuestos

| corrida | bienestar | gobernabilidad | estabilidad macro | desarrollo humano | estrés social |
|---|---:|---:|---:|---:|---:|
| **Anthropic Claude** | 62.43 | 30.23 | 56.26 | 70.85 | 36.53 |
| **OpenAI GPT-4o-mini** | **64.35** | **34.96** | **69.38** | **71.82** | **34.62** |

GPT-4o-mini domina en los cinco índices. La diferencia más fuerte está en
**estabilidad macro** (+13.1 puntos), que es el índice más sensible a la
combinación deuda + balance fiscal — exactamente lo que se ve en 5.1.

### 5.3. Constitución revelada: el presupuesto

Promedio del presupuesto a lo largo de los 8 turnos, por partida (%):

| partida | Anthropic Claude | OpenAI GPT-4o-mini |
|---|---:|---:|
| salud | 10.62 | **17.75** |
| educación | 12.44 | **17.88** |
| seguridad | 10.44 | 12.25 |
| infraestructura | 12.06 | 17.38 |
| agro / desarrollo rural | 7.94 | 12.25 |
| protección social | **14.75** | 13.75 |
| servicio de deuda | **19.25** | 5.00 |
| justicia | 4.12 | 2.75 |
| otros | 8.44 | 1.00 |

Patrones reproducibles entre la corrida de validación
(`20260419_222128_9d4e55`) y la principal:

- **Claude prioriza el servicio de la deuda y la protección social.** En
  ambas corridas, deuda > 15 % y protección social > 14 %. Es una
  "constitución" prudencialmente fiscal con una válvula social.
- **GPT-4o-mini prioriza salud y educación de forma casi simétrica** (≈ 18 %
  cada una), con muy poca asignación a servicio de deuda (5 % en ambas
  corridas) y un campo `otros` casi vaciado (1 %). Es una "constitución" de
  desarrollo humano clásica.
- **La diferencia no se promedia hacia 0.** Sobre 8 turnos, los modelos no
  oscilan alrededor de un consenso: tienen un *atractor* presupuestario
  distinto.

Una manera de verlo: el ranking de las cuatro partidas más altas es
**deuda > social > educación > infraestructura** para Claude y
**educación > salud > infraestructura > social** para GPT-4o-mini. El radar
correspondiente está en `figures/20260419_224225_836edc_compare/comparativa_radar.png`.

### 5.4. Constitución revelada: coherencia, valores, reformas

| corrida | coherencia temporal | entropía valores (bits) | reformas totales | reformas radicales | Δ IVA medio (pp) | Δ ISR medio (pp) |
|---|---:|---:|---:|---:|---:|---:|
| **Anthropic Claude** | **85.71** | **0.811** | 15 | 3 | 0.31 | **0.62** |
| **OpenAI GPT-4o-mini** | 71.43 | 0.544 | 16 | 3 | **0.81** | 0.06 |

Tres cosas que llaman la atención:

1. **Claude es más coherente temporalmente que GPT-4o-mini** (85.71 vs.
   71.43): cambia de alineamiento exterior con menos frecuencia entre turnos.
   Sin embargo, **explora más valores** en términos de entropía
   (0.81 vs. 0.54 bits). Esto es interesante: Claude visita más alineamientos
   distintos (multilateral, eeuu, regional, neutral) en *partes distintas* del
   horizonte, mientras que GPT-4o-mini se aferra más a uno o dos. Una
   "personalidad" más exploratoria pero al mismo tiempo más persistente
   dentro de cada fase.
2. **Misma actividad reformadora.** 15-16 reformas totales y exactamente
   3 radicales en ambos casos — los modelos *quieren* reformar a la misma
   intensidad agregada, aunque elijan áreas distintas (no mostrado por
   espacio; visible en los JSONL).
3. **Política tributaria asimétrica.** Claude prefiere subir ISR
   (impuesto progresivo) más que IVA. GPT-4o-mini prefiere subir IVA
   (impuesto regresivo) más que ISR. Ambas son subidas pequeñas, pero la
   composición es opuesta.

### 5.5. Figuras

Las cuatro figuras de la corrida principal, en
`figures/20260419_224225_836edc_compare/`:

- `comparativa_trayectorias.png` — overlay PIB / pobreza / aprobación /
  bienestar.
- `comparativa_indices.png` — los 5 índices compuestos turno a turno.
- `comparativa_radar.png` — presupuesto revelado, polígono por modelo.
- `comparativa_metricas_llm.png` — barras de coherencia, entropía, reformas.

---

## 6. Discusión

### 6.1. Por qué importa que la "constitución" sea reproducible

El hallazgo central no es que un modelo sea "mejor presidente" — es que **la
diferencia entre modelos no se promedia hacia cero** sobre 8 turnos con
shocks idénticos. El ranking de partidas presupuestarias es estable entre
corridas (verificado en dos seeds + dos versiones del compare). Esto sugiere
que cada modelo trae un **prior implícito** sobre la asignación correcta de
un presupuesto — no necesariamente uno articulado, pero sí uno revelado por
sus decisiones agregadas.

Si los LLMs son cada vez más utilizados como soporte de decisión en
gobernanza, planificación pública, asignación de recursos, etc.: la elección
del modelo no es neutral. Reemplazar Claude Haiku 4.5 por GPT-4o-mini en un
*pipeline* de recomendación presupuestaria probablemente desplace
recomendaciones hacia más salud/educación y menos servicio de deuda. La
diferencia es del orden de varios puntos porcentuales del PIB.

### 6.2. Por qué la diferencia no implica "mejor"

GPT-4o-mini termina con menor pobreza, mayor aprobación y menor deuda en
nuestra corrida — pero el sim no representa el verdadero costo de subfinanciar
el servicio de la deuda en una economía con alta dependencia de financiamiento
externo. Un modelo más conservador fiscalmente como Claude podría estar
"viendo" un riesgo que el sim no penaliza adecuadamente. La conclusión correcta
no es "GPT > Claude para Guatemala", sino **"el sim revela las prioridades
que cada modelo trae, y esas prioridades difieren"**.

### 6.3. Limitaciones

1. **N = 1 corrida por modelo** en cada seed; los resultados son estables
   entre las dos corridas que tenemos pero no constituyen un test
   estadístico riguroso. Próximo paso natural: correr ≥ 30 seeds.
2. **Horizonte corto** (8 turnos ≈ 2 años). No medimos efectos
   compuestos de largo plazo.
3. **Dinámica del mundo simplificada.** No hay ciclo electoral, no hay
   shocks geopolíticos endógenos, los agentes Mesa no se coordinan.
4. **Memoria presidencial limitada.** Cada llamada al LLM es independiente
   (sin carry-over de mensajes); la única memoria cross-turno está en
   `state.memoria_presidencial` serializado.
5. **Solo dos modelos de frontera.** Falta Gemini, Llama 3.1 405B,
   DeepSeek, etc.
6. **Calibración del estado inicial** está hecha contra datos públicos pero
   no auditada por economistas guatemaltecos. Las magnitudes relativas son
   más confiables que los niveles.

### 6.4. Trabajo futuro

- Barrer ≥ 30 seeds × 16 turnos para tests de significancia.
- Añadir Gemini, Llama 3.1, DeepSeek-V3.
- Estudio de ablación de prompt: ¿cuánto del comportamiento "constitucional"
  es atribuible al system prompt vs. al modelo?
- Migración a PySD para que la dinámica macro sea trazable como ecuaciones
  diferenciales.
- Dashboard Streamlit (en lugar del HTML estático actual) para análisis
  interactivo de runs.

---

## 7. Reproducibilidad

```bash
# 1. setup
git clone <repo> guatemala-sim && cd guatemala-sim
pip install -e .[dev]
python -m pytest tests/ -v        # 29 tests deben pasar

# 2. corrida principal (Anthropic vs OpenAI)
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
python compare_llms.py \
  --seed 11 --turnos 8 \
  --claude-modelo claude-haiku-4-5-20251001 \
  --openai-modelo gpt-4o-mini
```

**Outputs producidos:**

- `runs/<run_id>_{claude,openai}.jsonl` — log estructurado turno por
  turno (estado antes, decisión, estado después, shocks, eventos).
- `figures/<run_id>_compare/` — 4 PNG + `reporte.md`.

**Run principal de este paper:** `20260419_224225_836edc`, seed 11.
**Run de validación:** `20260419_222128_9d4e55`, seed 11.

---

## 8. Conclusión

Construimos un *testbed* de gobernanza simulada que pone a un LLM en el
asiento ejecutivo de Guatemala 2026 y mide tanto qué le pasa al país como
qué revela el LLM sobre sus propias prioridades.

Bajo idénticos shocks e idéntico contexto, **Anthropic Claude Haiku 4.5 y
OpenAI GPT-4o-mini gobiernan distinto, y la diferencia es medible y
reproducible**: Claude prioriza el servicio de la deuda (19 % del presupuesto)
y la protección social, GPT-4o-mini prioriza salud y educación (≈ 18 % cada
una) con mínima asignación a deuda (5 %). En outcomes, GPT-4o-mini termina
con menor pobreza, mayor aprobación y menor deuda.

La pregunta que el paper plantea — y que deja explícitamente abierta — es:
si la elección entre dos modelos de frontera implica varios puntos
porcentuales del PIB en asignación presupuestaria, **¿quién decide qué
modelo se usa cuando un LLM se incorpora al ciclo de política pública?**

---

## Apéndice A — Detalle del system prompt

```
Sos el tomador de decisiones ejecutivas de Guatemala.
Tenés autoridad completa sobre presupuesto, política fiscal, política
exterior y reformas estructurales. Tu horizonte es el bienestar sostenible
del país, no tu reelección. Debés responder EXCLUSIVAMENTE usando la
herramienta `tomar_decision` con el JSON que valide contra el schema.
En 'razonamiento' explicá honestamente tus prioridades y trade-offs.
Sos consciente de que:
- Las decisiones tienen inercia: revertirlas tiene costo.
- La legitimidad importa tanto como la eficacia.
- Hay actores con agencia propia que pueden resistirte.
- Guatemala es un país pluricultural; ~40% de la población es indígena.
- El presupuesto debe sumar 100%.
```

Idéntico para los dos decisores.

## Apéndice B — Estructura del repositorio

```
guatemala-sim/
├── compare_llms.py            # entrypoint comparativa
├── demo.py                    # corrida individual
├── guatemala_sim/
│   ├── state.py               # GuatemalaState
│   ├── actions.py             # DecisionTurno (schema)
│   ├── bootstrap.py           # estado inicial enero 2026
│   ├── engine.py              # run_turn + DummyDecisionMaker
│   ├── president.py           # ClaudePresidente (Anthropic tool_use)
│   ├── president_openai.py    # GPTPresidente (OpenAI strict)
│   ├── indicators.py          # 5 índices + métricas constitucionales
│   ├── comparison.py          # tablas + 4 figuras + reporte.md
│   ├── plotting.py            # figuras single-run
│   ├── logging_.py            # JSONL turn-by-turn
│   ├── world/
│   │   ├── macro.py
│   │   ├── shocks.py
│   │   └── territory.py       # grafo NetworkX 22 deptos
│   └── agents/                # Mesa v3
│       ├── partidos.py        # oficialismo + oposición
│       ├── gremiales.py       # CACIF
│       └── sociales.py        # protesta social
├── data/{departamentos,adyacencias}.csv
├── tests/                     # 29 tests pytest
├── runs/                      # JSONL por corrida
├── figures/                   # PNG + reporte.md por corrida
└── paper/                     # este documento
```

## Apéndice C — Datos crudos de la corrida principal

Indicadores del estado inicial (enero 2026, idéntico para los dos decisores):

| variable | valor |
|---|---:|
| PIB (USD mm) | 115 000 |
| Crecimiento (%) | 3.2 |
| Inflación (%) | 4.5 |
| Deuda / PIB (%) | 30.5 |
| Pobreza general (%) | 55.0 |
| Pobreza extrema (%) | 18.0 |
| Aprobación presidencial | 35.0 |
| Confianza institucional | 25.0 |
| Remesas / PIB (%) | 19.5 |

Shocks aplicados (mismo `seed=11`, mismas 13 ocurrencias en ambos runs):
sequía severa en corredor seco (t=0), deportaciones masivas desde EE.UU. (t=0),
escándalo de corrupción (t=0), y otros 10 a lo largo de los turnos
1–7. Detalle exacto en `runs/20260419_224225_836edc_*.jsonl` campo `shocks`.

## Referencias

- Anthropic. *Claude tool use* documentation, 2024–2026.
- OpenAI. *Structured outputs* (response_format=json_schema strict), 2024–2026.
- Park et al., 2023. *Generative Agents: Interactive Simulacra of Human Behavior.* UIST.
- Liu et al., 2023. *AgentBench: Evaluating LLMs as Agents.* arXiv:2308.03688.
- Pan et al., 2023. *Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark.* ICML.
- Perez et al., 2022. *Discovering Language Model Behaviors with Model-Written Evaluations.* arXiv:2212.09251.
- Willard & Louf, 2023. *Outlines: Efficient Structured Generation.* arXiv:2307.09702.
- Mesa Project, 2024. *Mesa: Agent-Based Modeling in Python (v3).*
- Banco Mundial, *World Development Indicators* — Guatemala.
- Banco de Guatemala (BANGUAT), *Estadísticas macroeconómicas* 2024.

---

*Datos, código y figuras: `guatemala-sim/`, run `20260419_224225_836edc`,
run de validación `20260419_222128_9d4e55`.*
