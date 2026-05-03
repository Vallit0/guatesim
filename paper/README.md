# Constituciones Reveladas — el storytelling

> El paper en una línea: **construimos un instrumento para medir lo que un LLM
> *prefiere* cuando le toca gobernar, no lo que dice que prefiere.**

---
Bayesian Compositional Inference for Revealed-Preference Auditing of LLM Decision-Makers

## Apertura: el problema

Cada vez se usan más LLMs como soporte de decisión en políticas públicas,
asignación presupuestaria, recomendación regulatoria. Cuando dos modelos
producen *texto libre* sobre el mismo tema dan respuestas distintas — eso ya
lo sabemos. Lo que **no sabíamos** era: si los obligás a *elegir* entre
acciones cuantitativas con restricciones agregadas, bajo exactamente los
mismos shocks, ¿se promedian sus respuestas hacia un consenso, o cada modelo
trae una "constitución" implícita y estable?

No podemos abrirle el cráneo a un LLM y leerle los valores. Pero hay 90 años
de tradición económica diciendo que **no hace falta**: alcanza con mirar lo
que elige bajo restricción.

---

## Acto I — Samuelson (1938): no necesitás leer la mente

A fines de los 30, el concepto de "utilidad" en economía era un objeto
cuasi-místico — un estado interno del consumidor que nadie podía medir. Paul
Samuelson, con 23 años, publica una nota corta de 11 páginas que rompe el
problema de raíz:

> **Olvidate de la utilidad. Mirá lo que la persona elige cuando los precios
> y el ingreso cambian. Las preferencias se *revelan* en las elecciones bajo
> restricción presupuestaria.**

— Samuelson, P. A. (1938). *A Note on the Pure Theory of Consumer's
Behaviour*, **Economica**, 5(17), 61–71.
[doi:10.2307/2548836](https://doi.org/10.2307/2548836)

Esto es **revealed preference theory**. La extienden Houthakker (1950) con
el *Strong Axiom*, Afriat (1967) con la prueba constructiva (dado un set de
elecciones que satisfacen GARP, existe una función de utilidad que las
racionaliza), y Varian (1982) con la versión no-paramétrica computable.

- Houthakker (1950), *Revealed Preference and the Utility Function*, Economica.
- Afriat (1967), *The Construction of Utility Functions from Expenditure Data*, IER.
- Varian (1982), *The Nonparametric Approach to Demand Analysis*, Econometrica.

**La idea grande, en una frase:** las preferencias no son un secreto que hay
que sacar a la fuerza; son un patrón que emerge cuando observás suficientes
elecciones bajo suficientes restricciones distintas.

> Esta es exactamente la posición epistemológica del paper. El LLM toma 8
> decisiones presupuestarias, cada una bajo la restricción "tiene que sumar
> 100 %" + shocks distintos en cada turno. **Ese es el set de elecciones
> samuelsoniano.** No le preguntamos qué prefiere. Lo miramos elegir.

---

## Acto II — El problema inverso en IA (1960–2017)

Sesenta años después de Samuelson, la IA se topa con el mismo problema, pero
al revés.

**El problema directo (forward problem):** le das a un agente una función de
recompensa $R$, lo entrenás con RL, y observás emerger una política $\pi$.
Sutton & Barto, *Reinforcement Learning* (1998), es la biblia.

**El problema inverso (Inverse RL):** observás la política $\pi$ de un agente
y querés recuperar la función de recompensa $R$ que él *parece* estar
optimizando.

— Ng, A. Y., & Russell, S. J. (2000). *Algorithms for Inverse Reinforcement
Learning*. ICML 2000.
— Abbeel, P., & Ng, A. Y. (2004). *Apprenticeship Learning via Inverse
Reinforcement Learning*. ICML.
— Ziebart, B. D. et al. (2008). *Maximum Entropy Inverse Reinforcement
Learning*. AAAI.

Esto es Samuelson reescrito en notación de Bellman: en vez de "preferencias
reveladas por elecciones de consumo", "función de recompensa revelada por
trayectorias de acción".

**Y el escalón siguiente — el que más nos toca:**

— Hadfield-Menell, D., Milli, S., Abbeel, P., Russell, S., & Dragan, A.
(2017). *Inverse Reward Design*. **NeurIPS**.
[arXiv:1711.02827](https://arxiv.org/abs/1711.02827)

Inverse Reward Design dice: la recompensa que un humano *escribió* es
ruidosa, incompleta, está optimizada para el ambiente de entrenamiento. La
**verdadera** función objetivo del humano hay que inferirla bayesianamente a
partir de la recompensa proxy + el contexto en que la escribió. Es Samuelson
+ Bayes para el problema de alineamiento.

**Mechanism design inverso** es el primo más viejo: la teoría clásica
(Hurwicz, Myerson, Maskin — Nobel 2007) diseña reglas del juego para
*elicitar* un comportamiento deseado. La versión inversa pregunta: dado el
comportamiento que observamos en este agente bajo este conjunto de reglas,
¿qué función de bienestar social actúa como si estuviera maximizando?

- Hurwicz (1960), *Optimality and informational efficiency in resource allocation processes*.
- Myerson (1981), *Optimal Auction Design*, Mathematics of Operations Research.

> En `guatemala-sim` el "mecanismo" son las reglas del simulador (presupuesto
> suma 100, ranges fiscales, schema Pydantic). El LLM actúa dentro de ese
> mecanismo. Estamos haciendo **inverse mechanism design**: dado el mecanismo
> (fijo) y las decisiones (observadas), inferimos la función objetivo
> implícita del decisor.

---

## Acto III — Por qué esto es AI Safety

Esta es la parte que conecta todo lo anterior con la pregunta urgente del
2026.

Los LLMs de frontera no tienen una función de utilidad articulada. Fueron
entrenados con next-token prediction sobre texto humano, después afinados con
RLHF (Christiano et al. 2017, *Deep RL from Human Preferences*,
[arXiv:1706.03741](https://arxiv.org/abs/1706.03741)) y, en el caso de
Anthropic, con Constitutional AI (Bai et al. 2022, *Constitutional AI:
Harmlessness from AI Feedback*,
[arXiv:2212.08073](https://arxiv.org/abs/2212.08073)). Esos procesos dejan al
modelo con un set de **valores latentes** que ni el equipo que lo entrenó
conoce con precisión.

Hay dos formas de tratar de entender qué hay adentro:

**Vía 1 — Preferencias declaradas.** Le hacés preguntas al modelo y mirás
qué dice. Es lo que hace Anthropic con *Discovering Language Model Behaviors
with Model-Written Evaluations*:

— Perez, E., et al. (2022).
[arXiv:2212.09251](https://arxiv.org/abs/2212.09251)

Y más reciente, *Persona Vectors* (Anthropic, 2024) inspecciona
representaciones internas. Problema: las preferencias declaradas son fáciles
de gamear — el modelo dice lo que el evaluador quiere oír. Es la versión IA
del problema clásico de las encuestas.

**Vía 2 — Preferencias reveladas.** Le das al modelo una restricción dura,
lo obligás a elegir, y mirás qué hace. **Esto es lo que hacemos.** Es más
cerca de Samuelson, más cerca de IRL, más difícil de gamear porque el modelo
no está "respondiendo una pregunta sobre sí mismo" — está actuando.

La pregunta de AI Safety se vuelve concreta:

> Si reemplazás Claude Haiku por GPT-4o-mini en un pipeline de recomendación
> presupuestaria — porque es más barato, o porque el contrato venció —
> *¿qué se desplaza en el mundo?*
>
> Resultado del paper: del orden de **varios puntos porcentuales del PIB**
> entre salud y servicio de deuda. La elección del modelo no es neutral; es
> una decisión política implícita.

Esta es la sub-rama exacta que en literatura de Anthropic se llama
**scalable oversight** + **value identification**: necesitamos métodos
cuantitativos para auditar lo que un LLM va a hacer cuando se le delega
autoridad, antes de delegársela.

Russell, S. (2019). *Human Compatible: AI and the Problem of Control*. — el
manifiesto del campo.

---

## Acto IV — Por qué Dirichlet-Multinomial es el instrumento correcto

Acá entra la matemática. Tenemos los datos: para cada modelo $m$ y cada
turno $t = 1, \ldots, T$, el LLM produjo un vector

$$\mathbf{p}_t^{(m)} = (p_{1,t}, p_{2,t}, \ldots, p_{9,t}), \qquad p_{k,t} \geq 0, \quad \sum_{k=1}^{9} p_{k,t} = 1$$

donde $p_{k,t}$ es la fracción del presupuesto asignada a la partida $k$ en
el turno $t$ (salud, educación, seguridad, infraestructura, agro, protección
social, deuda, justicia, otros).

**Esto es *compositional data*** — datos que viven en el simplex
$\Delta^{K-1}$, no en $\mathbb{R}^K$. Aitchison (1986) escribió el libro
fundacional de cómo tratarlos:

— Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*.
Chapman & Hall.

¿Por qué importa que vivan en el simplex? Porque las partidas no son
independientes: si el LLM sube salud al 30 %, **algo más tiene que bajar**.
Tratar cada $p_k$ como una variable separada (regresión lineal por partida,
por ejemplo) ignora esa restricción y produce inferencias inconsistentes.

**La distribución Dirichlet es *la* distribución sobre el simplex.** Es la
generalización multivariada de la Beta:

$$\mathbf{p} \sim \text{Dir}(\boldsymbol\alpha), \quad \boldsymbol\alpha = (\alpha_1, \ldots, \alpha_K), \quad \alpha_k > 0$$

con densidad

$$f(\mathbf{p} \mid \boldsymbol\alpha) = \frac{\Gamma\left(\sum_k \alpha_k\right)}{\prod_k \Gamma(\alpha_k)} \prod_{k=1}^{K} p_k^{\alpha_k - 1}, \qquad \mathbf{p} \in \Delta^{K-1}$$

Sus dos propiedades que necesitamos:

1. **Esperanza:** $\mathbb{E}[p_k] = \alpha_k / \sum_j \alpha_j$ — la
   asignación esperada del LLM en la partida $k$.
2. **Concentración:** $\sum_k \alpha_k$ — cuanto **mayor**, más rígido el
   LLM (varía poco entre turnos); cuanto **menor**, más volátil.

Eso significa que el vector $\boldsymbol\alpha$ no codifica una sino **dos
cosas distintas**: *qué* prefiere el modelo y *qué tan dogmático* es al
respecto. Ningún promedio aritmético te da la segunda.

— Ferguson, T. S. (1973). *A Bayesian Analysis of Some Nonparametric
Problems*. **Annals of Statistics**, 1(2), 209–230.
([JSTOR](https://www.jstor.org/stable/2958008))
— Minka, T. P. (2000). *Estimating a Dirichlet distribution*. (Technical
note de MSR — la receta práctica de cómo ajustar.)

### El modelo jerárquico que está en `bayesian.py`

```
α_k ~ HalfNormal(σ = 10),    k = 1, ..., 9       # prior débil sobre concentración
p_t | α ~ Dirichlet(α),       t = 1, ..., T       # cada turno = una observación
```

Implementación en PyMC, líneas 340–350 de `guatemala_sim/bayesian.py`. Se
hace MCMC (NUTS), 2000 draws × 2 chains, y se devuelve el posterior conjunto
sobre $\boldsymbol\alpha$.

**Lo que extraemos del posterior:**

| Cantidad | Qué te dice | Cómo se usa en el paper |
|---|---|---|
| $\mathbb{E}[\alpha_k / \sum_j \alpha_j \mid \text{datos}]$ | media posterior de la asignación esperada por partida | ranking de prioridades del LLM (es lo que llamamos "constitución") |
| HDI95 sobre cada $\alpha_k / \sum_j \alpha_j$ | barra de error bayesiana | si los HDI de Claude y GPT no se solapan en una partida → diferencia robusta |
| $\mathbb{E}[\sum_k \alpha_k \mid \text{datos}]$ | concentración total | "rigidez constitucional": Claude alta = decide casi siempre lo mismo; baja = oscila |

### Por qué este modelo, no otro

- **Vs. promedio aritmético + desvío estándar:** el promedio te da
  $\hat{p}_k$ pero ignora la restricción del simplex y no separa "preferencia
  central" de "rigidez".
- **Vs. regresión multinomial logit:** un MNL te da los logits, sí, pero no
  tenés la concentración como segundo eje, y la inferencia del simplex queda
  implícita en el link.
- **Vs. Latent Dirichlet Allocation (Blei et al. 2003):** LDA es
  Dirichlet-multinomial *jerárquico con tópicos latentes* — pero acá no hay
  tópicos latentes, hay 9 categorías observadas. Es el caso de uso más
  limpio: una sola capa de Dirichlet.
- **Vs. tests frecuentistas (Wilcoxon, t-test):** los frecuentistas comparan
  **una métrica a la vez**. El Dirichlet-multinomial te da la **distribución
  conjunta** sobre las 9 partidas, respetando que están atadas entre sí.

Y la cosa que no se puede subestimar para un paper de AI Safety: el output
del modelo no es un *p*-valor binario "hay/no hay diferencia", es **el
posterior completo de la constitución**. Podés decir "hay 97 % de
probabilidad de que GPT-4o-mini prefiera salud por encima del 15 % del
presupuesto". Esa es una afirmación operativa, no un rechazo de hipótesis
nula.

— Kruschke, J. K. (2013). *Bayesian Estimation Supersedes the t test*
(BEST). **JEP: General**, 142(2), 573–603. — el otro modelo bayesiano del
repo, complementa al Dirichlet con comparaciones pareadas robustas.

---

## El pitch de 30 segundos

Si tenés que contar esto en un pasillo:

> En 1938, Samuelson dijo que no hace falta leer la mente de un consumidor
> para conocer sus preferencias: alcanza con mirarlo elegir bajo restricción
> presupuestaria. En 2000, Ng y Russell trasladaron la idea a la IA con
> Inverse Reinforcement Learning. En 2026, los LLMs de frontera empiezan a
> ser usados como decisores en políticas públicas, pero nadie sabe qué
> *prefieren* — el RLHF les dejó una "constitución" implícita que ni
> Anthropic puede leer directamente.
>
> Construimos un testbed donde dos LLMs (Claude y GPT-4o-mini) gobiernan la
> misma Guatemala simulada bajo idénticos shocks, durante 8 turnos, y deben
> asignar el presupuesto entre 9 partidas que suman 100 %. Eso es un
> experimento samuelsoniano puro: 8 elecciones bajo 8 restricciones
> distintas.
>
> Modelamos las elecciones con un Dirichlet-multinomial jerárquico
> bayesiano, porque los datos viven en el simplex y queremos recuperar tanto
> las preferencias centrales como su rigidez, con incertidumbre cuantificada.
>
> Resultado: las constituciones reveladas son distintas y reproducibles.
> Claude prioriza servicio de deuda (19 %) y protección social. GPT-4o-mini
> prioriza salud y educación (~18 % cada una). La diferencia equivale a
> varios puntos del PIB. **Reemplazar un modelo por otro en un pipeline de
> gobernanza no es una decisión técnica — es una decisión política
> implícita, y nuestro instrumento la hace medible.**

---

## Lecturas en orden, si querés profundizar

1. **Samuelson 1938** (10 páginas, lo lee en una tarde). Es el fundamento
   conceptual.
2. **Varian 1982** para la versión computacional moderna de revealed
   preference.
3. **Ng & Russell 2000** para ver el puente a IA — paper corto, técnico
   pero accesible.
4. **Hadfield-Menell et al. 2017 (Inverse Reward Design)** para el ángulo
   de alineamiento. Este es el más cercano en espíritu a lo que estás
   haciendo.
5. **Bai et al. 2022 (Constitutional AI)** para entender de dónde sale la
   palabra "constitución" en este contexto.
6. **Aitchison 1986** (libro) si querés la teoría de compositional data en
   serio. Capítulos 1–3 son suficientes.
7. **Kruschke 2013 (BEST)** para la filosofía del análisis bayesiano sobre
   el frecuentista.
