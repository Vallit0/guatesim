# Diagnóstico de capacidad estructurada: `qwen2.5:0.5b`

**N = 20 llamadas** al mismo estado inicial (enero 2026), sobre el schema `DecisionTurno` vía OpenAI-compat API.

## Resumen

- **Tasa de éxito (valida schema completo): 0/20 = 0.0%**
- Latencia media: 21.63 s
- Latencia min/max: 1.79 / 90.25 s

## Distribución de modos de fallo

| Categoría | N | % |
|---|--:|--:|
| `schema_erroneo` | 10 | 50.0% |
| `rangos_fuera` | 5 | 25.0% |
| `campos_faltantes` | 3 | 15.0% |
| `json_invalido` | 1 | 5.0% |
| `otro_error_validacion` | 1 | 5.0% |

### Taxonomía

- `valid`: salida respeta el schema Pydantic completo.
- `json_invalido`: no es JSON parseable.
- `schema_erroneo`: JSON válido pero ninguna key coincide con el schema.
- `campos_faltantes`: algunos campos del schema, pero incompleto (< 4 de 7).
- `presupuesto_no_suma`: presupuesto presente pero Σ partidas ≠ 100 ±1.
- `rangos_fuera`: valores numéricos fuera del dominio Pydantic.
- `otro_error_validacion`: cualquier otra falla de Pydantic.
- `http_error`: fallo del transporte (timeout, conexión, etc.).

## Ejemplos por categoría

### `json_invalido` (intento #0, 90.25s)

Detalle: `Expecting property name enclosed in double quotes: line 219 column 4 (char 8429)`

```json
{
  "razonamiento": "La decisión tomada por el presidente en 2026-01-01 para un turno es relevante, ya que implica cambios importantes en el presupuesto, políticas exteriores, reformas estructurales y responder a los shockes de una crisis intransigente entre los partidos, la libertad prensa y la apertura multilateral. Los presupuestos deben sumar 100%, las políticas externas se ajustan al nuevo modelo bilateral, mientras que las reformas en el sector fiscal y estructura del gobierno promueven la transparencia y equilibrio", 
  "presupuesto": {
    "$ref": "#/$defs/PresupuestoAnual",
    "@context": "https://schema.org",
    "@type": "PolíticaEjecutiva",
    "desc": "Asignación porcentual del gasto público total. Debe sumar ~100.",
    "id": "",
    "amount": {
      "$ref": "#/$defs/Presup...
```

### `schema_erroneo` (intento #1, 13.05s)

Detalle: `keys inventadas: ['$defs', 'required', 'title', 'type']`

```json
{
  "$defs": {
    "Fiscal": {
      "delta_iva_pp": -5.0,
      "delta_isr_pp": 10.0,
      "aranceles_especificos": "",
      "alineamiento_priorizado": "eeuu",
      "acciones_diplomaticas": ["EE.UU.", "China", "México"]
    },
    "PoliticaExterior": {
      "alineamiento_priorificado": "multilateral",
      "acciones_diplomaticas": []
    },
    "PresupuestoAnual": {
      "salud": 100.0,
      "educacion": 100.0,
      "seguridad": 100.0,
      "infraestructura": 100.0,
      "agro_desarrollo_rural": 100.0,
      "proteccion_social": 100.0,
      "servicio_deuda": 100.0,
      "justicia": 100.0,
      "otros": 100.0
    },
    "Reforma": {
      "area": "catastro",
      "costo_politico": 0.0,
      "intensidad": "",
      "medida": ""
    }
  },
  "required": ["salud", "educacion", ...
```

### `otro_error_validacion` (intento #2, 9.97s)

Detalle: `16 validation errors for DecisionTurno`

```json
{"mensaje_al_pueblo": "La fiscalidad no se considera una prioridad para el presupuesto", "presupuesto": {"10.000": 4}, "fiscal": {"delta_iva_pp": 5, "delta_isr_pp": -10}, "exterior": {"alineamiento_priorizado": ["eeuu"]}, "respuestas_shocks": {"shock": "decrecimiento de la defrostación y el ahorro", "medida": "implementar revaluaciones de ingresos y aumentar el PIB para incrementar la eficiencia del presupuesto.", "costo_fiscal_pib": 3.50}, "reformas": {"area": "tributaria", "intensidad": "regional", "costo_politico": 10.4}, "Respuesta shock": {"shock": "decrecimiento en la prestación de servicios públicos", "medida": "realizar reformas en salud y educación para reducir el costo de las mismas, al mismo tiempo que aumentar la infraestructura pública", "costo_fiscal_pib": 30.4}, "decision_tu...
```

### `rangos_fuera` (intento #6, 10.87s)

Detalle: `razonamiento`

```json
{"presupuesto": "#9795", "fiscal": {"delta_iva_pp": 6.80, "delta_isr_pp": -3.20}, "exterior": {"alineamiento_priorizado": "eeuu", "acciones_diplomaticas": ["EE.UU.", "China", "México"]}, "respuestas_shocks": {"shock": "Reformas de Justicia", "medida": "Toma medidas para afrontar los desafíos del sistema político y la presión interna", "costo_fiscal_pib": 400}, "reformas": [{"area": "Catastro", "intensidad": "radical", "costo_politico": 650}, {"area": "Servicio Científico", "intensidad": "incremental", "costo_politico": 175}, {"area": "Justicia", "intensidad": "incremental", "costo_politico": 95}, {"area": "Triángulo Norte", "intensidad": "radical", "costo_politico": 0}], "mensaje_al_pueblo": "Para controlar la pobreza y mantener el sistema político, se implementará una reforma de justicia ...
```

### `campos_faltantes` (intento #12, 10.62s)

Detalle: `presentes=['mensaje_al_pueblo', 'razonamiento'] faltan=['exterior', 'fiscal', 'presupuesto', 'reformas', 'respuestas_shocks']`

```json
{"Fiscal":{"delta_iva_pp":0.5,"delta_isr_pp":3.5,"aranceles_especificos":"general","politica_exterior":false},"PoliticaExterior":{"alineamiento_priorizado":"regionale_comun","acciones_diplomaticas":"reunión_con_EU"},"PresupuestoAnual":{"salud":0, "educacion":18, "seguridad":23,"infraestructura":15, "agro_desarrollo_rural":2, "proteccion_social":0.5,"servicio_deuda":0, "justicia": 0,"otros":1},"Reforma":{"area":"Servicio Civil","costo_politico":"sustancial"},"RespuestaShock":{"shock":"Pobreza media ponderada aumentará con la política exterior pero no será un shock lógico. Esto se debía a la decisión de apertura multilateral y la resolución del desenlace (suficiencia)"},"DecisionTurno":"Declarar estrategia estructural para el desarrollo local alineando recursos con políticas externas y con l...
```

## Implicancia para el paper

El modelo `qwen2.5:0.5b` es **incapaz** de producir decisiones ejecutivas válidas bajo restricción de schema en este setup. La hipótesis de trabajo es que la capacidad de seguir instrucciones de schema complejas escala con parámetros; 0.5B está por debajo del umbral práctico.

Datos completos (outputs crudos): `raw_outputs.jsonl` en este directorio.
