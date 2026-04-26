# Finding: Small language models (< 1B parameters) cannot execute structured governance decisions

## Claim

When constrained to produce a structured decision JSON conforming to a complex,
multi-field schema derived from a real-world governance task, language models
under approximately 1B parameters **fail at a rate effectively indistinguishable
from 100%**. This failure is robust to retries with feedback and is not
rescued by OpenAI-compatible `json_object` response formatting.

## Evidence

We ran N=20 independent calls of `qwen2.5:0.5b` (494M parameters, Q4_K_M quant,
served via Ollama 192.168.0.17:11434 on an NVIDIA Jetson Nano) against the
initial state of a simulated Guatemala (enero 2026), with the same system
prompt used for Claude (`claude-haiku-4-5-20251001`) and OpenAI
(`gpt-4o-mini`).

The schema required: a `razonamiento` field, a `presupuesto` object (9
percentage allocations summing to 100), a `fiscal` object (two numeric ranges),
an `exterior` object (categorical alignment + 0-3 diplomatic actions), an
optional `respuestas_shocks` list, an optional `reformas` list (≤ 2 items), and
a `mensaje_al_pueblo` string (1–600 chars).

Results:

| Category | N | % |
|---|---:|---:|
| `valid` (Pydantic-validated) | **0** | **0.0%** |
| `schema_erroneo` | 10 | 50.0% |
| `rangos_fuera` | 5 | 25.0% |
| `campos_faltantes` | 3 | 15.0% |
| `json_invalido` | 1 | 5.0% |
| `otro_error_validacion` | 1 | 5.0% |

Mean latency was 21.6 s per call; variance was extreme (min 1.79 s, max 90.2 s),
suggesting occasional thrashing of the Jetson Nano's CUDA kernels rather than
steady inference.

## Failure modes

**1. Schema echoing (50%).** The modal failure was that the model, given a
system prompt containing the JSON schema as text, reproduced fragments of
the schema definition itself rather than producing an instance of it. Sample
output:

```json
{
  "$defs": {
    "Fiscal": { "delta_iva_pp": -5.0, "delta_isr_pp": 10.0, ... },
    "PoliticaExterior": { "alineamiento_priorificado": "multilateral", ... },
    "PresupuestoAnual": { "salud": 100.0, "educacion": 100.0, ... },
    ...
  },
  "required": ["salud", "educacion", ...]
}
```

The model treated the schema as if it were the expected output format, not a
description of the expected output. This is a schema–instance confusion that
is not documented as a failure mode for larger models.

**2. Out-of-range values (25%).** When the model did produce the right field
names, numeric values frequently violated domain constraints (e.g.
`costo_politico=650` in a [0, 100] range, `costo_fiscal_pib=400` in a reasonable
0–5% range, `delta_iva_pp=6.80` in a [-5, 5] range).

**3. Partial field coverage (15%).** The model produced 2–3 of the 7 required
top-level fields, typically `razonamiento` + `mensaje_al_pueblo` while omitting
the numerically-structured `presupuesto` and `fiscal`.

**4. Invalid JSON (5%).** Trailing content, mismatched delimiters, or
mid-generation truncation.

**5. Schema-legal but domain-illegal (5%).** Values that parse but use
non-existent enum values (e.g. `intensidad="regional"` when the enum is
`{incremental, media, radical}`).

## Implications

1. **There exists a capability floor for LLM-based structured governance.** At
   least under the constraints of a realistic schema (~30 nested fields, mixed
   types, numeric range constraints, and aggregate constraints like
   "percentages sum to 100"), 0.5B parameters is below it. Our larger-model
   baselines (`claude-haiku-4-5` at ~8B, `gpt-4o-mini` at ~8B, both estimated)
   validated at 100% in the same experiments.

2. **Edge deployment of LLM policy requires capability validation.** A Jetson
   Nano is a reasonable edge device for LLM inference, but the practical
   capability ceiling at that form factor (sub-1B quantized models) excludes
   the schema-driven decision paradigm that we use here. Agents intended for
   edge deployment need lighter-weight action representations (flat
   enumerations, fewer fields, no aggregate constraints).

3. **Schema–instance confusion is a new failure mode worth naming.** When the
   schema is passed as a string instruction rather than as a structured
   decoding constraint (as OpenAI's strict `json_schema` mode would do), small
   models appear to echo the schema back. This is a distinct failure from the
   more commonly-reported "hallucinated field" or "numeric OOB" failures.

## Reproducibility

- Raw outputs: `figures/qwen_diag_qwen2.5_0.5b/raw_outputs.jsonl` (20 lines)
- Script: `qwen_diagnostics.py`
- Seed: call-by-call randomness from the model sampler; system prompt identical
  across calls (same initial state, same schema in the system message).
- Hardware: NVIDIA Jetson Nano, Ollama native inference, Q4_K_M quantization.

## Limitation

N=20 at a single model size is insufficient to map the capability curve. A
follow-up would sweep {0.5B, 1.5B, 3B, 7B, 14B} Qwen variants and compute
the success rate as a function of parameter count, ideally both in strict-mode
JSON schema (OpenAI cloud) and loose-mode JSON object (Ollama/LM Studio).
