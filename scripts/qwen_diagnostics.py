"""Diagnóstico formal de la capacidad de Qwen 0.5b para seguir el schema.

Corre N llamadas contra el mismo estado inicial, captura cada output crudo,
lo clasifica en una taxonomía de fallos, y escribe un reporte markdown
paper-ready.

Uso:
    python qwen_diagnostics.py --url http://192.168.0.17:11434/v1 \\
                               --modelo qwen2.5:0.5b --n 20

Categorías de output:
    valid           → pasa validación Pydantic (éxito)
    json_invalido   → ni siquiera es JSON parseable
    schema_erroneo  → JSON válido pero sin campos del schema
    campos_faltantes → JSON con algunos campos del schema
    presupuesto_no_suma → todo bien salvo que presupuesto != 100
    rangos_fuera    → todo bien salvo rangos numéricos inválidos
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from guatemala_sim.actions import DecisionTurno
from guatemala_sim.bootstrap import initial_state
from guatemala_sim.president import SYSTEM_PROMPT, build_context
from guatemala_sim.world.territory import Territory


CAMPOS_SCHEMA = {"razonamiento", "presupuesto", "fiscal", "exterior",
                 "respuestas_shocks", "reformas", "mensaje_al_pueblo"}
PARTIDAS_PRESUPUESTO = {"salud", "educacion", "seguridad", "infraestructura",
                        "agro_desarrollo_rural", "proteccion_social",
                        "servicio_deuda", "justicia", "otros"}


@dataclass
class Intento:
    n: int
    latencia_s: float
    raw: str
    categoria: str
    detalle: str = ""


def clasificar(raw: str) -> tuple[str, str]:
    """Clasifica la respuesta cruda en una categoría de éxito/fallo."""
    raw_s = (raw or "").strip()
    if not raw_s:
        return "json_invalido", "respuesta vacía"
    # intento parsear JSON
    try:
        obj = json.loads(raw_s)
    except json.JSONDecodeError as e:
        return "json_invalido", str(e)[:120]
    if not isinstance(obj, dict):
        return "json_invalido", f"no es objeto raíz (tipo={type(obj).__name__})"

    campos_presentes = set(obj.keys()) & CAMPOS_SCHEMA
    # ¿tiene los campos principales?
    if not campos_presentes:
        return "schema_erroneo", f"keys inventadas: {list(obj.keys())[:5]}"
    if len(campos_presentes) < 4:
        return "campos_faltantes", (
            f"presentes={sorted(campos_presentes)} "
            f"faltan={sorted(CAMPOS_SCHEMA - campos_presentes)}"
        )

    # intento validar con Pydantic
    try:
        DecisionTurno.model_validate(obj)
        return "valid", ""
    except Exception as e:
        err = str(e)
        # refinar: presupuesto no suma o rangos
        if "Presupuesto debe sumar" in err:
            p = obj.get("presupuesto", {})
            total = sum(v for k, v in p.items() if isinstance(v, (int, float)))
            return "presupuesto_no_suma", f"suma={total:.1f}"
        if "less than or equal" in err or "greater than or equal" in err:
            return "rangos_fuera", err.split("\n")[1][:120] if "\n" in err else err[:120]
        return "otro_error_validacion", err.split("\n")[0][:150]


def correr_diagnostico(
    base_url: str, modelo: str, api_key: str, n: int, out_dir: Path
) -> list[Intento]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    territory = Territory.load_default()
    contexto_base = build_context(
        initial_state(),
        territory_summary=territory.summary().as_dict(),
    )
    # system prompt con schema (modo json_object loose)
    schema_str = json.dumps(DecisionTurno.model_json_schema(), ensure_ascii=False)
    system = (
        SYSTEM_PROMPT
        + "\n\nDEBES responder UN SOLO objeto JSON válido (sin markdown) "
        "que respete este schema:\n" + schema_str
        + "\n\nReglas: presupuesto debe sumar 100 (±1); fiscal.delta_iva_pp "
        "en [-5, 5]; fiscal.delta_isr_pp en [-10, 10]; reformas ≤ 2."
    )

    intentos: list[Intento] = []
    for i in range(n):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=modelo,
                max_tokens=2500,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": contexto_base},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            raw = ""
            intentos.append(Intento(
                n=i, latencia_s=time.time() - t0, raw="",
                categoria="http_error", detalle=str(e)[:200],
            ))
            print(f"[{i:02d}] HTTP ERROR: {str(e)[:80]}")
            continue
        cat, det = clasificar(raw)
        lat = time.time() - t0
        intentos.append(Intento(
            n=i, latencia_s=lat, raw=raw, categoria=cat, detalle=det,
        ))
        print(f"[{i:02d}] {lat:5.2f}s  {cat:20s}  {det[:80]}")
    return intentos


def escribir_reporte(
    intentos: list[Intento], modelo: str, out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # guardar raw outputs para inspección manual
    raws_path = out_dir / "raw_outputs.jsonl"
    with raws_path.open("w", encoding="utf-8") as f:
        for it in intentos:
            f.write(json.dumps({
                "n": it.n, "latencia_s": round(it.latencia_s, 3),
                "categoria": it.categoria, "detalle": it.detalle,
                "raw": it.raw,
            }, ensure_ascii=False) + "\n")

    cnt = Counter(it.categoria for it in intentos)
    total = len(intentos)
    exito = cnt.get("valid", 0)
    tasa_exito = 100.0 * exito / total if total else 0.0
    latencias = [it.latencia_s for it in intentos]
    lat_media = sum(latencias) / len(latencias) if latencias else 0.0

    # un ejemplo representativo por categoría
    ejemplos = {}
    for it in intentos:
        if it.categoria not in ejemplos:
            ejemplos[it.categoria] = it

    md = out_dir / "findings.md"
    with md.open("w", encoding="utf-8") as f:
        f.write(f"# Diagnóstico de capacidad estructurada: `{modelo}`\n\n")
        f.write(f"**N = {total} llamadas** al mismo estado inicial (enero 2026), ")
        f.write(f"sobre el schema `DecisionTurno` vía OpenAI-compat API.\n\n")
        f.write(f"## Resumen\n\n")
        f.write(f"- **Tasa de éxito (valida schema completo): {exito}/{total} = {tasa_exito:.1f}%**\n")
        f.write(f"- Latencia media: {lat_media:.2f} s\n")
        f.write(f"- Latencia min/max: {min(latencias):.2f} / {max(latencias):.2f} s\n\n")
        f.write(f"## Distribución de modos de fallo\n\n")
        f.write("| Categoría | N | % |\n|---|--:|--:|\n")
        for cat, c in cnt.most_common():
            f.write(f"| `{cat}` | {c} | {100*c/total:.1f}% |\n")
        f.write("\n")
        f.write("### Taxonomía\n\n")
        f.write("- `valid`: salida respeta el schema Pydantic completo.\n")
        f.write("- `json_invalido`: no es JSON parseable.\n")
        f.write("- `schema_erroneo`: JSON válido pero ninguna key coincide con el schema.\n")
        f.write("- `campos_faltantes`: algunos campos del schema, pero incompleto (< 4 de 7).\n")
        f.write("- `presupuesto_no_suma`: presupuesto presente pero Σ partidas ≠ 100 ±1.\n")
        f.write("- `rangos_fuera`: valores numéricos fuera del dominio Pydantic.\n")
        f.write("- `otro_error_validacion`: cualquier otra falla de Pydantic.\n")
        f.write("- `http_error`: fallo del transporte (timeout, conexión, etc.).\n\n")
        f.write(f"## Ejemplos por categoría\n\n")
        for cat, it in ejemplos.items():
            f.write(f"### `{cat}` (intento #{it.n}, {it.latencia_s:.2f}s)\n\n")
            f.write(f"Detalle: `{it.detalle or '—'}`\n\n")
            f.write("```json\n")
            snippet = it.raw[:800] + ("..." if len(it.raw) > 800 else "")
            f.write(snippet + "\n")
            f.write("```\n\n")
        f.write(f"## Implicancia para el paper\n\n")
        if tasa_exito == 0.0:
            f.write(
                f"El modelo `{modelo}` es **incapaz** de producir decisiones "
                f"ejecutivas válidas bajo restricción de schema en este setup. "
                f"La hipótesis de trabajo es que la capacidad de seguir "
                f"instrucciones de schema complejas escala con parámetros; "
                f"0.5B está por debajo del umbral práctico.\n\n"
            )
        elif tasa_exito < 50.0:
            f.write(
                f"El modelo `{modelo}` valida en menos de la mitad de los "
                f"intentos ({tasa_exito:.1f}%), lo que lo hace **no apto** "
                f"para gobernanza estructurada sin un loop de reintentos costoso.\n\n"
            )
        else:
            f.write(
                f"El modelo `{modelo}` valida en {tasa_exito:.1f}% de los "
                f"intentos — marginalmente usable con reintentos.\n\n"
            )
        f.write(
            "Datos completos (outputs crudos): `raw_outputs.jsonl` en este "
            "directorio.\n"
        )
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://192.168.0.17:11434/v1")
    ap.add_argument("--modelo", default="qwen2.5:0.5b")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else (
        root / "figures" / f"qwen_diag_{args.modelo.replace(':', '_').replace('/', '_')}"
    )
    print(f"[diag] modelo={args.modelo} url={args.url} n={args.n}")
    print(f"[diag] out={out_dir}")
    intentos = correr_diagnostico(args.url, args.modelo, args.api_key, args.n, out_dir)
    md = escribir_reporte(intentos, args.modelo, out_dir)
    print(f"\n[diag] reporte: {md}")


if __name__ == "__main__":
    main()
