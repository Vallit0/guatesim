"""Logging estructurado JSONL + consola `rich`.

Cada turno se serializa como una línea JSON en `runs/{run_id}.jsonl`
conteniendo el record completo (state antes, decisión, state después,
shocks, indicadores).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import sys

from rich.console import Console
from rich.table import Table

from .engine import TurnRecord
from .indicators import compute_indicators
from .state import GuatemalaState

# En Windows la consola por defecto es cp1252 y rompe con símbolos unicode.
# Forzamos UTF-8 en stdout antes de construir el Console.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

_console = Console(force_terminal=True, legacy_windows=False)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


class JsonlLogger:
    """Un logger JSONL por corrida. `with` cierra el archivo."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, record: TurnRecord) -> None:
        # Calcular indicadores para el state_after
        state_after = GuatemalaState.model_validate(record.state_after)
        ind = compute_indicators(state_after)
        line = {
            "t": record.t,
            "fecha": record.fecha_iso,
            "shocks": record.shocks_activos,
            "decision": record.decision,
            "state_before": record.state_before,
            "state_after": record.state_after,
            "indicadores": ind.as_dict(),
        }
        self._fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_run(path: Path) -> list[dict[str, Any]]:
    """Lee un archivo JSONL de corrida."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


# --- UI de terminal ----------------------------------------------------------


def print_turn_resumen(record: TurnRecord) -> None:
    """Imprime un resumen lindo del turno en la consola."""
    state_after = GuatemalaState.model_validate(record.state_after)
    ind = compute_indicators(state_after)
    t = record.t
    table = Table(
        title=f"turno t={t} ({record.fecha_iso})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("indicador")
    table.add_column("valor", justify="right")
    table.add_row("PIB (mm USD)", f"{state_after.macro.pib_usd_mm:,.0f}")
    table.add_row("crecimiento", f"{state_after.macro.crecimiento_pib:+.2f}%")
    table.add_row("inflación", f"{state_after.macro.inflacion:.2f}%")
    table.add_row("deuda/PIB", f"{state_after.macro.deuda_pib:.1f}%")
    table.add_row("pobreza", f"{state_after.social.pobreza_general:.1f}%")
    table.add_row("migración neta (miles)", f"{state_after.social.migracion_neta_miles:+.0f}")
    table.add_row("aprobación", f"{state_after.politico.aprobacion_presidencial:.1f}")
    table.add_row("bienestar", f"{ind.bienestar:.1f}")
    table.add_row("gobernabilidad", f"{ind.gobernabilidad:.1f}")
    table.add_row("estabilidad macro", f"{ind.estabilidad_macro:.1f}")
    table.add_row("estrés social", f"{ind.estres_social:.1f}")
    if record.shocks_activos:
        table.add_row("[red]shocks[/red]", ", ".join(record.shocks_activos))
    _console.print(table)


def print_corrida_resumen(records: Iterable[dict[str, Any]]) -> None:
    """Resumen agregado de una corrida completa."""
    records = list(records)
    if not records:
        _console.print("[yellow]corrida vacía[/yellow]")
        return
    t_ini = records[0]
    t_fin = records[-1]
    s0 = GuatemalaState.model_validate(t_ini["state_before"])
    s1 = GuatemalaState.model_validate(t_fin["state_after"])
    ind0 = compute_indicators(s0)
    ind1 = compute_indicators(s1)
    table = Table(
        title=f"resumen de corrida ({len(records)} turnos)",
        show_header=True,
    )
    table.add_column("indicador")
    table.add_column("inicio", justify="right")
    table.add_column("fin", justify="right")
    table.add_column("delta", justify="right")

    rows = [
        ("PIB (mm USD)", s0.macro.pib_usd_mm, s1.macro.pib_usd_mm),
        ("pobreza %", s0.social.pobreza_general, s1.social.pobreza_general),
        ("homicidios/100k", s0.social.homicidios_100k, s1.social.homicidios_100k),
        ("migración neta", s0.social.migracion_neta_miles, s1.social.migracion_neta_miles),
        ("aprobación", s0.politico.aprobacion_presidencial, s1.politico.aprobacion_presidencial),
        ("bienestar", ind0.bienestar, ind1.bienestar),
        ("gobernabilidad", ind0.gobernabilidad, ind1.gobernabilidad),
        ("IDH proxy", ind0.desarrollo_humano, ind1.desarrollo_humano),
        ("estabilidad", ind0.estabilidad_macro, ind1.estabilidad_macro),
        ("estrés", ind0.estres_social, ind1.estres_social),
    ]
    for name, a, b in rows:
        delta = b - a
        color = "green" if delta >= 0 else "red"
        if name in ("pobreza %", "homicidios/100k", "estrés"):
            color = "red" if delta >= 0 else "green"
        table.add_row(name, f"{a:,.1f}", f"{b:,.1f}", f"[{color}]{delta:+,.1f}[/]")
    _console.print(table)
