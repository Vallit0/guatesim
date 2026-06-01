"""Comparación entre corridas (p. ej. Claude vs GPT).

Cada "corrida" es un JSONL producido por `JsonlLogger`. Este módulo:
  - alinea los turnos por t
  - computa métricas pareadas (PIB, pobreza, aprobación, índices compuestos)
  - computa métricas constitucionales por corrida (coherencia, entropía, radar)
  - genera gráficas overlay (líneas comparando trayectorias)
  - genera un reporte markdown con tabla comparativa
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # backend headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .indicators import (
    coherencia_temporal,
    compute_indicators,
    diversidad_valores,
    resumen_presupuesto,
)
from .logging_ import read_run
from .state import GuatemalaState


@dataclass
class CorridaEtiquetada:
    label: str
    records: list[dict]

    @classmethod
    def from_path(cls, label: str, path: Path) -> "CorridaEtiquetada":
        return cls(label=label, records=read_run(Path(path)))


def _serie(records: list[dict], fn) -> pd.Series:
    """Aplica fn a cada record y devuelve una Serie indexada por t."""
    idx = [r["t"] for r in records]
    vals = [fn(r) for r in records]
    return pd.Series(vals, index=idx)


def _ind(records: list[dict], nombre: str) -> pd.Series:
    return _serie(records, lambda r: r["indicadores"][nombre])


def _state_attr(records: list[dict], dotted: str) -> pd.Series:
    """Ej: 'macro.pib_usd_mm' -> Serie."""
    def get(r):
        obj = r["state_after"]
        for k in dotted.split("."):
            obj = obj[k]
        return obj
    return _serie(records, get)


# --- métricas ----------------------------------------------------------------


def tabla_comparativa(corridas: Sequence[CorridaEtiquetada]) -> pd.DataFrame:
    """Tabla pareada: fila por métrica, columna por corrida."""
    rows = []
    for c in corridas:
        recs = c.records
        s_ini = GuatemalaState.model_validate(recs[0]["state_before"])
        s_fin = GuatemalaState.model_validate(recs[-1]["state_after"])
        ind_ini = compute_indicators(s_ini)
        ind_fin = compute_indicators(s_fin)
        decisiones = [r["decision"] for r in recs]
        presupuesto_avg = resumen_presupuesto(decisiones)
        rows.append({
            "corrida": c.label,
            "turnos": len(recs),
            "shocks_totales": sum(len(r.get("shocks", [])) for r in recs),
            "PIB_ini": s_ini.macro.pib_usd_mm,
            "PIB_fin": s_fin.macro.pib_usd_mm,
            "PIB_delta": s_fin.macro.pib_usd_mm - s_ini.macro.pib_usd_mm,
            "pobreza_ini": s_ini.social.pobreza_general,
            "pobreza_fin": s_fin.social.pobreza_general,
            "pobreza_delta": s_fin.social.pobreza_general - s_ini.social.pobreza_general,
            "aprobacion_ini": s_ini.politico.aprobacion_presidencial,
            "aprobacion_fin": s_fin.politico.aprobacion_presidencial,
            "deuda_fin": s_fin.macro.deuda_pib,
            "bienestar_ini": ind_ini.bienestar,
            "bienestar_fin": ind_fin.bienestar,
            "gobernabilidad_fin": ind_fin.gobernabilidad,
            "estabilidad_fin": ind_fin.estabilidad_macro,
            "idh_fin": ind_fin.desarrollo_humano,
            "estres_fin": ind_fin.estres_social,
            "coherencia_temporal": coherencia_temporal(decisiones),
            "diversidad_valores": diversidad_valores(decisiones),
            "reformas_totales": sum(len(d.get("reformas", [])) for d in decisiones),
            "reformas_radicales": sum(
                1 for d in decisiones for r in d.get("reformas", []) if r["intensidad"] == "radical"
            ),
            "delta_iva_medio": float(np.mean([d["fiscal"]["delta_iva_pp"] for d in decisiones])),
            "delta_isr_medio": float(np.mean([d["fiscal"]["delta_isr_pp"] for d in decisiones])),
            **{f"presup_{k}": v for k, v in presupuesto_avg.items()},
        })
    return pd.DataFrame(rows).set_index("corrida")


# --- gráficas ----------------------------------------------------------------


_COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def overlay_trayectorias(
    corridas: Sequence[CorridaEtiquetada], out_dir: Path
) -> Path:
    """4 subplots con overlay de las corridas: PIB, pobreza, aprobación, bienestar."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = axes.ravel()
    for i, c in enumerate(corridas):
        color = _COLORES[i % len(_COLORES)]
        r = c.records
        axes[0].plot(_state_attr(r, "macro.pib_usd_mm"), lw=2, color=color, label=c.label)
        axes[1].plot(_state_attr(r, "social.pobreza_general"), lw=2, color=color, label=c.label)
        axes[2].plot(_state_attr(r, "politico.aprobacion_presidencial"), lw=2, color=color, label=c.label)
        axes[3].plot(_ind(r, "bienestar"), lw=2, color=color, label=c.label)
    axes[0].set_title("PIB (mm USD)"); axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].set_title("Pobreza (%)"); axes[1].grid(True, alpha=0.3); axes[1].legend()
    axes[2].set_title("Aprobación presidencial"); axes[2].grid(True, alpha=0.3); axes[2].legend()
    axes[3].set_title("Bienestar (0-100)"); axes[3].grid(True, alpha=0.3); axes[3].legend()
    for ax in axes:
        ax.set_xlabel("turno")
    fig.suptitle("Comparativa de corridas", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "comparativa_trayectorias.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def overlay_indicadores_compuestos(
    corridas: Sequence[CorridaEtiquetada], out_dir: Path
) -> Path:
    """5 subplots de índices compuestos — una serie por corrida por subplot."""
    indices = ["bienestar", "gobernabilidad", "desarrollo_humano",
               "estabilidad_macro", "estres_social"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()
    for j, ind_name in enumerate(indices):
        ax = axes[j]
        for i, c in enumerate(corridas):
            color = _COLORES[i % len(_COLORES)]
            ax.plot(_ind(c.records, ind_name), lw=2, color=color, label=c.label)
        ax.set_title(ind_name)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    for ax in axes[:5]:
        ax.set_xlabel("turno")
    fig.suptitle("Índices compuestos por corrida", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "comparativa_indices.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def radar_comparativo(
    corridas: Sequence[CorridaEtiquetada], out_dir: Path
) -> Path:
    """Radar del presupuesto promedio — un polígono por corrida, superpuestos."""
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    cats: list[str] = []
    for i, c in enumerate(corridas):
        avg = resumen_presupuesto([r["decision"] for r in c.records])
        if not cats:
            cats = list(avg.keys())
        vals = [avg.get(k, 0.0) for k in cats]
        vals = vals + vals[:1]
        angs = np.linspace(0, 2 * np.pi, len(cats) + 1)
        color = _COLORES[i % len(_COLORES)]
        ax.plot(angs, vals, lw=2.5, color=color, label=c.label)
        ax.fill(angs, vals, alpha=0.20, color=color)
    ax.set_xticks(np.linspace(0, 2 * np.pi, len(cats), endpoint=False))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_title("Valores revelados: presupuesto promedio por decisor\n",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    out = out_dir / "comparativa_radar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def barras_metricas_llm(
    corridas: Sequence[CorridaEtiquetada], out_dir: Path
) -> Path:
    """Barras agrupadas: coherencia, entropía (x30), reformas, reformas radicales."""
    labels = [c.label for c in corridas]
    coh = [coherencia_temporal([r["decision"] for r in c.records]) for c in corridas]
    div = [diversidad_valores([r["decision"] for r in c.records]) * 30 for c in corridas]
    reformas = [sum(len(d.get("reformas", [])) for d in [r["decision"] for r in c.records]) for c in corridas]
    radicales = [
        sum(1 for d in [r["decision"] for r in c.records]
            for rr in d.get("reformas", [])
            if rr["intensidad"] == "radical")
        for c in corridas
    ]
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, coh, w, label="coherencia temporal", color="#2ca02c")
    ax.bar(x - 0.5 * w, div, w, label="diversidad × 30", color="#9467bd")
    ax.bar(x + 0.5 * w, reformas, w, label="reformas totales", color="#1f77b4")
    ax.bar(x + 1.5 * w, radicales, w, label="reformas radicales", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Métricas constitucionales por decisor", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "comparativa_metricas_llm.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --- reporte markdown --------------------------------------------------------


def reporte_markdown(
    corridas: Sequence[CorridaEtiquetada], out_path: Path
) -> Path:
    df = tabla_comparativa(corridas)
    lines = ["# Comparativa de decisores", ""]
    lines.append(f"Corridas comparadas: {', '.join(c.label for c in corridas)}.")
    lines.append(f"Horizonte: {df['turnos'].iloc[0]} turnos.")
    lines.append("")
    lines.append("## Indicadores macro y sociales (fin del horizonte)")
    lines.append("")
    keys_macro = ["PIB_fin", "PIB_delta", "pobreza_fin", "pobreza_delta",
                  "aprobacion_fin", "deuda_fin", "shocks_totales"]
    lines.append(df[keys_macro].round(2).to_markdown())
    lines.append("")
    lines.append("## Índices compuestos (0-100, mayor=mejor salvo estrés)")
    lines.append("")
    keys_ind = ["bienestar_fin", "gobernabilidad_fin", "estabilidad_fin",
                "idh_fin", "estres_fin"]
    lines.append(df[keys_ind].round(2).to_markdown())
    lines.append("")
    lines.append("## Métricas constitucionales del decisor")
    lines.append("")
    keys_llm = ["coherencia_temporal", "diversidad_valores",
                "reformas_totales", "reformas_radicales",
                "delta_iva_medio", "delta_isr_medio"]
    lines.append(df[keys_llm].round(3).to_markdown())
    lines.append("")
    lines.append("## Presupuesto promedio (%)")
    lines.append("")
    keys_pres = [c for c in df.columns if c.startswith("presup_")]
    lines.append(df[keys_pres].round(2).to_markdown())
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# --- entrypoint --------------------------------------------------------------


def generar_comparativa(
    corridas: Sequence[CorridaEtiquetada], out_dir: Path
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [
        overlay_trayectorias(corridas, out_dir),
        overlay_indicadores_compuestos(corridas, out_dir),
        radar_comparativo(corridas, out_dir),
        barras_metricas_llm(corridas, out_dir),
        reporte_markdown(corridas, out_dir / "reporte.md"),
    ]
    return outs
