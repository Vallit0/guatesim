"""Módulo de gráficas post-corrida.

Lee un archivo JSONL generado por `JsonlLogger` y produce:
    * trayectorias de indicadores macro/sociales (matplotlib)
    * stacked area del presupuesto a lo largo del tiempo (matplotlib)
    * radar de valores revelados (matplotlib)
    * heatmap territorial (matplotlib)
    * dashboard interactivo HTML (plotly)

Todas las funciones aceptan la lista `records` que viene de `read_run`
y devuelven la `Path` al archivo generado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # backend headless; evita TclError en tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .indicators import (
    coherencia_temporal,
    compute_indicators,
    diversidad_valores,
    resumen_presupuesto,
)
from .state import GuatemalaState


# --- helpers ------------------------------------------------------------------


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    """Series temporales de todos los escalares relevantes."""
    rows = []
    for r in records:
        sa = GuatemalaState.model_validate(r["state_after"])
        ind = compute_indicators(sa)
        rows.append({
            "t": r["t"],
            "fecha": r["fecha"],
            "pib_usd_mm": sa.macro.pib_usd_mm,
            "crecimiento_pib": sa.macro.crecimiento_pib,
            "inflacion": sa.macro.inflacion,
            "deuda_pib": sa.macro.deuda_pib,
            "balance_fiscal_pib": sa.macro.balance_fiscal_pib,
            "reservas_usd_mm": sa.macro.reservas_usd_mm,
            "tipo_cambio": sa.macro.tipo_cambio,
            "remesas_pib": sa.macro.remesas_pib,
            "ied_usd_mm": sa.macro.ied_usd_mm,
            "pobreza_general": sa.social.pobreza_general,
            "pobreza_extrema": sa.social.pobreza_extrema,
            "gini": sa.social.gini,
            "homicidios_100k": sa.social.homicidios_100k,
            "migracion_neta_miles": sa.social.migracion_neta_miles,
            "cobertura_salud": sa.social.cobertura_salud,
            "matricula_primaria": sa.social.matricula_primaria,
            "aprobacion_presidencial": sa.politico.aprobacion_presidencial,
            "indice_protesta": sa.politico.indice_protesta,
            "confianza_institucional": sa.politico.confianza_institucional,
            "coalicion_congreso": sa.politico.coalicion_congreso,
            "bienestar": ind.bienestar,
            "gobernabilidad": ind.gobernabilidad,
            "desarrollo_humano": ind.desarrollo_humano,
            "estabilidad_macro": ind.estabilidad_macro,
            "estres_social": ind.estres_social,
            "n_shocks": len(r.get("shocks", [])),
        })
    return pd.DataFrame(rows)


# --- plots matplotlib ---------------------------------------------------------


def plot_trayectorias_macro(records: list[dict], out_dir: Path) -> Path:
    df = _records_to_df(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.ravel()
    axes[0].plot(df["t"], df["pib_usd_mm"], lw=2, color="#1f77b4")
    axes[0].set_title("PIB (millones USD)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["t"], df["crecimiento_pib"], lw=2, color="#2ca02c")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_title("Crecimiento PIB (%)")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(df["t"], df["inflacion"], lw=2, color="#d62728", label="inflación")
    axes[2].plot(df["t"], df["deuda_pib"], lw=2, color="#9467bd", label="deuda/PIB")
    axes[2].legend(); axes[2].set_title("Inflación y deuda"); axes[2].grid(True, alpha=0.3)
    axes[3].plot(df["t"], df["tipo_cambio"], lw=2, color="#ff7f0e")
    axes[3].set_title("Tipo de cambio (GTQ/USD)")
    axes[3].grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlabel("turno")
    fig.suptitle("Trayectoria macro", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "trayectoria_macro.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_trayectorias_sociales(records: list[dict], out_dir: Path) -> Path:
    df = _records_to_df(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.ravel()
    axes[0].plot(df["t"], df["pobreza_general"], lw=2, color="#8c564b", label="general")
    axes[0].plot(df["t"], df["pobreza_extrema"], lw=2, color="#e377c2", label="extrema")
    axes[0].legend(); axes[0].set_title("Pobreza (%)"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["t"], df["homicidios_100k"], lw=2, color="#d62728")
    axes[1].set_title("Homicidios por 100k"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(df["t"], df["migracion_neta_miles"], lw=2, color="#17becf")
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_title("Migración neta (miles)"); axes[2].grid(True, alpha=0.3)
    axes[3].plot(df["t"], df["aprobacion_presidencial"], lw=2, color="#1f77b4", label="aprobación")
    axes[3].plot(df["t"], df["indice_protesta"], lw=2, color="#ff7f0e", label="protesta")
    axes[3].legend(); axes[3].set_title("Aprobación vs. protesta"); axes[3].grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlabel("turno")
    fig.suptitle("Trayectoria social y política", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "trayectoria_social.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_indicadores_compuestos(records: list[dict], out_dir: Path) -> Path:
    df = _records_to_df(records)
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, color in zip(
        ["bienestar", "gobernabilidad", "desarrollo_humano", "estabilidad_macro", "estres_social"],
        ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728"],
    ):
        ax.plot(df["t"], df[col], lw=2, color=color, label=col)
    ax.set_title("Indicadores compuestos a lo largo de la simulación", fontsize=13, fontweight="bold")
    ax.set_xlabel("turno")
    ax.set_ylabel("índice (0–100)")
    ax.set_ylim(0, 100)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "indicadores_compuestos.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_presupuesto_stacked(records: list[dict], out_dir: Path) -> Path:
    rows = []
    for r in records:
        p = r["decision"]["presupuesto"]
        row = {"t": r["t"], **{k: v for k, v in p.items()}}
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c != "t"]
    fig, ax = plt.subplots(figsize=(12, 6))
    # colores cualitativos
    cmap = plt.get_cmap("tab10")
    ax.stackplot(df["t"], [df[c] for c in cols], labels=cols,
                 colors=[cmap(i) for i in range(len(cols))], alpha=0.85)
    ax.set_title("Composición del presupuesto por turno", fontsize=13, fontweight="bold")
    ax.set_xlabel("turno")
    ax.set_ylabel("% del gasto público")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "presupuesto_stacked.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_radar_valores(records: list[dict], out_dir: Path) -> Path:
    """Radar del presupuesto promedio — muestra prioridades del presidente."""
    avg = resumen_presupuesto([r["decision"] for r in records])
    if not avg:
        return out_dir / "radar_valores.png"
    cats = list(avg.keys())
    vals = [avg[c] for c in cats]
    # cerrar el polígono
    vals = vals + vals[:1]
    angs = np.linspace(0, 2 * np.pi, len(cats) + 1)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.plot(angs, vals, lw=2, color="#1f77b4")
    ax.fill(angs, vals, alpha=0.25, color="#1f77b4")
    ax.set_xticks(angs[:-1])
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 20)
    ax.set_title("Valores revelados: presupuesto promedio (%)\n",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "radar_valores.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_heatmap_decisiones(records: list[dict], out_dir: Path) -> Path:
    """Heatmap turno×partida del presupuesto."""
    rows = []
    for r in records:
        p = r["decision"]["presupuesto"]
        rows.append({"t": r["t"], **p})
    df = pd.DataFrame(rows).set_index("t")
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(df.T.values, aspect="auto", cmap="YlGnBu")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index)
    ax.set_xlabel("turno")
    ax.set_title("Heatmap de asignación presupuestaria (%)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="% del presupuesto")
    fig.tight_layout()
    out = out_dir / "heatmap_presupuesto.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_metricas_del_llm(records: list[dict], out_dir: Path) -> Path:
    """Barras: coherencia temporal, diversidad, n shocks, n reformas."""
    decisiones = [r["decision"] for r in records]
    coherencia = coherencia_temporal(decisiones)
    diversidad = diversidad_valores(decisiones)
    n_shocks = sum(len(r.get("shocks", [])) for r in records)
    n_reformas = sum(len(r["decision"].get("reformas", [])) for r in records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(["coherencia\ntemporal", "diversidad\nvalores"],
                [coherencia, diversidad * 30],  # escalar diversidad para verla
                color=["#2ca02c", "#9467bd"])
    axes[0].set_title("Métricas constitucionales del LLM")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].bar(["shocks\ntotales", "reformas\ntotales"],
                [n_shocks, n_reformas], color=["#d62728", "#1f77b4"])
    axes[1].set_title("Actividad a lo largo de la corrida")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = out_dir / "metricas_llm.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# --- plotly (dashboard interactivo HTML) --------------------------------------


def dashboard_interactivo(records: list[dict], out_dir: Path) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = _records_to_df(records)
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "PIB (mm USD)", "Crecimiento PIB (%)",
            "Pobreza (%)", "Aprobación vs. protesta",
            "Indicadores compuestos", "Inflación y deuda (%)"
        ),
        vertical_spacing=0.09,
    )
    fig.add_trace(go.Scatter(x=df["t"], y=df["pib_usd_mm"], name="PIB"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["t"], y=df["crecimiento_pib"], name="crec"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["pobreza_general"], name="general"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["t"], y=df["pobreza_extrema"], name="extrema"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["t"], y=df["aprobacion_presidencial"], name="aprob"), row=2, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["indice_protesta"], name="protesta"), row=2, col=2)
    for col, name in [("bienestar", "bienestar"), ("gobernabilidad", "gobern"),
                      ("desarrollo_humano", "IDH"), ("estabilidad_macro", "estab"),
                      ("estres_social", "estrés")]:
        fig.add_trace(go.Scatter(x=df["t"], y=df[col], name=name), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["t"], y=df["inflacion"], name="inflación"), row=3, col=2)
    fig.add_trace(go.Scatter(x=df["t"], y=df["deuda_pib"], name="deuda/PIB"), row=3, col=2)
    fig.update_layout(
        title="Guatemala Sim — Dashboard de corrida",
        height=1000, showlegend=True,
    )
    out = out_dir / "dashboard.html"
    fig.write_html(out)
    return out


# --- entrypoint conveniente ---------------------------------------------------


def generar_todo(records: list[dict], out_dir: Path) -> list[Path]:
    """Genera todas las gráficas; devuelve la lista de archivos creados."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = [
        plot_trayectorias_macro(records, out_dir),
        plot_trayectorias_sociales(records, out_dir),
        plot_indicadores_compuestos(records, out_dir),
        plot_presupuesto_stacked(records, out_dir),
        plot_radar_valores(records, out_dir),
        plot_heatmap_decisiones(records, out_dir),
        plot_metricas_del_llm(records, out_dir),
        dashboard_interactivo(records, out_dir),
    ]
    return outs
