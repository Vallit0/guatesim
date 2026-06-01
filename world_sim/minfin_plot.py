"""Plots comparativos contra el baseline humano MINFIN.

Convierte el hallazgo abstracto "los modelos difieren entre sí" en uno
narrable: "todos se desvían del baseline humano de referencia, en
direcciones medibles y opuestas".

Esta es la pieza que cierra §5 del paper con un anchor humano.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .actions import PresupuestoAnual
from .minfin_ingest import MinfinBaseline, load_minfin_baseline


PARTIDAS_ORDEN: tuple[str, ...] = (
    "salud",
    "educacion",
    "seguridad",
    "infraestructura",
    "agro_desarrollo_rural",
    "proteccion_social",
    "servicio_deuda",
    "justicia",
    "otros",
)
"""Orden canónico de las partidas para plots — coherente entre figuras."""


PARTIDAS_LABEL_CORTO: dict[str, str] = {
    "salud": "salud",
    "educacion": "educación",
    "seguridad": "seguridad",
    "infraestructura": "infra.",
    "agro_desarrollo_rural": "agro",
    "proteccion_social": "prot. soc.",
    "servicio_deuda": "deuda",
    "justicia": "justicia",
    "otros": "otros",
}


def _to_share_dict(p: PresupuestoAnual | dict) -> dict[str, float]:
    """Acepta PresupuestoAnual o dict con las 9 keys."""
    if isinstance(p, PresupuestoAnual):
        return p.model_dump()
    return dict(p)


def deviation_table(
    model_budgets: dict[str, PresupuestoAnual | dict],
    baseline: MinfinBaseline | None = None,
) -> pd.DataFrame:
    """Tabla de desviaciones de cada modelo vs MINFIN baseline.

    Args:
        model_budgets: dict {label: presupuesto} con un presupuesto por
            modelo (típicamente el promedio sobre seeds × turnos).
        baseline: opcional, defaults a `load_minfin_baseline()`.

    Returns:
        DataFrame con índice = partidas, columnas = modelos + 'MINFIN'.
        Valores en puntos porcentuales. Última fila (`abs_dev_total`)
        suma las desviaciones absolutas vs MINFIN — métrica de "qué tan
        lejos está cada modelo del baseline humano".
    """
    if baseline is None:
        baseline = load_minfin_baseline()

    bl = baseline.presupuesto.model_dump()
    rows = []
    for partida in PARTIDAS_ORDEN:
        row = {"partida": partida, "MINFIN_2024": float(bl[partida])}
        for label, p in model_budgets.items():
            shares = _to_share_dict(p)
            row[label] = float(shares[partida])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("partida")

    # Fila resumen: desviación total absoluta vs MINFIN por modelo
    if model_budgets:
        abs_devs = {
            label: float(np.sum(np.abs(df[label] - df["MINFIN_2024"])))
            for label in model_budgets
        }
        abs_devs["MINFIN_2024"] = 0.0
        summary_row = pd.DataFrame([abs_devs], index=["abs_dev_total"])
        df = pd.concat([df, summary_row])
    return df


def plot_budgets_vs_minfin(
    model_budgets: dict[str, PresupuestoAnual | dict],
    output_path: Path | str,
    baseline: MinfinBaseline | None = None,
    title: str = "Asignación presupuestaria: LLMs vs baseline humano (MINFIN 2024)",
    figsize: tuple[float, float] = (12, 5.5),
) -> Path:
    """Bar chart agrupado por partida con N modelos + MINFIN baseline.

    Args:
        model_budgets: {label: presupuesto}. Cada label es una columna
            en el grupo de cada partida.
        output_path: PNG output.
        baseline: opcional, defaults a `load_minfin_baseline()`.
        title: título del plot.

    Returns:
        Path al PNG generado.
    """
    if baseline is None:
        baseline = load_minfin_baseline()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bl_shares = baseline.presupuesto.model_dump()

    labels = list(model_budgets.keys())
    n_groups = len(PARTIDAS_ORDEN)
    n_bars = len(labels) + 1  # + baseline
    width = 0.85 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline MINFIN — primera barra del grupo, color destacado
    bl_values = [bl_shares[p] for p in PARTIDAS_ORDEN]
    ax.bar(
        x + 0 * width,
        bl_values,
        width,
        label="MINFIN 2024 (baseline humano)",
        color="#2c3e50",
        edgecolor="black",
        linewidth=0.5,
    )

    # Modelos LLM — barras siguientes con colores tab10
    cmap = plt.get_cmap("tab10")
    for i, label in enumerate(labels):
        shares = _to_share_dict(model_budgets[label])
        values = [shares[p] for p in PARTIDAS_ORDEN]
        ax.bar(
            x + (i + 1) * width,
            values,
            width,
            label=label,
            color=cmap(i),
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(
        [PARTIDAS_LABEL_CORTO[p] for p in PARTIDAS_ORDEN],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("% del presupuesto")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_deviation_summary(
    model_budgets: dict[str, PresupuestoAnual | dict],
    output_path: Path | str,
    baseline: MinfinBaseline | None = None,
) -> Path:
    """Escribe un markdown con la tabla de desviaciones + lectura.

    El texto resultante es directamente quotable para §5 del paper.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = deviation_table(model_budgets, baseline=baseline)

    lines: list[str] = []
    lines.append("# Desviación de los LLMs vs baseline humano MINFIN 2024")
    lines.append("")
    lines.append(
        "Tabla por partida (% del presupuesto) y resumen agregado de "
        "desviación absoluta total vs MINFIN."
    )
    lines.append("")
    lines.append(df.round(2).to_markdown())
    lines.append("")
    lines.append("## Lectura")
    lines.append("")
    devs = df.loc["abs_dev_total"].drop("MINFIN_2024").sort_values(ascending=True)
    lines.append("**Modelos ordenados por proximidad al baseline humano**:")
    for label, val in devs.items():
        lines.append(f"- `{label}`: desviación absoluta total = {val:.1f} pp")
    lines.append("")
    if len(devs) >= 2:
        mejor = devs.index[0]
        peor = devs.index[-1]
        lines.append(
            f"`{mejor}` es el más cercano al baseline humano "
            f"(desviación {devs.iloc[0]:.1f} pp). `{peor}` es el más lejano "
            f"(desviación {devs.iloc[-1]:.1f} pp). Diferencia: "
            f"{devs.iloc[-1] - devs.iloc[0]:.1f} pp."
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
