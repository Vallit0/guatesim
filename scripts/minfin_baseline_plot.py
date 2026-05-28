"""Genera el plot comparativo MINFIN 2024 vs los candidatos del menú IRL.

Outputs (en figures/minfin_baseline/ por default):
    comparison.png        — bar chart agrupado: 5 candidatos + MINFIN baseline
    deviations.md         — tabla markdown de desviaciones + lectura

Cuando haya datos reales de Claude/GPT (después de correr
`compare_llms_multiseed.py --menu-mode`), se puede llamar
`plot_budgets_vs_minfin` desde el pipeline de análisis post-corrida
pasando los presupuestos promedio reales en vez de los candidatos
canónicos.

Uso:
    python minfin_baseline_plot.py
    python minfin_baseline_plot.py --output figures/minfin_baseline_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from guatemala_sim.irl.candidates import generate_candidate_menu
from guatemala_sim.minfin_ingest import load_minfin_baseline
from guatemala_sim.minfin_plot import (
    plot_budgets_vs_minfin,
    write_deviation_summary,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", type=Path, default=Path("figures/minfin_baseline"),
        help="directorio de salida.",
    )
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bl = load_minfin_baseline()
    print(f"[minfin] baseline: {bl.fuente}")
    print(f"[minfin] aproximacion: {bl.es_aproximacion}")

    menu = generate_candidate_menu()
    model_budgets = {c.name: c.presupuesto for c in menu}

    fig_path = args.output / "comparison.png"
    plot_budgets_vs_minfin(
        model_budgets=model_budgets,
        output_path=fig_path,
        baseline=bl,
        title=(
            "Asignacion presupuestaria: 5 candidatos canonicos del menu "
            "IRL vs baseline humano (MINFIN 2024)"
        ),
    )
    print(f"[minfin] figura -> {fig_path}")

    md_path = args.output / "deviations.md"
    write_deviation_summary(
        model_budgets=model_budgets,
        output_path=md_path,
        baseline=bl,
    )
    print(f"[minfin] tabla  -> {md_path}")


if __name__ == "__main__":
    main()
