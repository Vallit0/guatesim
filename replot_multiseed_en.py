"""Regenera las 3 figuras del análisis multi-seed con labels en inglés.

Toma los CSVs ya guardados (sin re-correr el análisis) y llama las
funciones de plot del módulo `guatemala_sim.multiseed` (ya traducidas
a inglés). Sobreescribe los PNGs existentes en la misma carpeta.

Uso:
    python replot_multiseed_en.py \\
        --analysis-dir figures/20260503_181558_dceacd_multiseed_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from guatemala_sim.multiseed import (
    plot_budget_with_ci,
    plot_mixed_effects_forest,
    plot_outcomes_box,
)


# Traducciones: column / index names en datos → display labels en inglés
BUDGET_LABEL_MAP = {
    "presup_salud":                "presup_health",
    "presup_educacion":            "presup_education",
    "presup_seguridad":            "presup_security",
    "presup_infraestructura":      "presup_infrastructure",
    "presup_agro_desarrollo_rural": "presup_agriculture",
    "presup_proteccion_social":    "presup_social_protection",
    "presup_servicio_deuda":       "presup_debt_service",
    "presup_justicia":             "presup_justice",
    "presup_otros":                "presup_other",
}

METRIC_LABEL_MAP = {
    "aprobacion_presidencial":     "presidential_approval",
    "ind_gobernabilidad":          "governance_index",
    "presup_proteccion_social":    "budget_social_protection",
    "pobreza_general":             "general_poverty",
    "indice_protesta":             "protest_index",
    "pib_usd_mm":                  "gdp_usd_mm",
    "deuda_pib":                   "debt_gdp",
    "estabilidad_macro":           "macro_stability",
    "estres_social":               "social_stress",
    "idh":                         "hdi",
    "bienestar":                   "wellbeing",
    "coherencia_temporal":         "temporal_coherence",
    "diversidad_valores":          "value_diversity",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", type=str, required=True)
    args = ap.parse_args()

    d = Path(args.analysis_dir)
    if not d.exists():
        raise FileNotFoundError(d)

    # df: metrics_per_seed con MultiIndex (seed, replica, modelo)
    df = pd.read_csv(d / "metrics_per_seed.csv")
    df = df.set_index(["seed", "replica", "modelo"])
    df = df.rename(columns=BUDGET_LABEL_MAP)

    # agg: aggregate_by_model con MultiIndex (modelo, metrica)
    agg = pd.read_csv(d / "aggregate_by_model.csv")
    agg["metrica"] = agg["metrica"].replace(BUDGET_LABEL_MAP)
    agg = agg.set_index(["modelo", "metrica"])

    # me_table: mixed_effects con metric como índice
    me = pd.read_csv(d / "mixed_effects.csv")
    me["metric"] = me["metric"].replace(METRIC_LABEL_MAP)
    me = me.set_index("metric")

    # Regenerar
    p1 = plot_budget_with_ci(df, agg, d / "presupuesto_ic95.png")
    print(f"[ok] {p1}")
    p2 = plot_outcomes_box(df, d / "outcomes_box.png")
    print(f"[ok] {p2}")
    p3 = plot_mixed_effects_forest(me, d / "mixed_effects_forest.png")
    print(f"[ok] {p3}")


if __name__ == "__main__":
    main()
