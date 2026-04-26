"""CLI: re-descarga el snapshot de datos del Banco Mundial para Guatemala.

Uso:

    python -m guatemala_sim.refresh_data
    python -m guatemala_sim.refresh_data --country GTM --from 2018 --to 2024
    python -m guatemala_sim.refresh_data --output data/world_bank_gtm.csv

Después, `bootstrap.initial_state()` puede llamar a
`data_ingest.calibrate_initial_state()` para construir el estado inicial
con los datos reales en vez de los defaults hardcodeados.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from .data_ingest import (
    NOT_FROM_WB,
    calibrate_initial_state,
    fetch_world_bank,
    latest_per_indicator,
    save_snapshot,
)


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--country", default="GTM",
                    help="ISO-3 del país (default GTM)")
    ap.add_argument("--from", dest="year_from", type=int, default=2018)
    ap.add_argument("--to", dest="year_to", type=int, default=None,
                    help="último año a descargar (default: año actual)")
    ap.add_argument("--output", type=str, default=None,
                    help="ruta CSV de salida (default data/world_bank_gtm.csv)")
    ap.add_argument("--no-resumen", action="store_true",
                    help="no imprimir tabla resumen")
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else (ROOT / "data" / "world_bank_gtm.csv")

    print(f"[refresh_data] descargando {args.country} desde Banco Mundial…")
    df = fetch_world_bank(
        country=args.country,
        year_from=args.year_from,
        year_to=args.year_to,
    )
    save_snapshot(df, out_path)
    print(f"[refresh_data] snapshot guardado en {out_path}")
    print(f"[refresh_data] indicadores: {len(df)}, años: {len(df.columns)}")

    if args.no_resumen:
        return

    latest = latest_per_indicator(df)
    print()
    print("=== último año disponible por indicador ===")
    print(f"{'campo':28s} {'año':>5s}  {'valor':>14s}  (unidad de state)")
    print("-" * 60)
    for field, ind in sorted(latest.items()):
        print(f"{field:28s} {ind.año:>5d}  {ind.valor:>14.3f}")

    no_wb = sorted(set(NOT_FROM_WB.keys()))
    if no_wb:
        print()
        print(f"=== campos del state que NO vienen del WB ({len(no_wb)}) ===")
        for f in no_wb:
            print(f"  - {f:28s}  fuente sugerida: {NOT_FROM_WB[f]}")

    # mostrar también qué cambios produciría el snapshot en el state
    state, meta = calibrate_initial_state(snapshot_path=out_path)
    n_repl = len(meta["campos_reemplazados"])
    n_def = len(meta["campos_default"])
    print()
    print(f"=== calibración del estado inicial ===")
    print(f"campos reemplazados con datos reales: {n_repl}")
    print(f"campos en default (sin dato WB):      {n_def}")
    if meta["campos_default"]:
        print(f"  defaults: {', '.join(meta['campos_default'])}")


if __name__ == "__main__":
    main()
