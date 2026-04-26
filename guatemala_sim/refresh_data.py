"""CLI: re-descarga el snapshot de datos del Banco Mundial para Guatemala.

Uso:

    python -m guatemala_sim.refresh_data
    python -m guatemala_sim.refresh_data --country GTM --from 2018 --to 2024
    python -m guatemala_sim.refresh_data --output data/world_bank_gtm.csv
    python -m guatemala_sim.refresh_data --banguat               # solo Banguat
    python -m guatemala_sim.refresh_data --banguat --dias 365    # último año

Después, `bootstrap.initial_state()` puede llamar a
`data_ingest.calibrate_initial_state()` para construir el estado inicial
con los datos reales en vez de los defaults hardcodeados.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from .data_ingest import (
    NOT_FROM_WB,
    calibrate_initial_state,
    fetch_world_bank,
    latest_per_indicator,
    save_snapshot,
)


ROOT = Path(__file__).resolve().parent.parent


def _refresh_banguat(out_path: Path, dias_atras: int) -> None:
    """Descarga la serie reciente de tipo de cambio USD/GTQ desde Banguat."""
    from .banguat_ingest import (
        latest_tipo_cambio_promedio,
        save_tipo_cambio_snapshot,
        tipo_cambio_rango,
    )
    fin = date.today()
    inicio = fin - timedelta(days=dias_atras)
    print(f"[refresh_data] Banguat: descargando USD/GTQ {inicio} → {fin}…")
    df = tipo_cambio_rango(inicio, fin)
    save_tipo_cambio_snapshot(df, out_path)
    avg30 = latest_tipo_cambio_promedio(df, ventana_dias=30)
    print(f"[refresh_data] {len(df)} observaciones diarias guardadas en {out_path}")
    print(f"[refresh_data] promedio últimos 30 días: {avg30:.4f} GTQ/USD")
    if not df.empty:
        ult = df.iloc[-1]
        print(f"[refresh_data] última fecha: {ult['fecha']}  ref={ult['referencia']:.4f}")


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
    ap.add_argument("--banguat", action="store_true",
                    help="descargar también tipo de cambio USD/GTQ desde Banguat")
    ap.add_argument("--solo-banguat", action="store_true",
                    help="saltarse la descarga del Banco Mundial")
    ap.add_argument("--dias", type=int, default=365,
                    help="rango (en días) hacia atrás para Banguat (default 365)")
    ap.add_argument("--banguat-output", type=str, default=None,
                    help="CSV de Banguat (default data/banguat_tipo_cambio.csv)")
    args = ap.parse_args()

    if args.solo_banguat or args.banguat:
        bg_path = (
            Path(args.banguat_output) if args.banguat_output
            else (ROOT / "data" / "banguat_tipo_cambio.csv")
        )
        _refresh_banguat(bg_path, args.dias)
        if args.solo_banguat:
            return
        print()  # separador antes del bloque WB

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
