"""Estado inicial (enero 2026).

`initial_state()` devuelve los valores hardcodeados — están calibrados a
órdenes de magnitud razonables a partir de Banguat, INE ENCOVI, Banco
Mundial y MINFIN, pero no son auditables a una fecha exacta. Sirven como
fallback cuando no hay datos reales descargados.

`initial_state_calibrated()` carga el snapshot del Banco Mundial en
`data/world_bank_gtm.csv` (generado por
`python -m guatemala_sim.refresh_data`) y reemplaza los campos macro/social
disponibles con los datos reales. Los campos políticos/perceptuales
(aprobación, libertad de prensa, alineamientos exteriores) quedan en sus
defaults porque no tienen fuente WB estable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from .state import (
    Externo,
    GuatemalaState,
    Macro,
    Politico,
    Social,
    Turno,
)


def initial_state() -> GuatemalaState:
    """Devuelve un `GuatemalaState` hardcodeado a enero 2026.

    Para una versión calibrada con datos reales del Banco Mundial,
    usá `initial_state_calibrated()`.
    """
    return GuatemalaState(
        turno=Turno(t=0, fecha=date(2026, 1, 1), periodo="anual"),
        macro=Macro(
            pib_usd_mm=115_000.0,       # ~USD 115 mil millones nominales
            crecimiento_pib=3.5,        # tendencia reciente
            inflacion=4.2,
            deuda_pib=28.5,             # baja vs. región
            reservas_usd_mm=22_000.0,
            balance_fiscal_pib=-1.7,
            cuenta_corriente_pib=2.3,   # superávit por remesas
            remesas_pib=19.5,
            tipo_cambio=7.75,
            ied_usd_mm=1_600.0,
        ),
        social=Social(
            poblacion_mm=18.2,
            pobreza_general=55.0,
            pobreza_extrema=21.0,
            gini=0.48,
            desempleo=3.0,              # oficial; enmascara subempleo
            informalidad=70.0,
            homicidios_100k=16.5,
            migracion_neta_miles=-120.0,
            matricula_primaria=89.0,
            cobertura_salud=55.0,
        ),
        politico=Politico(
            aprobacion_presidencial=48.0,
            indice_protesta=30.0,
            confianza_institucional=25.0,
            coalicion_congreso=38.0,
            libertad_prensa=45.0,
        ),
        externo=Externo(
            alineamiento_eeuu=0.6,
            alineamiento_china=0.1,
            relacion_mexico=0.5,
            relacion_triangulo_norte=0.4,
            apoyo_multilateral=60.0,
        ),
        shocks_activos=[],
        eventos_turno=["estado inicial: enero 2026"],
        memoria_presidencial=[
            "doctrina inicial: continuidad institucional y apertura multilateral"
        ],
    )


def initial_state_calibrated(
    snapshot_path: Path | None = None,
    target_date: date = date(2026, 1, 1),
    fallback: bool = True,
) -> tuple[GuatemalaState, dict[str, Any]]:
    """Estado inicial calibrado con el snapshot real del Banco Mundial.

    Si el snapshot no existe y `fallback=True`, devuelve `initial_state()`
    hardcodeado. Es un wrapper de `data_ingest.calibrate_initial_state`
    expuesto desde acá para que sea el punto de entrada canónico.
    """
    from .data_ingest import calibrate_initial_state
    return calibrate_initial_state(
        snapshot_path=snapshot_path,
        target_date=target_date,
        fallback=fallback,
    )


if __name__ == "__main__":
    import json

    s = initial_state()
    print(json.dumps(s.model_dump(mode="json"), indent=2, ensure_ascii=False))
