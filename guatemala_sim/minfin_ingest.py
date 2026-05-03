"""Snapshot del presupuesto público ejecutado de Guatemala 2024 (MINFIN).

Fuente primaria: Ministerio de Finanzas Públicas (MINFIN), Liquidación
del Presupuesto General de Ingresos y Egresos del Estado para el
ejercicio fiscal 2024. Datos públicos disponibles vía:
    - https://www.minfin.gob.gt/   (Portal de Transparencia Fiscal)
    - SICOIN web                   (Sistema de Contabilidad Integrada)
    - ICEFI                        (https://icefi.org/) — análisis fiscal centroamericano

**Limitación honesta**: el snapshot en `data/minfin_2024_ejecutado.csv`
es una **aproximación** basada en la estructura conocida del gasto
público guatemalteco, agregada manualmente a las 9 partidas del schema
`PresupuestoAnual`. Los porcentajes son del orden correcto pero NO son
los números oficiales exactos extraídos automáticamente del SICOIN.
Para uso en publicación, **verificar contra la Liquidación oficial**
del MINFIN o contra las series de ICEFI.

El propósito de este baseline es transformar la narrativa del paper
de "los modelos difieren entre sí" a "ambos modelos se desvían del
baseline humano de referencia, en direcciones opuestas". Eso requiere
un baseline plausible, no necesariamente exacto al cuarto decimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .actions import PresupuestoAnual


DEFAULT_CSV_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "minfin_2024_ejecutado.csv"
)


@dataclass(frozen=True)
class MinfinBaseline:
    """Baseline humano del presupuesto guatemalteco para comparación."""

    presupuesto: PresupuestoAnual
    year: int
    fuente: str
    es_aproximacion: bool
    csv_path: Path
    notas: dict[str, str]  # partida → nota textual sobre qué incluye

    def to_share_dict(self) -> dict[str, float]:
        """Devuelve {partida: share_pct} para tablas comparativas."""
        return self.presupuesto.model_dump()


def load_minfin_baseline(
    csv_path: Path | str | None = None,
) -> MinfinBaseline:
    """Carga el snapshot MINFIN como `MinfinBaseline`.

    Args:
        csv_path: opcional. Si None, usa `data/minfin_2024_ejecutado.csv`
            del repo. Si pasas otro CSV, debe tener columnas
            `partida, share_pct, nota` con las 9 partidas exactas del
            schema `PresupuestoAnual`.

    Returns:
        MinfinBaseline con el presupuesto validado contra el schema.

    Raises:
        FileNotFoundError si el CSV no existe.
        ValueError si el CSV no contiene las 9 partidas o no suma ~100.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"snapshot MINFIN no encontrado en {path}")

    df = pd.read_csv(path)
    expected_cols = {"partida", "share_pct", "nota"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"CSV debe tener columnas {expected_cols}; tiene {set(df.columns)}"
        )

    expected_partidas = {
        "salud", "educacion", "seguridad", "infraestructura",
        "agro_desarrollo_rural", "proteccion_social", "servicio_deuda",
        "justicia", "otros",
    }
    got_partidas = set(df["partida"])
    if got_partidas != expected_partidas:
        falta = expected_partidas - got_partidas
        sobra = got_partidas - expected_partidas
        raise ValueError(
            f"partidas no coinciden con el schema. Falta: {falta}, sobra: {sobra}"
        )

    shares = df.set_index("partida")["share_pct"].to_dict()
    notas = df.set_index("partida")["nota"].to_dict()

    # PresupuestoAnual valida que sume 100 ± 1
    presupuesto = PresupuestoAnual(
        salud=float(shares["salud"]),
        educacion=float(shares["educacion"]),
        seguridad=float(shares["seguridad"]),
        infraestructura=float(shares["infraestructura"]),
        agro_desarrollo_rural=float(shares["agro_desarrollo_rural"]),
        proteccion_social=float(shares["proteccion_social"]),
        servicio_deuda=float(shares["servicio_deuda"]),
        justicia=float(shares["justicia"]),
        otros=float(shares["otros"]),
    )

    return MinfinBaseline(
        presupuesto=presupuesto,
        year=2024,
        fuente="MINFIN Liquidación 2024 (aproximación, ver minfin_ingest.py)",
        es_aproximacion=True,
        csv_path=path,
        notas=notas,
    )
