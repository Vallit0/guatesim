"""Snapshot del presupuesto público ejecutado de Guatemala 2024 (MINFIN).

Fuente: validado contra ICEFI (Instituto Centroamericano de Estudios
Fiscales), que opera con datos oficiales del SICOIN (Sistema de
Contabilidad Integrada del MINFIN).

  - Por finalidad (salud, educación, protección social): Tabla 8,
    ICEFI "Análisis del proyecto de presupuesto para 2026" (nov 2025),
    serie ejecutada 2020-2024. Total ejecutado 2024 ≈ Q120,903 MM.
  - Por entidad (seguridad, infraestructura, agro): Tabla 7, ICEFI
    "Análisis del presupuesto aprobado para 2025" (dic 2024). Vigente
    2024 = Q131,196.5 MM (techo presupuestario).
  - Servicio de deuda: clasificación económica, ICEFI nov 2025 Fig 18,
    14.7% incluye intereses + amortización (12.0% solo intereses).
  - Justicia (OJ+MP+IDPP+INACIF+TSE): estimación basada en
    asignaciones constitucionales (Art. 213 CPRG: OJ ≥2%) más
    incrementos confirmados en Decreto 36-2024.

**Marcos heterogéneos**: las shares mezclan clasificación funcional
(finalidad) y administrativa (entidad). La columna `nota` documenta
el marco de cada partida. La suma cierra a 100 ± 1.

Cita primaria del paper:
  ICEFI (2024). *Análisis del presupuesto aprobado para 2025*.
  ICEFI (2025). *Análisis y recomendaciones para el proyecto de
    presupuesto para 2026*.
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
        fuente=(
            "ICEFI Tabla 7 (vigente 2024, dic 2024) + Tabla 8 "
            "(ejecutado 2024, nov 2025), datos primarios SICOIN"
        ),
        es_aproximacion=False,
        csv_path=path,
        notas=notas,
    )
