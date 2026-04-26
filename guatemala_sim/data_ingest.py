"""Ingesta de datos reales para calibrar el simulador.

Fuentes:

* **Banco Mundial** (vía `wbgapi`): macro, social, demografía. Es la única
  fuente con API estable y serie temporal larga para Guatemala. Ver
  `WB_INDICATORS` para el mapeo `código → campo del state`.
* **Banguat / INE / MINFIN**: no exponen APIs estables. Acá los soportamos
  cargando CSVs manualmente curados desde `data/`. Ver
  `data/README_DATOS.md` para el procedimiento de update.

Diseño:

* `fetch_world_bank()` descarga; `save_snapshot()` lo persiste a CSV;
  `load_snapshot()` lo lee. La separación permite correr el sim offline.
* `latest_per_indicator()` toma el último año no-NaN por indicador
  (porque distintos indicadores tienen distinto lag de publicación).
* `calibrate_initial_state()` produce un `GuatemalaState` reemplazando los
  defaults hardcodeados por los datos reales disponibles, manteniendo los
  defaults como *fallback* para indicadores ausentes.

Ver también `python -m guatemala_sim.refresh_data --help`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .bootstrap import initial_state
from .state import GuatemalaState, Macro, Social


# --- mapeo Banco Mundial → campos del state ---------------------------------
#
# Cada entrada: (nombre_canónico, conversor) donde el conversor toma el valor
# crudo del Banco Mundial y lo lleva a las unidades que `state.py` espera.

WB_INDICATORS: dict[str, tuple[str, Callable[[float], float]]] = {
    # macro
    "NY.GDP.MKTP.CD": ("pib_usd_mm", lambda x: x / 1_000_000.0),       # USD → MM
    "NY.GDP.MKTP.KD.ZG": ("crecimiento_pib", lambda x: x),              # %
    "FP.CPI.TOTL.ZG": ("inflacion", lambda x: x),                       # %
    "GC.DOD.TOTL.GD.ZS": ("deuda_pib", lambda x: x),                    # % PIB
    "FI.RES.TOTL.CD": ("reservas_usd_mm", lambda x: x / 1_000_000.0),   # USD → MM
    "GC.BAL.CASH.GD.ZS": ("balance_fiscal_pib", lambda x: x),           # % PIB
    "BN.CAB.XOKA.GD.ZS": ("cuenta_corriente_pib", lambda x: x),         # % PIB
    "BX.TRF.PWKR.DT.GD.ZS": ("remesas_pib", lambda x: x),               # % PIB
    "PA.NUS.FCRF": ("tipo_cambio", lambda x: x),                        # GTQ/USD
    "BX.KLT.DINV.CD.WD": ("ied_usd_mm", lambda x: x / 1_000_000.0),     # USD → MM
    # social
    "SP.POP.TOTL": ("poblacion_mm", lambda x: x / 1_000_000.0),         # → MM
    "SI.POV.NAHC": ("pobreza_general", lambda x: x),                    # %
    "SI.POV.GINI": ("gini", lambda x: x / 100.0),                       # 0-100 → 0-1
    "SL.UEM.TOTL.ZS": ("desempleo", lambda x: x),                       # %
    "VC.IHR.PSRC.P5": ("homicidios_100k", lambda x: x),                 # tasa
    "SE.PRM.NENR": ("matricula_primaria", lambda x: x),                 # %
    "SH.UHC.SRVS.CV.XD": ("cobertura_salud", lambda x: x),              # 0-100
}


# Indicadores que son políticos/perceptuales y NO vienen del Banco Mundial.
# Los listamos para documentación; quedan en sus defaults de `bootstrap.py`.
NOT_FROM_WB = {
    "informalidad": "ENEI / ILO; manual en data/banguat_macro_*.csv",
    "pobreza_extrema": "ENCOVI; última disponible 2014. Manual.",
    "migracion_neta_miles": "OIM / DGM; manual",
    "aprobacion_presidencial": "encuestas (CIEP, Latinobarómetro); manual",
    "indice_protesta": "construido; sin fuente directa",
    "confianza_institucional": "Latinobarómetro / LAPOP; manual",
    "coalicion_congreso": "Congreso de la República; manual",
    "libertad_prensa": "RSF / Freedom House; manual",
    "alineamiento_eeuu": "experto; manual",
    "alineamiento_china": "experto; manual",
    "relacion_mexico": "experto; manual",
    "relacion_triangulo_norte": "experto; manual",
    "apoyo_multilateral": "experto / IMF Article IV; manual",
}


# --- fetch + persistencia --------------------------------------------------


def _fetch_one(wb, code, country, years, retries=3, backoff=2.0):
    """Fetch un único indicador con retry exponencial.

    Devuelve un DataFrame de 1 fila indexado por el código del indicador,
    o `None` si falla todos los reintentos. (`wb.data.DataFrame` con un
    solo indicador devuelve el DataFrame indexado por economía, así que
    re-etiquetamos el índice con el código.)
    """
    import time
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = wb.data.DataFrame([code], country, time=years)
            if df is None or df.empty:
                return None
            df = df.copy()
            # solo una fila esperada (la del país); renombrar a `code`
            df.index = pd.Index([code] * len(df), name="indicator")
            return df
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    print(f"[fetch_world_bank] indicador {code} falló tras {retries} intentos: "
          f"{type(last_err).__name__}: {str(last_err)[:120]}")
    return None


def fetch_world_bank(
    country: str = "GTM",
    year_from: int = 2018,
    year_to: int | None = None,
) -> pd.DataFrame:
    """Descarga los indicadores `WB_INDICATORS` para `country` en el rango.

    Devuelve un DataFrame con índice = código del indicador y columnas
    `YR2018`, `YR2019`, …, `YRyear_to`. Valores en unidades crudas del WB
    (todavía no convertidos).

    Pide cada indicador por separado con retry exponencial. Si un indicador
    individual falla (JSON malformado, 504, etc.) se omite y se continúa
    con el resto, en vez de abortar la corrida entera.

    Requiere `wbgapi` instalado: `pip install wbgapi`.
    """
    try:
        import wbgapi as wb
    except ImportError as e:
        raise RuntimeError(
            "wbgapi no instalado. Corré `pip install wbgapi` o "
            "`pip install -e .[ingest]`."
        ) from e
    if year_to is None:
        # WB publica con ~1 año de lag; pedir el año actual rara vez tiene datos
        year_to = date.today().year - 1
    indicators = list(WB_INDICATORS.keys())
    years = list(range(year_from, year_to + 1))

    chunks: list[pd.DataFrame] = []
    fallidos: list[str] = []
    for code in indicators:
        df_one = _fetch_one(wb, code, country, years)
        if df_one is not None and not df_one.empty:
            chunks.append(df_one)
        else:
            fallidos.append(code)
    if not chunks:
        raise RuntimeError(
            f"WB API: ningún indicador descargado para {country}. "
            f"Fallidos: {fallidos}"
        )
    if fallidos:
        print(f"[fetch_world_bank] {len(fallidos)} indicadores omitidos: {fallidos}")
    return pd.concat(chunks)


def save_snapshot(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def load_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


# --- transformación a campos del state -------------------------------------


@dataclass
class IndicadorRecuperado:
    field: str
    año: int
    valor: float          # ya convertido a unidades del state
    valor_crudo: float    # valor original del Banco Mundial


def latest_per_indicator(df: pd.DataFrame) -> dict[str, IndicadorRecuperado]:
    """Para cada indicador, último año no-NaN. Devuelve `{field_name: IndicadorRecuperado}`.

    Si el código no está en `df`, queda fuera (no se incluye con valor `None`).
    """
    out: dict[str, IndicadorRecuperado] = {}
    for code, (field, conv) in WB_INDICATORS.items():
        if code not in df.index:
            continue
        row = df.loc[code]
        # las columnas son strings tipo "YR2024"
        year_cols = [c for c in row.index if isinstance(c, str) and c.startswith("YR")]
        years_sorted = sorted(year_cols, key=lambda c: int(c.replace("YR", "")))
        latest_year: int | None = None
        latest_value: float | None = None
        for col in years_sorted:
            v = row[col]
            if pd.notna(v):
                latest_year = int(col.replace("YR", ""))
                latest_value = float(v)
        if latest_value is not None and latest_year is not None:
            out[field] = IndicadorRecuperado(
                field=field,
                año=latest_year,
                valor=float(conv(latest_value)),
                valor_crudo=latest_value,
            )
    return out


# --- calibración del estado inicial ---------------------------------------


def calibrate_initial_state(
    snapshot_path: Path | None = None,
    target_date: date = date(2026, 1, 1),
    fallback: bool = True,
) -> tuple[GuatemalaState, dict[str, Any]]:
    """Devuelve `(state, metadata)` donde `state` es el estado inicial con
    los campos macro/social reemplazados por los últimos datos disponibles
    del Banco Mundial, y `metadata` documenta qué se reemplazó.

    Si `snapshot_path` no existe y `fallback=True`, devuelve el estado
    hardcodeado de `bootstrap.initial_state()` y reporta los campos no
    reemplazados.
    """
    base = initial_state()
    metadata: dict[str, Any] = {
        "target_date": target_date.isoformat(),
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "campos_reemplazados": {},
        "campos_default": [],
        "campos_sin_fuente_wb": list(NOT_FROM_WB.keys()),
    }

    if snapshot_path is None:
        snapshot_path = (
            Path(__file__).resolve().parent.parent / "data" / "world_bank_gtm.csv"
        )
    metadata["snapshot_path"] = str(snapshot_path)

    if not Path(snapshot_path).exists():
        if fallback:
            metadata["error"] = (
                f"snapshot {snapshot_path} no existe; usando defaults hardcoded. "
                f"corré `python -m guatemala_sim.refresh_data` para generarlo."
            )
            return base, metadata
        raise FileNotFoundError(f"snapshot no encontrado: {snapshot_path}")

    df = load_snapshot(Path(snapshot_path))
    latest = latest_per_indicator(df)

    # construimos los nuevos Macro y Social a partir del state base
    macro_dict = base.macro.model_dump()
    social_dict = base.social.model_dump()

    macro_fields = set(Macro.model_fields.keys())
    social_fields = set(Social.model_fields.keys())

    for field, ind in latest.items():
        if field in macro_fields:
            macro_dict[field] = ind.valor
            metadata["campos_reemplazados"][field] = {
                "valor": ind.valor, "año": ind.año,
                "fuente": "World Bank", "valor_crudo": ind.valor_crudo,
            }
        elif field in social_fields:
            social_dict[field] = ind.valor
            metadata["campos_reemplazados"][field] = {
                "valor": ind.valor, "año": ind.año,
                "fuente": "World Bank", "valor_crudo": ind.valor_crudo,
            }

    metadata["campos_default"] = sorted(
        set(macro_fields | social_fields)
        - set(metadata["campos_reemplazados"].keys())
    )

    new_state = base.model_copy(update={
        "macro": Macro(**macro_dict),
        "social": Social(**social_dict),
        "eventos_turno": [
            f"estado inicial: {target_date.isoformat()} "
            f"(calibrado vs. Banco Mundial, {len(metadata['campos_reemplazados'])} "
            f"campos reemplazados)"
        ],
    })
    return new_state, metadata
