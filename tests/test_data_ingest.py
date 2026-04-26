"""Tests offline de `data_ingest`. No tocan la API del Banco Mundial:
fixtures sintéticos con la misma forma que devuelve `wbgapi`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.data_ingest import (
    NOT_FROM_WB,
    WB_INDICATORS,
    calibrate_initial_state,
    latest_per_indicator,
    load_snapshot,
    save_snapshot,
)


def _fake_snapshot() -> pd.DataFrame:
    """Snapshot con datos parecidos a los reales 2024 para Guatemala.

    Estructura idéntica a la que devuelve `fetch_world_bank`: índice =
    código del indicador, columnas = `YR2020`–`YR2024`. Algunos
    indicadores tienen NaN en el año más reciente (lag de publicación
    real, p. ej. SI.POV.NAHC viene con 1-2 años de atraso).
    """
    cols = ["YR2020", "YR2021", "YR2022", "YR2023", "YR2024"]
    data = {
        "NY.GDP.MKTP.CD": [77.7e9, 86.5e9, 95.6e9, 104.4e9, 113.2e9],
        "NY.GDP.MKTP.KD.ZG": [-1.79, 8.04, 4.18, 3.53, 3.65],
        "FP.CPI.TOTL.ZG": [3.21, 4.26, 6.89, 6.21, 2.87],
        "GC.DOD.TOTL.GD.ZS": [31.5, 30.8, 29.4, 27.5, np.nan],
        "FI.RES.TOTL.CD": [18e9, 21e9, 23e9, 23.5e9, 24e9],
        "BN.CAB.XOKA.GD.ZS": [4.82, 2.36, 1.13, 3.20, np.nan],
        "BX.TRF.PWKR.DT.GD.ZS": [14.66, 17.86, 18.89, 19.50, 20.10],
        "PA.NUS.FCRF": [7.72, 7.72, 7.74, 7.85, 7.74],
        "BX.KLT.DINV.CD.WD": [800e6, 1.4e9, 1.5e9, 1.6e9, 1.7e9],
        "SP.POP.TOTL": [16.86e6, 17.11e6, 17.35e6, 17.60e6, 17.84e6],
        "SI.POV.NAHC": [np.nan, np.nan, 56.0, np.nan, np.nan],  # lag típico
        "SI.POV.GINI": [np.nan, np.nan, 47.8, np.nan, np.nan],
        "SL.UEM.TOTL.ZS": [3.47, 2.91, 2.32, 2.79, 2.85],
        "VC.IHR.PSRC.P5": [15.4, 17.3, 17.5, 16.7, np.nan],
        "SE.PRM.NENR": [np.nan, 92.5, 91.8, np.nan, np.nan],
    }
    return pd.DataFrame(data, index=cols).T


def test_wb_indicators_cubre_macro_y_social():
    """Sanity: el mapeo cubre la mayoría de campos de Macro y Social."""
    fields_mapped = {field for (field, _) in WB_INDICATORS.values()}
    # macro
    assert "pib_usd_mm" in fields_mapped
    assert "inflacion" in fields_mapped
    assert "deuda_pib" in fields_mapped
    assert "remesas_pib" in fields_mapped
    # social
    assert "poblacion_mm" in fields_mapped
    assert "gini" in fields_mapped
    assert "homicidios_100k" in fields_mapped


def test_no_overlap_entre_wb_y_not_from_wb():
    fields_wb = {field for (field, _) in WB_INDICATORS.values()}
    fields_not_wb = set(NOT_FROM_WB.keys())
    assert fields_wb.isdisjoint(fields_not_wb), \
        f"campos solapados: {fields_wb & fields_not_wb}"


def test_latest_per_indicator_toma_ultimo_no_nan():
    df = _fake_snapshot()
    latest = latest_per_indicator(df)
    # PIB en 2024 → en MM
    assert "pib_usd_mm" in latest
    assert latest["pib_usd_mm"].año == 2024
    assert latest["pib_usd_mm"].valor == pytest.approx(113200.0, rel=0.001)
    # Deuda con NaN en 2024 → debe agarrar 2023 (27.5)
    assert latest["deuda_pib"].año == 2023
    assert latest["deuda_pib"].valor == pytest.approx(27.5)
    # Pobreza solo presente en 2022
    assert latest["pobreza_general"].año == 2022
    assert latest["pobreza_general"].valor == pytest.approx(56.0)
    # Gini convertido de 0-100 a 0-1
    assert latest["gini"].año == 2022
    assert latest["gini"].valor == pytest.approx(0.478)


def test_save_load_snapshot_roundtrip(tmp_path: Path):
    df = _fake_snapshot()
    p = tmp_path / "snap.csv"
    save_snapshot(df, p)
    df2 = load_snapshot(p)
    assert list(df.index) == list(df2.index)
    assert list(df.columns) == list(df2.columns)
    pd.testing.assert_frame_equal(df, df2)


def test_calibrate_initial_state_reemplaza_campos(tmp_path: Path):
    df = _fake_snapshot()
    p = tmp_path / "snap.csv"
    save_snapshot(df, p)
    state, meta = calibrate_initial_state(snapshot_path=p)
    # PIB debería venir del snapshot (≈ 113 200 MM USD)
    assert state.macro.pib_usd_mm == pytest.approx(113200.0, rel=0.001)
    # crecimiento del 2024 (3.65)
    assert state.macro.crecimiento_pib == pytest.approx(3.65, abs=0.01)
    # Pobreza 56% (último ENCOVI cargado)
    assert state.social.pobreza_general == pytest.approx(56.0)
    # gini convertido a [0, 1]
    assert 0 <= state.social.gini <= 1
    assert state.social.gini == pytest.approx(0.478, abs=0.001)
    # metadata documenta los reemplazos
    assert "pib_usd_mm" in meta["campos_reemplazados"]
    assert meta["campos_reemplazados"]["pib_usd_mm"]["año"] == 2024
    assert meta["campos_reemplazados"]["pib_usd_mm"]["fuente"] == "World Bank"


def test_calibrate_fallback_si_snapshot_no_existe(tmp_path: Path):
    """Si no hay snapshot, devuelve el estado hardcodeado y reporta el error."""
    p = tmp_path / "no_existe.csv"
    state, meta = calibrate_initial_state(snapshot_path=p, fallback=True)
    # estado válido (los campos default de bootstrap)
    assert state.macro.pib_usd_mm > 0
    assert "error" in meta
    assert "no existe" in meta["error"]


def test_calibrate_sin_fallback_levanta(tmp_path: Path):
    p = tmp_path / "no_existe.csv"
    with pytest.raises(FileNotFoundError):
        calibrate_initial_state(snapshot_path=p, fallback=False)


def test_calibrate_preserva_campos_no_wb(tmp_path: Path):
    """Indicadores políticos / perceptuales (NO en WB) deben quedar
    en sus defaults de bootstrap.py."""
    df = _fake_snapshot()
    p = tmp_path / "snap.csv"
    save_snapshot(df, p)
    state, meta = calibrate_initial_state(snapshot_path=p)
    # estos no vienen del WB y deben quedar como base
    assert state.politico.aprobacion_presidencial == 48.0
    assert state.externo.alineamiento_eeuu == 0.6
    assert state.social.informalidad == 70.0
    # y listamos correctamente que no tienen fuente WB
    assert "informalidad" in meta["campos_sin_fuente_wb"]
    assert "aprobacion_presidencial" in meta["campos_sin_fuente_wb"]


def test_unidades_conversion_correcta():
    """Verifica que los conversores producen unidades del state.
    PIB en USD → MM (×1e-6); población → MM; gini /100; etc.
    """
    df = _fake_snapshot()
    latest = latest_per_indicator(df)
    # PIB: 113.2 mil millones USD → 113 200 MM USD
    assert latest["pib_usd_mm"].valor == pytest.approx(113200.0, rel=0.001)
    # Población: 17.84 M → 17.84 MM
    assert latest["poblacion_mm"].valor == pytest.approx(17.84, abs=0.01)
    # Gini: 47.8 (escala WB) → 0.478 (escala state)
    assert latest["gini"].valor == pytest.approx(0.478, abs=0.001)
    # Reservas: 24e9 USD → 24 000 MM USD
    assert latest["reservas_usd_mm"].valor == pytest.approx(24000.0, rel=0.001)
