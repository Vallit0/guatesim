"""Tests del baseline MINFIN."""

from __future__ import annotations

import pytest

from guatemala_sim.actions import PresupuestoAnual
from guatemala_sim.minfin_ingest import (
    DEFAULT_CSV_PATH,
    MinfinBaseline,
    load_minfin_baseline,
)


def test_default_csv_existe():
    assert DEFAULT_CSV_PATH.exists(), f"falta {DEFAULT_CSV_PATH}"


def test_load_minfin_devuelve_baseline_valido():
    bl = load_minfin_baseline()
    assert isinstance(bl, MinfinBaseline)
    assert isinstance(bl.presupuesto, PresupuestoAnual)
    assert bl.year == 2024
    assert bl.es_aproximacion is True


def test_minfin_suma_100_pp_tolerancia_1():
    bl = load_minfin_baseline()
    p = bl.presupuesto
    total = (
        p.salud + p.educacion + p.seguridad + p.infraestructura
        + p.agro_desarrollo_rural + p.proteccion_social + p.servicio_deuda
        + p.justicia + p.otros
    )
    assert abs(total - 100.0) <= 1.0, f"suma = {total}"


def test_minfin_partidas_en_rango():
    bl = load_minfin_baseline()
    p = bl.presupuesto.model_dump()
    for partida, valor in p.items():
        assert 0 <= valor <= 100, f"{partida} = {valor} fuera de [0, 100]"


def test_minfin_notas_cubren_las_9_partidas():
    bl = load_minfin_baseline()
    expected = {
        "salud", "educacion", "seguridad", "infraestructura",
        "agro_desarrollo_rural", "proteccion_social", "servicio_deuda",
        "justicia", "otros",
    }
    assert set(bl.notas.keys()) == expected


def test_minfin_estructura_macro_es_realista():
    """Sanity: educación debe ser una cartera grande (mínimo
    constitucional), deuda debe existir y ser significativa, salud
    debe estar en rango plausible."""
    bl = load_minfin_baseline()
    p = bl.presupuesto
    # Educación: ≥ 10 % (mínimo constitucional aproximado)
    assert p.educacion >= 10.0
    # Salud: en torno a 10-15 %
    assert 5.0 <= p.salud <= 20.0
    # Servicio de deuda: significativo en Guatemala
    assert p.servicio_deuda >= 5.0


def test_load_minfin_csv_inexistente_levanta():
    from pathlib import Path
    with pytest.raises(FileNotFoundError):
        load_minfin_baseline(csv_path=Path("/no/existe/baseline.csv"))


def test_to_share_dict_contiene_las_9_partidas(tmp_path):
    bl = load_minfin_baseline()
    d = bl.to_share_dict()
    assert len(d) == 9
    assert sum(d.values()) == pytest.approx(100.0, abs=1.0)
