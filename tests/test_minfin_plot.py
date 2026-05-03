"""Tests del módulo de plots MINFIN baseline."""

from __future__ import annotations

import pandas as pd
import pytest

from guatemala_sim.actions import PresupuestoAnual
from guatemala_sim.irl.candidates import generate_candidate_menu
from guatemala_sim.minfin_plot import (
    PARTIDAS_LABEL_CORTO,
    PARTIDAS_ORDEN,
    deviation_table,
    plot_budgets_vs_minfin,
    write_deviation_summary,
)


def _menu_dict() -> dict[str, PresupuestoAnual]:
    return {c.name: c.presupuesto for c in generate_candidate_menu()}


def test_partidas_orden_son_las_9_canonicas():
    assert len(PARTIDAS_ORDEN) == 9
    assert set(PARTIDAS_ORDEN) == set(PARTIDAS_LABEL_CORTO.keys())


def test_deviation_table_tiene_columna_minfin():
    df = deviation_table(_menu_dict())
    assert "MINFIN_2024" in df.columns


def test_deviation_table_tiene_fila_resumen_abs_dev_total():
    df = deviation_table(_menu_dict())
    assert "abs_dev_total" in df.index
    # MINFIN tiene desviación 0 (es el baseline)
    assert df.loc["abs_dev_total", "MINFIN_2024"] == 0.0


def test_deviation_table_indice_son_partidas_mas_resumen():
    df = deviation_table(_menu_dict())
    expected = list(PARTIDAS_ORDEN) + ["abs_dev_total"]
    assert df.index.tolist() == expected


def test_deviation_table_acepta_dict_como_input():
    """Debe aceptar dict[str, float] además de PresupuestoAnual."""
    menu = _menu_dict()
    inputs = {label: p.model_dump() for label, p in menu.items()}
    df = deviation_table(inputs)
    assert df.shape[1] == len(menu) + 1  # +1 por MINFIN


def test_deviation_table_status_quo_uniforme_es_simétrico():
    """Status quo uniforme reparte 11.11 por partida — su desviación
    absoluta total contra MINFIN debe ser igual a la suma de
    |11.11 - minfin[k]| para las 9 partidas."""
    df = deviation_table(_menu_dict())
    # MINFIN 2024 (aproximación):
    #   salud=12, educ=17, seg=7, infra=6, agro=4, prot=13, deuda=17, just=5, otros=19
    # status_quo todos a 11.11 → desviaciones esperadas:
    minfin = [12, 17, 7, 6, 4, 13, 17, 5, 19]
    esperada = sum(abs(11.11 - m) for m in minfin) + abs(11.12 - minfin[-1]) - abs(11.11 - minfin[-1])
    obtenida = float(df.loc["abs_dev_total", "status_quo_uniforme"])
    # tolerancia laxa por el 11.12 vs 11.11 en "otros"
    assert abs(obtenida - esperada) < 0.5


def test_plot_budgets_vs_minfin_genera_archivo(tmp_path):
    out = tmp_path / "test_comparison.png"
    p = plot_budgets_vs_minfin(_menu_dict(), out)
    assert p.exists()
    assert p.stat().st_size > 1000  # PNG no vacío


def test_write_deviation_summary_genera_markdown(tmp_path):
    out = tmp_path / "test_dev.md"
    p = write_deviation_summary(_menu_dict(), out)
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "MINFIN" in text
    assert "Modelos ordenados por proximidad al baseline humano" in text
    # Cada candidato del menú aparece
    for c in generate_candidate_menu():
        assert c.name in text


def test_plot_acepta_dict_de_dicts(tmp_path):
    """Backwards: el input puede ser dict[str, dict[partida, float]]."""
    menu = _menu_dict()
    inputs = {label: p.model_dump() for label, p in menu.items()}
    out = tmp_path / "x.png"
    plot_budgets_vs_minfin(inputs, out)
    assert out.exists()
