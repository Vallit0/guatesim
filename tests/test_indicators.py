"""Tests de los indicadores derivados."""

from __future__ import annotations

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.indicators import (
    coherencia_temporal,
    compute_indicators,
    diversidad_valores,
    indice_bienestar,
)


def test_indicadores_en_rango():
    s = initial_state()
    ind = compute_indicators(s)
    for v in (ind.bienestar, ind.gobernabilidad, ind.desarrollo_humano,
              ind.estabilidad_macro, ind.estres_social):
        assert 0.0 <= v <= 100.0


def test_bienestar_monotono_en_pobreza():
    s = initial_state()
    b1 = indice_bienestar(s)
    s.social.pobreza_general = 15.0
    b2 = indice_bienestar(s)
    assert b2 > b1


def test_coherencia_temporal():
    decisiones = [
        {"exterior": {"alineamiento_priorizado": "multilateral"}},
        {"exterior": {"alineamiento_priorizado": "multilateral"}},
        {"exterior": {"alineamiento_priorizado": "multilateral"}},
    ]
    assert coherencia_temporal(decisiones) == 100.0
    decisiones2 = [
        {"exterior": {"alineamiento_priorizado": "eeuu"}},
        {"exterior": {"alineamiento_priorizado": "china"}},
        {"exterior": {"alineamiento_priorizado": "eeuu"}},
    ]
    assert coherencia_temporal(decisiones2) == 0.0


def test_diversidad_valores():
    decisiones = [
        {"exterior": {"alineamiento_priorizado": "eeuu"}},
        {"exterior": {"alineamiento_priorizado": "eeuu"}},
    ]
    assert diversidad_valores(decisiones) == 0.0
