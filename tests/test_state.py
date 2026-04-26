"""Tests de validación y round-trip del estado."""

from __future__ import annotations

import json
from datetime import date

import pytest
from pydantic import ValidationError

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.state import GuatemalaState, Macro


def test_initial_state_valida():
    s = initial_state()
    assert s.turno.t == 0
    assert s.turno.fecha == date(2026, 1, 1)
    assert s.macro.pib_usd_mm > 0
    assert 0 <= s.social.pobreza_general <= 100


def test_round_trip_json():
    s = initial_state()
    as_json = s.model_dump_json()
    s2 = GuatemalaState.model_validate_json(as_json)
    assert s == s2


def test_round_trip_dict():
    s = initial_state()
    d = s.model_dump(mode="json")
    # debe poder serializarse a JSON (fechas, etc.)
    json.dumps(d)
    s2 = GuatemalaState.model_validate(d)
    assert s == s2


def test_macro_pib_debe_ser_positivo():
    with pytest.raises(ValidationError):
        Macro(
            pib_usd_mm=-1.0,
            crecimiento_pib=1.0,
            inflacion=1.0,
            deuda_pib=10.0,
            reservas_usd_mm=100.0,
            balance_fiscal_pib=-1.0,
            cuenta_corriente_pib=0.0,
            remesas_pib=10.0,
            tipo_cambio=7.75,
            ied_usd_mm=100.0,
        )


def test_rangos_sociales():
    with pytest.raises(ValidationError):
        s = initial_state()
        s.social.pobreza_general = 120.0  # fuera de rango [0, 100]
        GuatemalaState.model_validate(s.model_dump(mode="json"))


def test_memoria_capped():
    s = initial_state()
    s.memoria_presidencial = [f"doctrina {i}" for i in range(80)]
    s2 = GuatemalaState.model_validate(s.model_dump(mode="json"))
    assert len(s2.memoria_presidencial) == 50
