"""Tests del CountryProfile y la integración con bootstrap."""

from __future__ import annotations

from datetime import date

import pytest

from guatemala_sim.bootstrap import initial_state
from guatemala_sim.country_profile import (
    COUNTRY_PROFILES,
    CountryProfile,
    GTM,
    HND_STUB,
    get_country,
)
from guatemala_sim.state import GuatemalaState
from guatemala_sim.world.macro import MacroParams


# --- registry ---------------------------------------------------------------


def test_registry_contiene_gtm_y_hnd():
    assert "GTM" in COUNTRY_PROFILES
    assert "HND" in COUNTRY_PROFILES
    assert COUNTRY_PROFILES["GTM"] is GTM
    assert COUNTRY_PROFILES["HND"] is HND_STUB


def test_get_country_case_insensitive():
    assert get_country("gtm") is GTM
    assert get_country("GTM") is GTM


def test_get_country_falla_util():
    with pytest.raises(KeyError) as exc:
        get_country("XYZ")
    assert "no registrado" in str(exc.value)
    assert "GTM" in str(exc.value)


# --- GTM: idempotencia con bootstrap legacy ---------------------------------


def test_gtm_es_calibrated():
    assert GTM.calibration_status == "CALIBRATED"
    GTM.assert_calibrated_for_publication()  # no debe lanzar


def test_gtm_initial_state_idempotente_con_legacy():
    """Garantía crítica: pasar `country=GTM` debe producir el MISMO
    state que el bootstrap legacy. Si esto falla, los runs históricos
    del paper no se reproducen.
    """
    legacy = initial_state()
    via_gtm = initial_state(country=GTM)
    assert legacy.model_dump() == via_gtm.model_dump()


def test_gtm_macro_params_default_son_los_globales():
    """GTM usa los PARAMS globales (que históricamente fueron calibrados
    contra Guatemala). No deben divergir sin advertencia."""
    from guatemala_sim.world.macro import PARAMS
    # No exigimos identidad de objeto, sí de valores
    assert GTM.macro_params.crecimiento_tendencial == PARAMS.crecimiento_tendencial
    assert GTM.macro_params.inflacion_objetivo_banguat == PARAMS.inflacion_objetivo_banguat


# --- HND_STUB: gate de honestidad -------------------------------------------


def test_hnd_marcado_uncalibrated():
    assert HND_STUB.calibration_status == "UNCALIBRATED"
    assert "STUB" in HND_STUB.notes or "no calibrado" in HND_STUB.notes.lower()


def test_hnd_assert_calibrated_falla():
    """Si alguien intenta publicar con HND_STUB, el gate debe atrapar."""
    with pytest.raises(RuntimeError) as exc:
        HND_STUB.assert_calibrated_for_publication()
    msg = str(exc.value)
    assert "UNCALIBRATED" in msg
    assert "HND" in msg


def test_hnd_initial_state_es_valido_pydantic():
    """Aunque sea stub, el estado debe pasar las validaciones Pydantic
    del schema (rangos, no-negatividad, etc.). Si no, el pipeline
    rompe en el primer turno."""
    s = HND_STUB.initial_state()
    assert isinstance(s, GuatemalaState)
    # Re-validar serializando + deserializando
    s2 = GuatemalaState.model_validate(s.model_dump())
    assert s2.macro.pib_usd_mm > 0
    assert 0 <= s2.social.pobreza_general <= 100
    assert 0 <= s2.politico.aprobacion_presidencial <= 100


def test_hnd_difiere_de_gtm():
    """Sanity: HND y GTM no son el mismo estado disfrazado."""
    a = GTM.initial_state()
    b = HND_STUB.initial_state()
    assert a.macro.pib_usd_mm != b.macro.pib_usd_mm
    assert a.social.poblacion_mm != b.social.poblacion_mm


# --- bootstrap integration --------------------------------------------------


def test_initial_state_sin_args_no_se_rompe():
    """Backward compat: TODOS los tests existentes llaman
    `initial_state()` sin args. Esa firma debe seguir funcionando."""
    s = initial_state()
    assert isinstance(s, GuatemalaState)


def test_initial_state_con_country_hnd():
    s = initial_state(country=HND_STUB)
    assert s.turno.fecha == date(2026, 1, 1)
    assert s.macro.tipo_cambio == 24.8  # HNL/USD


def test_initial_state_rechaza_objeto_invalido():
    with pytest.raises(TypeError, match="CountryProfile"):
        initial_state(country="GTM")  # string en vez de profile


# --- CountryProfile en sí ---------------------------------------------------


def test_country_profile_es_frozen():
    """Frozen para prevenir mutación accidental compartida entre tests."""
    with pytest.raises(Exception):  # FrozenInstanceError de dataclasses
        GTM.iso3 = "XXX"  # type: ignore[misc]


def test_country_profile_macro_params_son_macro_params():
    assert isinstance(GTM.macro_params, MacroParams)
    assert isinstance(HND_STUB.macro_params, MacroParams)
