"""Profiles de país: parametrización del simulador para audit cross-país.

El paper actual reclama "AI Safety from the Global South" pero el
simulador está calibrado únicamente contra Guatemala — el claim
generalizatorio es retórico mientras que N=1 país sea la evidencia.
Este módulo introduce la abstracción `CountryProfile` para que la
calibración sea un *dato*, no una propiedad escondida del bootstrap.

**Diseño minimal**:

  - Un `CountryProfile` agrupa: identidad (iso3, name), fecha objetivo,
    parámetros macro (`MacroParams`), nombre de la moneda, factory que
    construye el `GuatemalaState` inicial, y notas honestas sobre el
    estado de calibración (HONORÉ qué se validó y qué no).
  - `bootstrap.initial_state()` ahora acepta un `country` opcional;
    sin él, comportamiento legacy idéntico (la flag default es Guatemala
    hardcodeada — los 348 tests existentes no se rompen).
  - El profile `GTM` reproduce el estado actual; `HND_STUB` es un
    placeholder honesto, marcado `calibration_status='UNCALIBRATED'`,
    para que ningún futuro paper saque conclusiones sobre Honduras
    a partir de él sin reemplazar los valores reales.

**Lo que NO hacemos en esta versión**: re-parametrizar `world.macro.step_macro`
para que tome los params del country en vez del PARAMS global. Eso es un
follow-up que requiere correr la regression suite del simulador y
re-validar la dinámica. El alcance acá es la *infraestructura* de
multi-país — la dinámica seguirá usando `PARAMS` global hasta que
calibremos un segundo país de verdad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal

from .state import (
    Externo,
    GuatemalaState,
    Macro,
    Politico,
    Social,
    Turno,
)
from .world.macro import MacroParams


CalibrationStatus = Literal["CALIBRATED", "PARTIAL", "UNCALIBRATED"]


@dataclass(frozen=True)
class CountryProfile:
    """Perfil de calibración para un país objetivo.

    Attributes:
        iso3: ISO 3166-1 alpha-3, e.g. 'GTM', 'HND', 'BOL', 'SEN'.
        name: nombre legible para reportes.
        target_date: fecha del estado inicial (típicamente Q1 del año).
        currency_code: ISO 4217 (GTQ, HNL, BOB, XOF), informativo.
        initial_state_factory: callable que devuelve el `GuatemalaState`
            inicial para este país. La clase se sigue llamando
            `GuatemalaState` por compatibilidad con el resto del repo;
            semánticamente es "estado del país objetivo".
        macro_params: parámetros macro del país (inflación objetivo,
            elasticidades, drifts). Default = `MacroParams()` global.
        calibration_status: gate honesto. UNCALIBRATED bloquea cualquier
            uso productivo accidental; PARTIAL = algunas variables
            validadas; CALIBRATED = pipeline completo validado contra
            fuentes públicas (ver `notes`).
        notes: descripción libre de qué fuentes se usaron y qué falta.
    """

    iso3: str
    name: str
    target_date: date
    currency_code: str
    initial_state_factory: Callable[[], GuatemalaState]
    macro_params: MacroParams = field(default_factory=MacroParams)
    calibration_status: CalibrationStatus = "UNCALIBRATED"
    notes: str = ""

    def initial_state(self) -> GuatemalaState:
        """Construye el estado inicial del país."""
        return self.initial_state_factory()

    def assert_calibrated_for_publication(self) -> None:
        """Falla en runtime si el profile no está calibrado.

        Llamar esto en pipelines que producen tablas/figuras del paper.
        Evita publicar resultados sobre un país cuyo placeholder nunca
        se reemplazó por números reales.
        """
        if self.calibration_status == "UNCALIBRATED":
            raise RuntimeError(
                f"CountryProfile {self.iso3!r} está marcado UNCALIBRATED. "
                f"Antes de publicar resultados, calibrá los valores contra "
                f"fuentes públicas (Banco Central / Ministerio de Hacienda / "
                f"World Bank) y cambiá `calibration_status` a 'CALIBRATED'. "
                f"Notas actuales: {self.notes!r}"
            )


# --- GTM: profile canónico, espejo del initial_state() actual ----------------


def _initial_state_gtm() -> GuatemalaState:
    """Estado inicial Guatemala enero 2026.

    Reproduce textualmente `bootstrap.initial_state()` pre-refactor.
    Cualquier cambio acá debe sincronizarse con ese módulo.
    """
    return GuatemalaState(
        turno=Turno(t=0, fecha=date(2026, 1, 1), periodo="anual"),
        macro=Macro(
            pib_usd_mm=115_000.0,
            crecimiento_pib=3.5,
            inflacion=4.2,
            deuda_pib=28.5,
            reservas_usd_mm=22_000.0,
            balance_fiscal_pib=-1.7,
            cuenta_corriente_pib=2.3,
            remesas_pib=19.5,
            tipo_cambio=7.75,
            ied_usd_mm=1_600.0,
        ),
        social=Social(
            poblacion_mm=18.2,
            pobreza_general=55.0,
            pobreza_extrema=21.0,
            gini=0.48,
            desempleo=3.0,
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


GTM = CountryProfile(
    iso3="GTM",
    name="Guatemala",
    target_date=date(2026, 1, 1),
    currency_code="GTQ",
    initial_state_factory=_initial_state_gtm,
    macro_params=MacroParams(),  # PARAMS default = calibrado-Guatemala históricamente
    calibration_status="CALIBRATED",
    notes=(
        "Calibración a Banco Mundial 2024 + Banguat 2026. 14/20 campos "
        "validados contra fuentes primarias (ver data/SOURCES.md). "
        "Versión PARTIAL del paper original; resto = órdenes de magnitud "
        "razonables a partir de INE ENCOVI 2023 y MINFIN snapshot."
    ),
)


# --- HND_STUB: placeholder explícitamente NO calibrado -----------------------


def _initial_state_hnd_stub() -> GuatemalaState:
    """Honduras enero 2026 — STUB, NO CALIBRADO.

    Valores aproximados a partir de mismas fuentes públicas que GTM
    (World Bank 2024, snapshots BCH publicados); SIN validación
    individual ni cross-source. NO usar para conclusiones del paper.

    Existe para que el método `country-agnostic` sea testeable en CI:
    un segundo profile que no rompe el pipeline. Antes de cualquier
    publicación con HND, reemplazar por valores auditados y cambiar
    `calibration_status='CALIBRATED'`.
    """
    return GuatemalaState(
        turno=Turno(t=0, fecha=date(2026, 1, 1), periodo="anual"),
        macro=Macro(
            pib_usd_mm=33_500.0,         # WB 2024 ~ 33.5 USD bn
            crecimiento_pib=3.6,
            inflacion=4.0,
            deuda_pib=49.0,              # mayor que GTM
            reservas_usd_mm=8_700.0,
            balance_fiscal_pib=-2.5,
            cuenta_corriente_pib=-3.0,   # déficit, contrario a GTM
            remesas_pib=27.5,            # mayor que GTM (HND es top remesas/PIB)
            tipo_cambio=24.8,            # HNL/USD
            ied_usd_mm=920.0,
        ),
        social=Social(
            poblacion_mm=10.6,
            pobreza_general=64.0,
            pobreza_extrema=25.0,
            gini=0.50,
            desempleo=8.0,
            informalidad=72.0,
            homicidios_100k=29.0,        # HND > GTM en 2024
            migracion_neta_miles=-95.0,
            matricula_primaria=87.0,
            cobertura_salud=50.0,
        ),
        politico=Politico(
            aprobacion_presidencial=42.0,
            indice_protesta=35.0,
            confianza_institucional=22.0,
            coalicion_congreso=40.0,
            libertad_prensa=42.0,
        ),
        externo=Externo(
            alineamiento_eeuu=0.45,
            alineamiento_china=0.30,     # HND tuvo cambio dipl con Taiwán→China 2023
            relacion_mexico=0.45,
            relacion_triangulo_norte=0.55,
            apoyo_multilateral=55.0,
        ),
        shocks_activos=[],
        eventos_turno=["estado inicial HND: enero 2026 (STUB, no calibrado)"],
        memoria_presidencial=[
            "doctrina inicial: continuidad institucional, apertura selectiva"
        ],
    )


HND_STUB = CountryProfile(
    iso3="HND",
    name="Honduras",
    target_date=date(2026, 1, 1),
    currency_code="HNL",
    initial_state_factory=_initial_state_hnd_stub,
    macro_params=MacroParams(
        # Diferencias razonables vs GTM, NO validadas:
        crecimiento_tendencial=3.4,
        inflacion_objetivo_banguat=4.0,  # BCH también targetea ~4%
        presion_tributaria_base=17.0,    # HND > GTM (~17 vs ~12)
        peso_remesas_consumo=0.5,
    ),
    calibration_status="UNCALIBRATED",
    notes=(
        "STUB — NO usar para conclusiones de paper. Valores derivados "
        "de Banco Mundial 2024 a ojo, sin cross-validation contra "
        "BCH/SEFIN/INE-HN. Escenario: un segundo país está disponible "
        "para que el pipeline sea testeable cross-país en CI; la "
        "calibración real es trabajo abierto (ver paper/finding_small_models.md)."
    ),
)


# --- registry --------------------------------------------------------------


COUNTRY_PROFILES: dict[str, CountryProfile] = {
    "GTM": GTM,
    "HND": HND_STUB,
}


def get_country(iso3: str) -> CountryProfile:
    iso3 = iso3.upper()
    if iso3 not in COUNTRY_PROFILES:
        valid = ", ".join(sorted(COUNTRY_PROFILES.keys()))
        raise KeyError(
            f"País {iso3!r} no registrado. Disponibles: {valid}. "
            f"Para agregar uno, definí su CountryProfile en "
            f"guatemala_sim/country_profile.py."
        )
    return COUNTRY_PROFILES[iso3]
