"""Modelo de estado de Guatemala (Pydantic).

El `GuatemalaState` es el único objeto que viaja entre módulos;
todo lo demás son funciones (relativamente) puras sobre él.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Macro(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pib_usd_mm: float = Field(gt=0, description="PIB en millones de USD")
    crecimiento_pib: float = Field(description="crecimiento real anual, %")
    inflacion: float = Field(description="inflación interanual, %")
    deuda_pib: float = Field(ge=0, description="deuda pública / PIB, %")
    reservas_usd_mm: float = Field(ge=0, description="reservas internacionales, millones USD")
    balance_fiscal_pib: float = Field(description="balance fiscal / PIB, % (negativo = déficit)")
    cuenta_corriente_pib: float = Field(description="cuenta corriente / PIB, %")
    remesas_pib: float = Field(ge=0, description="remesas / PIB, %")
    tipo_cambio: float = Field(gt=0, description="GTQ por USD")
    ied_usd_mm: float = Field(description="IED anual neta, millones USD")


class Social(BaseModel):
    model_config = ConfigDict(extra="forbid")

    poblacion_mm: float = Field(gt=0, description="millones de habitantes")
    pobreza_general: float = Field(ge=0, le=100)
    pobreza_extrema: float = Field(ge=0, le=100)
    gini: float = Field(ge=0, le=1)
    desempleo: float = Field(ge=0, le=100)
    informalidad: float = Field(ge=0, le=100)
    homicidios_100k: float = Field(ge=0)
    migracion_neta_miles: float = Field(description="miles, negativo = emigración neta")
    matricula_primaria: float = Field(ge=0, le=100)
    cobertura_salud: float = Field(ge=0, le=100)


class Politico(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aprobacion_presidencial: float = Field(ge=0, le=100)
    indice_protesta: float = Field(ge=0, le=100)
    confianza_institucional: float = Field(ge=0, le=100)
    coalicion_congreso: float = Field(ge=0, le=100, description="% escaños alineados con ejecutivo")
    libertad_prensa: float = Field(ge=0, le=100)


class Externo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alineamiento_eeuu: float = Field(ge=-1, le=1)
    alineamiento_china: float = Field(ge=-1, le=1)
    relacion_mexico: float = Field(ge=-1, le=1)
    relacion_triangulo_norte: float = Field(ge=-1, le=1, description="HND + SLV")
    apoyo_multilateral: float = Field(ge=0, le=100, description="acceso a BM/BID/FMI")


Periodo = Literal["Q1", "Q2", "Q3", "Q4", "anual"]


class Turno(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: int = Field(ge=0)
    fecha: date
    periodo: Periodo = "anual"


class GuatemalaState(BaseModel):
    """Estado total del país en el turno actual."""

    model_config = ConfigDict(extra="forbid")

    turno: Turno
    macro: Macro
    social: Social
    politico: Politico
    externo: Externo
    shocks_activos: list[str] = Field(default_factory=list)
    eventos_turno: list[str] = Field(default_factory=list)
    memoria_presidencial: list[str] = Field(default_factory=list)

    @field_validator("memoria_presidencial")
    @classmethod
    def _cap_memoria(cls, v: list[str]) -> list[str]:
        # Evitar que la memoria crezca sin límite — truncar a las últimas 50 entradas.
        return v[-50:]
