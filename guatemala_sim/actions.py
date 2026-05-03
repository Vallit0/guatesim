"""Espacio de acción: schema de la decisión presidencial por turno.

El LLM debe devolver un JSON que valide contra `DecisionTurno`.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, conlist, model_validator


class PresupuestoAnual(BaseModel):
    """Asignación porcentual del gasto público total. Debe sumar ~100."""

    model_config = ConfigDict(extra="forbid")

    salud: float = Field(ge=0, le=100)
    educacion: float = Field(ge=0, le=100)
    seguridad: float = Field(ge=0, le=100, description="mingob + ejército")
    infraestructura: float = Field(ge=0, le=100)
    agro_desarrollo_rural: float = Field(ge=0, le=100)
    proteccion_social: float = Field(ge=0, le=100)
    servicio_deuda: float = Field(ge=0, le=100)
    justicia: float = Field(ge=0, le=100)
    otros: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _suma_100(self) -> "PresupuestoAnual":
        total = (
            self.salud
            + self.educacion
            + self.seguridad
            + self.infraestructura
            + self.agro_desarrollo_rural
            + self.proteccion_social
            + self.servicio_deuda
            + self.justicia
            + self.otros
        )
        if abs(total - 100.0) > 1.0:  # tolerancia de 1 punto porcentual
            raise ValueError(
                f"Presupuesto debe sumar ~100 (toler. ±1). Suma actual: {total:.2f}"
            )
        return self

    def normalizado(self) -> "PresupuestoAnual":
        """Reescala al 100% exacto, útil para post-procesar la decisión del LLM."""
        total = (
            self.salud
            + self.educacion
            + self.seguridad
            + self.infraestructura
            + self.agro_desarrollo_rural
            + self.proteccion_social
            + self.servicio_deuda
            + self.justicia
            + self.otros
        )
        if total == 0:
            return self
        f = 100.0 / total
        return PresupuestoAnual(
            salud=self.salud * f,
            educacion=self.educacion * f,
            seguridad=self.seguridad * f,
            infraestructura=self.infraestructura * f,
            agro_desarrollo_rural=self.agro_desarrollo_rural * f,
            proteccion_social=self.proteccion_social * f,
            servicio_deuda=self.servicio_deuda * f,
            justicia=self.justicia * f,
            otros=self.otros * f,
        )


class Fiscal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta_iva_pp: float = Field(ge=-5, le=5, description="puntos porcentuales vs. base")
    delta_isr_pp: float = Field(ge=-10, le=10)
    aranceles_especificos: list[str] = Field(default_factory=list)


class PoliticaExterior(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alineamiento_priorizado: Literal["eeuu", "china", "multilateral", "regional", "neutral"]
    acciones_diplomaticas: Annotated[list[str], Field(max_length=3)] = Field(default_factory=list)


class RespuestaShock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shock: str
    medida: str
    costo_fiscal_pib: float = Field(description="% PIB gastado en la respuesta")


class Reforma(BaseModel):
    model_config = ConfigDict(extra="forbid")

    area: Literal[
        "catastro",
        "servicio_civil",
        "justicia",
        "tributaria",
        "electoral",
        "salud",
        "educacion",
    ]
    intensidad: Literal["incremental", "media", "radical"]
    costo_politico: float = Field(ge=0, le=100, description="auto-reportado por el LLM")


class DecisionTurno(BaseModel):
    """Decisión presidencial para un turno."""

    model_config = ConfigDict(extra="forbid")

    razonamiento: str = Field(
        min_length=1,
        description="Por qué el presidente tomó estas decisiones. Clave para análisis.",
    )
    presupuesto: PresupuestoAnual
    fiscal: Fiscal
    exterior: PoliticaExterior
    respuestas_shocks: list[RespuestaShock] = Field(default_factory=list)
    reformas: conlist(Reforma, max_length=2) = Field(default_factory=list)
    mensaje_al_pueblo: str = Field(min_length=1, max_length=600)


class ChosenDecision(BaseModel):
    """Schema para *menu-choice mode*: el presupuesto NO se compone libremente,
    se elige un índice del menú de candidatos. El resto de la decisión sí se
    compone normalmente.

    Habilita el likelihood Boltzmann tractable del IRL bayesiano (ver
    `guatemala_sim/irl/`). Para construir un `DecisionTurno` regular a partir
    de una `ChosenDecision` + el candidato elegido, usar `decision_from_choice`.
    """

    model_config = ConfigDict(extra="forbid")

    razonamiento: str = Field(
        min_length=1,
        description="Por qué se eligió ese candidato y se tomaron estas decisiones.",
    )
    chosen_index: int = Field(
        ge=0,
        le=4,
        description=(
            "Índice del candidato presupuestario elegido (0..4). El menú "
            "está fijo en 5 candidatos (ver guatemala_sim/irl/candidates.py)."
        ),
    )
    fiscal: Fiscal
    exterior: PoliticaExterior
    respuestas_shocks: list[RespuestaShock] = Field(default_factory=list)
    reformas: conlist(Reforma, max_length=2) = Field(default_factory=list)
    mensaje_al_pueblo: str = Field(min_length=1, max_length=600)


def decision_from_choice(
    chosen: ChosenDecision,
    presupuesto: PresupuestoAnual,
) -> DecisionTurno:
    """Construye un `DecisionTurno` válido desde una `ChosenDecision` + el
    presupuesto del candidato seleccionado.

    El `presupuesto` debe corresponder a `candidates[chosen.chosen_index]` —
    el caller es responsable de hacer la indexación. Este helper sólo
    ensambla los campos.
    """
    return DecisionTurno(
        razonamiento=chosen.razonamiento,
        presupuesto=presupuesto,
        fiscal=chosen.fiscal,
        exterior=chosen.exterior,
        respuestas_shocks=list(chosen.respuestas_shocks),
        reformas=list(chosen.reformas),
        mensaje_al_pueblo=chosen.mensaje_al_pueblo,
    )
