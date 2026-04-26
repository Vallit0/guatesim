"""Tests del schema de decisión presidencial."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guatemala_sim.actions import (
    DecisionTurno,
    Fiscal,
    PoliticaExterior,
    PresupuestoAnual,
    Reforma,
)


def _presupuesto_valido() -> PresupuestoAnual:
    return PresupuestoAnual(
        salud=12,
        educacion=18,
        seguridad=12,
        infraestructura=12,
        agro_desarrollo_rural=8,
        proteccion_social=10,
        servicio_deuda=10,
        justicia=6,
        otros=12,
    )


def _decision_valida() -> DecisionTurno:
    return DecisionTurno(
        razonamiento="priorizar pobreza rural y estabilidad macro",
        presupuesto=_presupuesto_valido(),
        fiscal=Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.5, aranceles_especificos=[]),
        exterior=PoliticaExterior(
            alineamiento_priorizado="multilateral",
            acciones_diplomaticas=["reunión con BID"],
        ),
        respuestas_shocks=[],
        reformas=[Reforma(area="tributaria", intensidad="incremental", costo_politico=20.0)],
        mensaje_al_pueblo="seguimos con prudencia fiscal y enfoque territorial.",
    )


def test_decision_valida_round_trip():
    d = _decision_valida()
    d2 = DecisionTurno.model_validate_json(d.model_dump_json())
    assert d == d2


def test_presupuesto_debe_sumar_100():
    with pytest.raises(ValidationError):
        PresupuestoAnual(
            salud=50, educacion=50, seguridad=50,
            infraestructura=0, agro_desarrollo_rural=0,
            proteccion_social=0, servicio_deuda=0,
            justicia=0, otros=0,
        )


def test_presupuesto_normalizado():
    # model_construct salta la validación (el total acá es 200) para probar el normalizador.
    p = PresupuestoAnual.model_construct(
        salud=24, educacion=36, seguridad=24,
        infraestructura=24, agro_desarrollo_rural=16,
        proteccion_social=20, servicio_deuda=20,
        justicia=12, otros=24,
    )
    n = p.normalizado()
    total = (
        n.salud + n.educacion + n.seguridad + n.infraestructura
        + n.agro_desarrollo_rural + n.proteccion_social
        + n.servicio_deuda + n.justicia + n.otros
    )
    assert abs(total - 100.0) < 1e-6


def test_fiscal_rangos():
    with pytest.raises(ValidationError):
        Fiscal(delta_iva_pp=10.0, delta_isr_pp=0.0)


def test_reformas_limite_2():
    with pytest.raises(ValidationError):
        DecisionTurno(
            razonamiento="test",
            presupuesto=_presupuesto_valido(),
            fiscal=Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.0),
            exterior=PoliticaExterior(
                alineamiento_priorizado="neutral",
                acciones_diplomaticas=[],
            ),
            reformas=[
                Reforma(area="salud", intensidad="media", costo_politico=10.0),
                Reforma(area="educacion", intensidad="media", costo_politico=10.0),
                Reforma(area="justicia", intensidad="media", costo_politico=10.0),
            ],
            mensaje_al_pueblo="test",
        )
