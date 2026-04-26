"""Tests de los agentes Mesa."""

from __future__ import annotations

from guatemala_sim.actions import (
    DecisionTurno,
    Fiscal,
    PoliticaExterior,
    PresupuestoAnual,
)
from guatemala_sim.agents import (
    AgentesModel,
    CACIF,
    CongresoOposicion,
    PartidoOficialista,
    ProtestaSocial,
)
from guatemala_sim.bootstrap import initial_state


def _presupuesto_normal() -> PresupuestoAnual:
    return PresupuestoAnual(
        salud=12, educacion=18, seguridad=11, infraestructura=14,
        agro_desarrollo_rural=8, proteccion_social=10, servicio_deuda=10,
        justicia=5, otros=12,
    )


def _decision(fiscal_delta_iva: float = 0.0, fiscal_delta_isr: float = 0.0) -> DecisionTurno:
    return DecisionTurno(
        razonamiento="test",
        presupuesto=_presupuesto_normal(),
        fiscal=Fiscal(delta_iva_pp=fiscal_delta_iva, delta_isr_pp=fiscal_delta_isr),
        exterior=PoliticaExterior(alineamiento_priorizado="multilateral", acciones_diplomaticas=[]),
        mensaje_al_pueblo="test",
    )


def test_agentes_reaccionan_a_iva():
    state = initial_state()
    model = AgentesModel([PartidoOficialista, CongresoOposicion, CACIF, ProtestaSocial], seed=1)
    _, impactos = model.step_con_decision(state, _decision(fiscal_delta_iva=2.0))
    # alza de IVA debería generar evento de oposición
    eventos = [i.evento for i in impactos if i.evento]
    assert any("IVA" in e or "iva" in e.lower() for e in eventos)


def test_cacif_responde_a_alza_isr():
    state = initial_state()
    model = AgentesModel([CACIF], seed=1)
    _, impactos = model.step_con_decision(state, _decision(fiscal_delta_isr=2.0))
    assert impactos[0].delta_ied_mm < 0
    assert impactos[0].evento != ""


def test_magisterio_reclama_si_recortan_educacion():
    state = initial_state()
    model = AgentesModel([ProtestaSocial], seed=1)
    d = _decision()
    # subir "otros" y bajar educación al mínimo: 12→20 otros, 18→10 educación
    p_raw = d.presupuesto.model_dump()
    p_raw["educacion"] = 10
    p_raw["otros"] = 20
    from guatemala_sim.actions import PresupuestoAnual as PA
    d.presupuesto = PA(**p_raw)
    _, impactos = model.step_con_decision(state, d)
    assert impactos[0].delta_protesta > 0
