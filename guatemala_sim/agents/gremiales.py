"""Gremiales empresariales (CACIF)."""

from __future__ import annotations

from ..actions import DecisionTurno
from ..state import GuatemalaState
from .base import AgenteBase, Impacto


class CACIF(AgenteBase):
    nombre = "cacif"

    def reaccionar(self, state: GuatemalaState, decision: DecisionTurno) -> Impacto:
        imp = Impacto()
        # alza ISR asusta a CACIF
        if decision.fiscal.delta_isr_pp > 0.5:
            imp.delta_ied_mm = -80.0 * decision.fiscal.delta_isr_pp
            imp.delta_aprobacion = -1.5
            imp.evento = "CACIF advierte sobre alza de ISR y anuncia pausa en inversión"
        # alza IVA no les molesta tanto; alivio tributario gusta
        if decision.fiscal.delta_isr_pp < -0.5:
            imp.delta_ied_mm = 120.0 * abs(decision.fiscal.delta_isr_pp)
            imp.evento = "CACIF celebra alivio tributario y anuncia inversiones"
        # reformas tributarias radicales = mucho ruido
        for r in decision.reformas:
            if r.area == "tributaria" and r.intensidad == "radical":
                imp.delta_ied_mm += -200.0
                imp.delta_confianza = -3.0
                imp.evento = "CACIF llama a diálogo urgente por reforma tributaria radical"
        return imp
