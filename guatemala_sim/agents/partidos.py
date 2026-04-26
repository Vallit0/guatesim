"""Partidos políticos en el Congreso."""

from __future__ import annotations

from ..actions import DecisionTurno
from ..state import GuatemalaState
from .base import AgenteBase, Impacto


class PartidoOficialista(AgenteBase):
    nombre = "oficialismo"

    def reaccionar(self, state: GuatemalaState, decision: DecisionTurno) -> Impacto:
        # el oficialismo pierde cohesión si hay reformas radicales
        n_radicales = sum(1 for r in decision.reformas if r.intensidad == "radical")
        delta_coal = -3.0 * n_radicales + 1.0  # leve soporte base
        evento = ""
        if n_radicales > 0:
            evento = "oficialismo presiona por moderar reformas radicales"
        return Impacto(delta_coalicion=delta_coal, evento=evento)


class CongresoOposicion(AgenteBase):
    nombre = "oposicion"

    def reaccionar(self, state: GuatemalaState, decision: DecisionTurno) -> Impacto:
        imp = Impacto()
        # sube IVA: oposición bloquea
        if decision.fiscal.delta_iva_pp > 0.5:
            imp.delta_coalicion -= 4.0
            imp.delta_protesta += 3.0
            imp.evento = "oposición bloquea alza de IVA en comisión"
        # alza ISR empresarial: depende del contexto, pero suele haber ruido
        if decision.fiscal.delta_isr_pp > 1.0:
            imp.delta_coalicion -= 2.0
        # recortar educación o salud: oposición hace campaña
        p = decision.presupuesto.normalizado()
        if p.educacion < 12 or p.salud < 8:
            imp.delta_aprobacion -= 2.0
            imp.evento = (imp.evento + "; ") if imp.evento else ""
            imp.evento += "oposición denuncia recorte social"
        return imp
