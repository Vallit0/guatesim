"""Movilización social: sindicatos magisteriales, organizaciones indígenas, iglesia."""

from __future__ import annotations

from ..actions import DecisionTurno
from ..state import GuatemalaState
from .base import AgenteBase, Impacto


class ProtestaSocial(AgenteBase):
    """Agregado de sindicatos, organizaciones indígenas y movimientos estudiantiles."""

    nombre = "protesta_social"

    def reaccionar(self, state: GuatemalaState, decision: DecisionTurno) -> Impacto:
        imp = Impacto()
        # pobreza alta + IVA arriba = movilización
        if state.social.pobreza_general > 55 and decision.fiscal.delta_iva_pp > 0.0:
            imp.delta_protesta += 6.0
            imp.delta_aprobacion -= 2.5
            imp.evento = "organizaciones indígenas convocan paro nacional"
        # recortar educación enfurece a magisterio
        p = decision.presupuesto.normalizado()
        if p.educacion < 13:
            imp.delta_protesta += 4.0
            imp.evento = (imp.evento + "; ") if imp.evento else ""
            imp.evento += "sindicato magisterial convoca marcha"
        # reforma tributaria que suba a los ricos puede reducir protesta
        for r in decision.reformas:
            if r.area == "tributaria" and decision.fiscal.delta_isr_pp > 0.5:
                imp.delta_protesta -= 2.0
            if r.area == "electoral" and r.intensidad != "radical":
                imp.delta_confianza = (imp.delta_confianza or 0.0) + 2.0
        return imp
