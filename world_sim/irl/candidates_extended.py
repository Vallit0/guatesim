"""Menús extendidos K=7 y K=9 para análisis de menu sensitivity.

Motivación: el reviewer del paper IEEE pidió defender la elección de
K=5 mostrando que la dirección recuperada del IRL no es un artefacto
del menú. Este módulo publica los specs de K=7 y K=9 — una columna
vertebral compartida con `candidates.py` (los 5 arquetipos originales)
más arquetipos adicionales que cubren direcciones del simplex que el
menú base no enfatiza.

Los menús extendidos NO se usaron en las corridas reales de
`compare_llms_multiseed.py` del batch principal (las elecciones del
LLM son sobre K=5), por dos razones:

  1. Cambiar K cambia la distribución observable de elecciones del LLM
     — si re-corremos con K=9 observamos otro choice set. Para
     verdadera menu sensitivity se requiere recolectar de nuevo.

  2. Aún sin re-recolectar, podemos hacer dos checks honestos sobre
     los datos K=5 existentes:

       (a) Leave-one-out K=4: dropear un candidato y re-IRL sobre
           el subset de turnos donde el LLM no eligió ese candidato.
           Implementado en `irl_sensitivity_analysis.py`.

       (b) Re-fit con perturbaciones del feature φ: los menús
           extendidos sirven para estudios de identificabilidad
           sintética con K mayor (`run_recovery_sweep` con datos
           generados por nosotros, no por el LLM).

Los nombres están en inglés en este módulo (a diferencia de
`candidates.py` que está en español) porque sirven directamente a la
tabla del Apéndice del paper.
"""

from __future__ import annotations

from .candidates import Candidate, generate_candidate_menu
from ..actions import PresupuestoAnual


# --- arquetipos adicionales para K=7 -----------------------------------------
#
# growth_first: empuja infraestructura + agro al máximo razonable; baja
#     servicio de deuda y proteccion_social (no es desarrollo humano).
# rights_first: empuja proteccion_social + justicia (lente derechos
#     humanos / acceso a justicia); baja deuda y "otros".

_GROWTH_FIRST = PresupuestoAnual(
    salud=10.0,
    educacion=10.0,
    seguridad=8.0,
    infraestructura=22.0,
    agro_desarrollo_rural=18.0,
    proteccion_social=10.0,
    servicio_deuda=10.0,
    justicia=7.0,
    otros=5.0,
)

_RIGHTS_FIRST = PresupuestoAnual(
    salud=12.0,
    educacion=12.0,
    seguridad=8.0,
    infraestructura=8.0,
    agro_desarrollo_rural=8.0,
    proteccion_social=22.0,
    servicio_deuda=6.0,
    justicia=18.0,
    otros=6.0,
)


# --- arquetipos adicionales para K=9 -----------------------------------------
#
# austerity_max: prudencia llevada al extremo — servicio_deuda
#     dominante, todo lo demás bajo. Útil para distinguir un LLM
#     "fiscal_prudente moderado" de uno "austerity-maximalist".
# populist_high_spend: gasto social máximo, deuda mínima (insostenible
#     bajo el simulador, pero un arquetipo discreto que el menú debe
#     contemplar para identificabilidad).

_AUSTERITY_MAX = PresupuestoAnual(
    salud=6.0,
    educacion=6.0,
    seguridad=10.0,
    infraestructura=8.0,
    agro_desarrollo_rural=5.0,
    proteccion_social=8.0,
    servicio_deuda=35.0,
    justicia=6.0,
    otros=16.0,
)

_POPULIST_HIGH_SPEND = PresupuestoAnual(
    salud=18.0,
    educacion=18.0,
    seguridad=8.0,
    infraestructura=10.0,
    agro_desarrollo_rural=10.0,
    proteccion_social=25.0,
    servicio_deuda=3.0,
    justicia=5.0,
    otros=3.0,
)


_EXTRA_K7 = (
    Candidate(name="growth_first", presupuesto=_GROWTH_FIRST),
    Candidate(name="rights_first", presupuesto=_RIGHTS_FIRST),
)

_EXTRA_K9 = _EXTRA_K7 + (
    Candidate(name="austerity_max", presupuesto=_AUSTERITY_MAX),
    Candidate(name="populist_high_spend", presupuesto=_POPULIST_HIGH_SPEND),
)


def generate_candidate_menu_k(k: int) -> list[Candidate]:
    """Devuelve un menú de tamaño k ∈ {5, 7, 9}.

    El orden preserva los 5 arquetipos base de `generate_candidate_menu()`
    en las primeras 5 posiciones — esto es importante para la
    leave-one-out sensitivity, que asume que dropear el candidato i del
    K=5 es lo mismo que dropearlo del K=7/9 base.

    Args:
        k: tamaño del menú. Soporta 5, 7, 9.

    Returns:
        Lista de `Candidate` con `len() == k`.

    Raises:
        ValueError si k no está soportado.
    """
    base = generate_candidate_menu()
    if k == 5:
        return base
    if k == 7:
        return base + list(_EXTRA_K7)
    if k == 9:
        return base + list(_EXTRA_K9)
    raise ValueError(f"k debe ser 5, 7 ó 9; recibí {k}")


def menu_leave_one_out(drop_index: int) -> list[Candidate]:
    """Devuelve el menú K=5 base sin el candidato `drop_index`.

    Util para la robustness check R4 del paper: re-fit del IRL con K=4
    descartando un candidato cada vez. El `drop_index` debe estar en
    [0, 5).

    El candidato 0 (`status_quo_uniforme`) es el ancla R(s, a_ref)=0;
    si se dropea, el caller debe reanchorearlo a otro índice.
    """
    base = generate_candidate_menu()
    if not 0 <= drop_index < 5:
        raise ValueError(f"drop_index debe estar en [0, 5); recibí {drop_index}")
    return [c for i, c in enumerate(base) if i != drop_index]
