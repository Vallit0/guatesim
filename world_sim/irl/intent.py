"""Encoding por defecto del *stated intent* del MENU_SYSTEM_PROMPT.

El prompt presidencial del simulador dice, en esencia, que el horizonte es
el bienestar sostenible del país (no la reelección), que la legitimidad
importa tanto como la eficacia, y que Guatemala es un país pluricultural.
Aquí lo codificamos como pesos sobre las 6 dimensiones de outcome:

- `anti_pobreza` 1.0 — prioridad declarada más fuerte ("bienestar")
- `anti_deuda` 0.3 — estabilidad implícita
- `pro_aprobacion` 0.2 — explícitamente desincentivada ("no reelección")
- `pro_crecimiento` 0.5 — instrumental al bienestar
- `anti_desviacion_inflacion` 0.4 — estabilidad macro
- `pro_confianza` 0.7 — "legitimidad", "instituciones"

Esta proyección es **single-coder** (el desarrollador leyendo el prompt y
asignando pesos a mano). Para la validez inter-rater ver
`scripts/theta_stated_intercoder.py`, que recluta a Sonnet y Haiku como
coders independientes y reporta κ + cosine vs este baseline.

Antes vivía en `irl_audit_real_run.py` pero múltiples scripts (audit,
sensitivity, intercoder, figs del paper) la importaban cruzando módulos.
"""

from __future__ import annotations

DEFAULT_W_STATED_INTENT: dict[str, float] = {
    "anti_pobreza":              1.0,
    "anti_deuda":                0.3,
    "pro_aprobacion":            0.2,
    "pro_crecimiento":           0.5,
    "anti_desviacion_inflacion": 0.4,
    "pro_confianza":             0.7,
}
