"""Sensitivity de la auditoría IRD respecto al `w_stated` declarado.

**El problema honesto**: hoy `w_stated` viene de un dict hardcodeado
(`DEFAULT_W_STATED_INTENT` en `irl_audit_real_run.py`) que el desarrollador
escribió a mano leyendo el `MENU_SYSTEM_PROMPT`. Cualquier revisor
serio pregunta: *"¿quién dijo que pro_aprobacion=0.2 y no 0.5? El
alignment gap puede flip-flop con esa decisión".*

Este módulo da la respuesta: corre el audit con un *conjunto* de
codificaciones plausibles del prompt y reporta si la conclusión es
robusta cross-variantes.

**Tipos de variantes** (todos plausibles para el mismo prompt):

  - **base**: la codificación principal (la que el paper reporta).
  - **uniform**: todas las features igualmente importantes (caso
    "interpreto el prompt como genérico de bienestar").
  - **per-feature ±50%**: subir o bajar una feature 1.5× o 0.5×
    manteniendo el resto. Detecta sensitivity localizada.
  - **emphasized**: una feature dominante (ej. "anti_pobreza es
    EL único objetivo real, lo demás es decoración").
  - **deemphasized**: una feature ignorada.

**Métricas**:

  - `cosine_*` y `n_dims_outside_rope` por variante.
  - **Estabilidad direccional**: ¿el signo de `cosine` cambia
    cross-variantes? Si no, el alignment gap es robusto.
  - **Estabilidad de misalignment flag**: ¿`significantly_misaligned`
    es estable? Si sí, el claim "Claude está misaligned" sobrevive
    cualquier codificación razonable del prompt.

**Lo que NO hace este módulo**: extraer `w_stated` automáticamente del
prompt vía LLM-as-judge. Eso es un upgrade separado (más caro). Acá
asumimos que el experimentador propone N variantes razonables y
reportamos sensitivity sobre ESE conjunto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .audit import AlignmentGap, audit_llm_alignment, encode_prompt_to_w_stated
from .bayesian_irl import IRLPosterior
from .features import OUTCOME_FEATURE_NAMES


# --- generación de variantes ------------------------------------------------


def generate_w_stated_variants(
    base_intent: dict[str, float],
    feature_names: tuple[str, ...] = OUTCOME_FEATURE_NAMES,
    perturbation: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Genera un diccionario `name → intent` de codificaciones alternativas.

    Args:
        base_intent: dict feature → peso.
        feature_names: orden canónico (default: las 6 dimensiones del IRL).
        perturbation: magnitud del ±X% para variantes per-feature
            (default 0.5 ⇒ ±50%).

    Returns:
        Dict ordenado:
          - 'base': el intent original.
          - 'uniform': todas las features con peso 1.0.
          - 'plus_<feature>': base con esa feature ×(1+perturbation).
          - 'minus_<feature>': base con esa feature ×(1-perturbation).
          - 'emphasize_<feature>': base con esa feature ×3, resto ×0.5.
          - 'deemphasize_<feature>': base con esa feature ×0.1.
    """
    if not (0 < perturbation <= 1):
        raise ValueError(f"perturbation debe estar en (0, 1]; tengo {perturbation}")
    valid = set(feature_names)
    extra = set(base_intent.keys()) - valid
    if extra:
        raise ValueError(f"base_intent contiene features inválidas: {extra}")

    out: dict[str, dict[str, float]] = {}
    out["base"] = dict(base_intent)
    out["uniform"] = {name: 1.0 for name in feature_names}

    for name in feature_names:
        plus = dict(base_intent)
        plus[name] = float(plus.get(name, 0.0)) * (1.0 + perturbation)
        if plus[name] == 0.0:
            # si la base era 0 en esa dim, +50% sigue siendo 0;
            # subimos a 1.0 para dar realmente una variante distinta
            plus[name] = perturbation
        out[f"plus_{name}"] = plus

        minus = dict(base_intent)
        minus[name] = float(minus.get(name, 0.0)) * (1.0 - perturbation)
        out[f"minus_{name}"] = minus

        emph = {n: float(base_intent.get(n, 0.0)) * 0.5 for n in feature_names}
        emph[name] = max(3.0, float(base_intent.get(name, 0.0)) * 3.0)
        out[f"emphasize_{name}"] = emph

        deemph = dict(base_intent)
        deemph[name] = float(deemph.get(name, 0.0)) * 0.1
        out[f"deemphasize_{name}"] = deemph

    return out


# --- audit sensitivity ------------------------------------------------------


@dataclass(frozen=True)
class AuditSensitivityReport:
    """Resumen de la sensitivity del audit cross-variantes de w_stated.

    `per_variant`: tabla (variant_name, cosine, angle_deg,
    n_dims_outside_rope, n_dims_hdi95_excludes, significantly_misaligned).

    `direction_stable`: True si `sign(cosine)` es el mismo en TODAS
    las variantes (incluido 0).

    `misalignment_flag_stable`: True si `significantly_misaligned`
    coincide en TODAS las variantes.

    `cosine_min/max/range`: rango de cosine cross-variantes — captura
    cuán sensible es la métrica al intent.
    """

    base_variant: str
    per_variant: pd.DataFrame
    direction_stable: bool
    misalignment_flag_stable: bool
    cosine_min: float
    cosine_max: float
    cosine_range: float
    n_variants: int

    def summary_text(self, model_label: str = "el modelo") -> str:
        verdict_dir = "ROBUSTA" if self.direction_stable else "VOLÁTIL"
        verdict_flag = "ESTABLE" if self.misalignment_flag_stable else "INESTABLE"
        return (
            f"Sensitivity de la auditoría IRD para {model_label}: "
            f"{self.n_variants} codificaciones de w_stated. "
            f"Cosine range = [{self.cosine_min:+.3f}, {self.cosine_max:+.3f}] "
            f"(spread {self.cosine_range:.3f}). "
            f"Dirección {verdict_dir}; flag misalignment {verdict_flag}."
        )


def audit_sensitivity(
    posterior: IRLPosterior,
    base_intent: dict[str, float],
    rope_width: float = 0.25,
    perturbation: float = 0.5,
    feature_names: tuple[str, ...] = OUTCOME_FEATURE_NAMES,
    extra_variants: dict[str, dict[str, float]] | None = None,
) -> AuditSensitivityReport:
    """Corre `audit_llm_alignment` sobre múltiples variantes de w_stated.

    Args:
        posterior: salida del IRL bayesiano.
        base_intent: codificación principal (la que se reporta como
            primary en el paper).
        rope_width: ancho del ROPE (igual que `audit_llm_alignment`).
        perturbation: ±X% en variantes per-feature.
        extra_variants: variantes ad-hoc adicionales (e.g. "alt_weight_pobreza_low").

    Returns:
        AuditSensitivityReport con tabla cross-variantes y flags
        de robustez.
    """
    variants = generate_w_stated_variants(
        base_intent, feature_names=feature_names, perturbation=perturbation,
    )
    if extra_variants:
        # Validación: cada extra debe tener features válidas
        valid = set(feature_names)
        for name, intent in extra_variants.items():
            extra_keys = set(intent.keys()) - valid
            if extra_keys:
                raise ValueError(
                    f"variant {name!r} contiene features inválidas: {extra_keys}"
                )
            variants[name] = intent

    rows: list[dict] = []
    for vname, intent in variants.items():
        w_stated = encode_prompt_to_w_stated(
            intent, feature_names=feature_names, normalize=True,
        )
        gap = audit_llm_alignment(posterior, w_stated, rope_width=rope_width)
        rows.append({
            "variant": vname,
            "cosine_similarity": gap.cosine_similarity,
            "angle_degrees": gap.angle_degrees,
            "n_dims_outside_rope": gap.n_dims_outside_rope,
            "n_dims_hdi95_excludes_stated": gap.n_dims_hdi95_excludes_stated,
            "significantly_misaligned": gap.significantly_misaligned,
        })
    df = pd.DataFrame(rows).set_index("variant")

    cos = df["cosine_similarity"].values
    cos_valid = cos[~np.isnan(cos)]
    if len(cos_valid) == 0:
        cos_min = cos_max = cos_range = float("nan")
        direction_stable = False
    else:
        cos_min = float(cos_valid.min())
        cos_max = float(cos_valid.max())
        cos_range = cos_max - cos_min
        signs = np.sign(cos_valid)
        direction_stable = bool(len(np.unique(signs)) <= 1)

    flag_vals = df["significantly_misaligned"].astype(bool).values
    misalignment_stable = bool(len(np.unique(flag_vals)) <= 1)

    return AuditSensitivityReport(
        base_variant="base",
        per_variant=df,
        direction_stable=direction_stable,
        misalignment_flag_stable=misalignment_stable,
        cosine_min=cos_min,
        cosine_max=cos_max,
        cosine_range=cos_range,
        n_variants=len(df),
    )
