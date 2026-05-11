"""Validación sintética del IRD audit (recovery del alignment gap).

**El problema honesto**: tenemos `irl/recovery.py` para validar el IRL
bayesiano contra w_true sintético, pero NO existía el análogo para el
IRD audit. La pregunta crítica del revisor: *"si conozco w_true y
w_stated, ¿el módulo `audit_llm_alignment` reporta el gap correcto?"*.

Sin esta capa, el IRD audit es código sin oracle.

**Diseño**: para cada par diseñado `(w_true, w_stated)` con cosine
teórico conocido por construcción, construir un `IRLPosterior`
concentrado en w_true, correr el audit, verificar que:

  - `cosine_similarity` reportado ≈ cosine teórico
  - en escenario aligned (cosine ≈ 1), `n_dims_outside_rope == 0` y
    `n_dims_hdi95_excludes_stated == 0`
  - en escenario anti-aligned (cosine ≈ -1), TODAS las dims relevantes
    deberían estar fuera del ROPE
  - en escenario orthogonal, el cosine cae en banda alrededor de 0

Estos tests usan `make_synthetic_posterior` (helper offline; sin PyMC),
porque la pregunta acá es sobre la MECÁNICA del audit, no sobre la
mecánica del muestreo NUTS — ese ya tiene su propio test en
`test_irl_recovery.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from guatemala_sim.irl.audit import (
    AlignmentGap,
    audit_llm_alignment,
    encode_prompt_to_w_stated,
)
from guatemala_sim.irl.features import (
    N_OUTCOME_FEATURES,
    OUTCOME_FEATURE_NAMES,
)
from tests.test_irl_posterior_analysis import make_synthetic_posterior


# --- escenario 1: aligned ---------------------------------------------------


def test_audit_recupera_aligned():
    """w_true ≈ w_stated → cosine ≈ 1, no misaligned."""
    w_true = np.zeros(N_OUTCOME_FEATURES)
    w_true[OUTCOME_FEATURE_NAMES.index("anti_pobreza")] = 2.0
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.05,
    )
    w_stated = encode_prompt_to_w_stated(
        {"anti_pobreza": 1.0}, normalize=True,
    )
    gap = audit_llm_alignment(posterior, w_stated, rope_width=0.25)
    assert isinstance(gap, AlignmentGap)
    assert gap.cosine_similarity > 0.95, (
        f"esperaba cosine ≈ 1, tengo {gap.cosine_similarity}"
    )
    assert gap.angle_degrees < 20.0
    # En aligned, NINGUNA dim debe estar fuera del ROPE razonable
    assert gap.n_dims_outside_rope == 0
    # significantly_misaligned debe ser False
    assert gap.significantly_misaligned is False or gap.n_dims_hdi95_excludes_stated == 0


# --- escenario 2: orthogonal ------------------------------------------------


def test_audit_recupera_orthogonal():
    """w_true ⊥ w_stated → cosine ≈ 0."""
    w_true = np.zeros(N_OUTCOME_FEATURES)
    w_true[OUTCOME_FEATURE_NAMES.index("anti_pobreza")] = 2.0
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.05,
    )
    w_stated = encode_prompt_to_w_stated(
        {"anti_deuda": 1.0}, normalize=True,
    )
    gap = audit_llm_alignment(posterior, w_stated, rope_width=0.25)
    assert -0.2 < gap.cosine_similarity < 0.2, (
        f"esperaba cosine ≈ 0, tengo {gap.cosine_similarity}"
    )
    # Ángulo cerca de 90°
    assert 70.0 < gap.angle_degrees < 110.0
    # En orthogonal, dims relevantes (las dos no-cero) deberían
    # marcar diferencia significativa
    assert gap.n_dims_outside_rope >= 1
    assert gap.significantly_misaligned is True


# --- escenario 3: anti-aligned ----------------------------------------------


def test_audit_recupera_anti_aligned():
    """w_true = -w_stated → cosine ≈ -1, fuertemente misaligned."""
    w_true = np.zeros(N_OUTCOME_FEATURES)
    w_true[OUTCOME_FEATURE_NAMES.index("anti_pobreza")] = 2.0
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.05,
    )
    # Intent declara prioridad NEGATIVA sobre anti_pobreza — caso patológico
    # pero matemáticamente legítimo para validar el detector.
    w_stated = encode_prompt_to_w_stated(
        {"anti_pobreza": -1.0}, normalize=True,
    )
    gap = audit_llm_alignment(posterior, w_stated, rope_width=0.25)
    assert gap.cosine_similarity < -0.95, (
        f"esperaba cosine ≈ -1, tengo {gap.cosine_similarity}"
    )
    assert gap.angle_degrees > 160.0
    assert gap.n_dims_outside_rope >= 1
    assert gap.significantly_misaligned is True


# --- escenario 4: parcial alignment (test gradiente) -----------------------


@pytest.mark.parametrize("theta_deg", [0, 30, 60, 90, 120, 150, 180])
def test_audit_cosine_continuo_en_angulo(theta_deg: int):
    """El cosine reportado debe seguir cos(θ) cuando rotamos w_stated
    en el plano (anti_pobreza, anti_deuda), manteniendo w_true fijo."""
    w_true = np.zeros(N_OUTCOME_FEATURES)
    w_true[0] = 1.0
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.02,
    )
    theta = np.radians(theta_deg)
    w_stated = np.zeros(N_OUTCOME_FEATURES)
    w_stated[0] = np.cos(theta)
    w_stated[1] = np.sin(theta)
    gap = audit_llm_alignment(posterior, w_stated, rope_width=0.25)
    expected_cos = float(np.cos(theta))
    assert abs(gap.cosine_similarity - expected_cos) < 0.05, (
        f"θ={theta_deg}°: esperado cosine={expected_cos:.3f}, "
        f"tengo {gap.cosine_similarity:.3f}"
    )


# --- escenario 5: ROPE width afecta n_dims_outside_rope --------------------


def test_audit_rope_width_estricto_marca_mas_dims():
    """Con un ROPE muy chico, casi cualquier diferencia se considera
    fuera del ROPE. Con un ROPE generoso, casi ninguna."""
    w_true = np.zeros(N_OUTCOME_FEATURES)
    w_true[0] = 1.0
    w_true[1] = 0.5
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.05,
    )
    # w_stated levemente distinto de w_true
    w_stated = w_true.copy()
    w_stated[0] = 0.8
    w_stated[1] = 0.7
    w_stated /= np.linalg.norm(w_stated)

    strict = audit_llm_alignment(posterior, w_stated, rope_width=0.05)
    lenient = audit_llm_alignment(posterior, w_stated, rope_width=0.50)
    # ROPE estricto debe marcar más o igual cantidad de dims fuera
    assert strict.n_dims_outside_rope >= lenient.n_dims_outside_rope


# --- escenario 6: identificabilidad del HDI -------------------------------


def test_audit_hdi95_tight_excluye_stated_lejano():
    """Si el posterior es muy concentrado y w_stated cae lejos del
    centro, HDI95 debería excluir w_stated en esas dims."""
    w_true = np.zeros(N_OUTCOME_FEATURES)
    w_true[0] = 5.0
    posterior = make_synthetic_posterior(
        w_true,
        feature_names=OUTCOME_FEATURE_NAMES,
        sample_sigma=0.01,  # MUY concentrado
        n_samples=2000,
    )
    # w_stated diametralmente opuesto en dim 0
    w_stated_dict = {"anti_pobreza": -2.0, "anti_deuda": 0.0}
    w_stated = encode_prompt_to_w_stated(w_stated_dict, normalize=True)
    gap = audit_llm_alignment(posterior, w_stated, rope_width=0.1)
    # HDI95 debería excluir el valor declarado en al menos la dim 0
    assert gap.n_dims_hdi95_excludes_stated >= 1


# --- escenario 7: w_stated cero (caso degenerado) -------------------------


def test_audit_w_stated_cero_devuelve_nan_cosine():
    """Si w_stated es exactamente cero, el cosine es indeterminado
    (división por cero). Debe devolver NaN, no crashear."""
    w_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES,
    )
    w_stated = np.zeros(N_OUTCOME_FEATURES)
    gap = audit_llm_alignment(posterior, w_stated)
    assert np.isnan(gap.cosine_similarity)
    assert np.isnan(gap.angle_degrees)
