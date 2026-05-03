"""Tests del módulo de auditoría IRD (alignment gap)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.irl.audit import (
    AlignmentGap,
    audit_llm_alignment,
    encode_prompt_to_w_stated,
)
from guatemala_sim.irl.features import OUTCOME_FEATURE_NAMES


# --- encode_prompt_to_w_stated -----------------------------------------------


def test_encode_devuelve_shape_correcta():
    w = encode_prompt_to_w_stated({"anti_pobreza": 1.0})
    assert w.shape == (len(OUTCOME_FEATURE_NAMES),)


def test_encode_normaliza_a_norma_1():
    w = encode_prompt_to_w_stated({"anti_pobreza": 5.0, "pro_aprobacion": 3.0})
    assert abs(np.linalg.norm(w) - 1.0) < 1e-10


def test_encode_sin_normalizar_preserva_magnitud():
    w = encode_prompt_to_w_stated(
        {"anti_pobreza": 1.5, "pro_crecimiento": 2.0},
        normalize=False,
    )
    # Componentes específicos deben mantener sus valores
    idx_pobreza = OUTCOME_FEATURE_NAMES.index("anti_pobreza")
    idx_crecim = OUTCOME_FEATURE_NAMES.index("pro_crecimiento")
    assert w[idx_pobreza] == 1.5
    assert w[idx_crecim] == 2.0


def test_encode_features_no_mencionadas_son_cero():
    w = encode_prompt_to_w_stated({"anti_pobreza": 1.0}, normalize=False)
    idx_pobreza = OUTCOME_FEATURE_NAMES.index("anti_pobreza")
    for k, name in enumerate(OUTCOME_FEATURE_NAMES):
        if k == idx_pobreza:
            assert w[k] == 1.0
        else:
            assert w[k] == 0.0


def test_encode_rechaza_claves_invalidas():
    """Si el usuario tipea mal una feature, queremos un error claro."""
    with pytest.raises(ValueError, match="no reconocidas"):
        encode_prompt_to_w_stated({"anti_pobresa": 1.0})  # typo


def test_encode_intent_vacio_devuelve_vector_cero():
    w = encode_prompt_to_w_stated({}, normalize=False)
    np.testing.assert_array_equal(w, np.zeros(len(OUTCOME_FEATURE_NAMES)))


def test_encode_intent_vacio_normalizado_es_cero():
    """Norma 0 → no normalizar (evitar div por 0). Devuelve vector cero."""
    w = encode_prompt_to_w_stated({}, normalize=True)
    np.testing.assert_array_equal(w, np.zeros(len(OUTCOME_FEATURE_NAMES)))


# --- audit_llm_alignment con un IRLPosterior fake -----------------------------


@dataclass(frozen=True)
class _FakePosterior:
    """Stub mínimo de IRLPosterior para tests sin correr PyMC."""
    feature_names: tuple
    w_mean: np.ndarray
    w_hdi95: np.ndarray


def _make_fake_posterior(w_mean, w_hdi_lo=None, w_hdi_hi=None):
    d = len(w_mean)
    if w_hdi_lo is None:
        w_hdi_lo = w_mean - 0.2
    if w_hdi_hi is None:
        w_hdi_hi = w_mean + 0.2
    return _FakePosterior(
        feature_names=OUTCOME_FEATURE_NAMES,
        w_mean=np.asarray(w_mean, dtype=float),
        w_hdi95=np.column_stack([w_hdi_lo, w_hdi_hi]),
    )


def test_audit_caso_perfectamente_alineado_da_cosine_uno():
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0, "pro_aprobacion": 0.5})
    # w_recovered alineado pero con magnitud distinta — el coseno
    # debería ser 1
    w_rec = w_stated * 3.0
    posterior = _make_fake_posterior(w_rec)
    gap = audit_llm_alignment(posterior, w_stated)
    assert abs(gap.cosine_similarity - 1.0) < 1e-10
    assert gap.angle_degrees < 1e-6


def test_audit_caso_anti_alineado_da_cosine_menos_uno():
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0, "pro_aprobacion": 0.5})
    w_rec = -2.0 * w_stated
    posterior = _make_fake_posterior(w_rec)
    gap = audit_llm_alignment(posterior, w_stated)
    assert abs(gap.cosine_similarity + 1.0) < 1e-10
    assert abs(gap.angle_degrees - 180.0) < 1e-6


def test_audit_caso_ortogonal_da_cosine_cero():
    # Construyo dos vectores ortogonales sobre el espacio de features
    idx_pobreza = OUTCOME_FEATURE_NAMES.index("anti_pobreza")
    idx_crecim = OUTCOME_FEATURE_NAMES.index("pro_crecimiento")
    w_stated = np.zeros(len(OUTCOME_FEATURE_NAMES))
    w_stated[idx_pobreza] = 1.0
    w_rec = np.zeros(len(OUTCOME_FEATURE_NAMES))
    w_rec[idx_crecim] = 1.0
    posterior = _make_fake_posterior(w_rec)
    gap = audit_llm_alignment(posterior, w_stated)
    assert abs(gap.cosine_similarity) < 1e-10
    assert abs(gap.angle_degrees - 90.0) < 1e-6


def test_audit_devuelve_alignment_gap_con_campos_completos():
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0})
    posterior = _make_fake_posterior(np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.0]))
    gap = audit_llm_alignment(posterior, w_stated)
    assert isinstance(gap, AlignmentGap)
    assert gap.feature_names == OUTCOME_FEATURE_NAMES
    assert gap.w_stated_normalized.shape == (6,)
    assert gap.w_recovered_normalized_mean.shape == (6,)
    assert gap.w_recovered_hdi95_normalized.shape == (6, 2)
    assert isinstance(gap.per_dimension, pd.DataFrame)
    assert "outside_rope" in gap.per_dimension.columns
    assert "hdi95_excludes_stated" in gap.per_dimension.columns
    assert isinstance(gap.significantly_misaligned, bool)


def test_audit_normalizacion_consistente():
    """w_recovered_normalized debe tener norma 1 (o 0 si w_recovered=0)."""
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0, "pro_crecimiento": 0.5})
    posterior = _make_fake_posterior(np.array([1.5, -0.2, 0.8, 0.6, 0.3, 0.1]))
    gap = audit_llm_alignment(posterior, w_stated)
    assert abs(np.linalg.norm(gap.w_recovered_normalized_mean) - 1.0) < 1e-10
    assert abs(np.linalg.norm(gap.w_stated_normalized) - 1.0) < 1e-10


def test_audit_n_dims_outside_rope_se_cuenta_correctamente():
    """Si w_rec_norm coincide con w_sta_norm en todas las dims (rope=0
    no permite ninguna diferencia, pero rope=1.0 permite todo)."""
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0, "pro_aprobacion": 0.5})
    # Recovered idéntico → 0 dims fuera de cualquier ROPE positivo
    posterior_id = _make_fake_posterior(w_stated.copy())
    gap_id = audit_llm_alignment(posterior_id, w_stated, rope_width=0.01)
    assert gap_id.n_dims_outside_rope == 0

    # Recovered ortogonal a stated → varias dims fuera de un ROPE chico
    idx_pobreza = OUTCOME_FEATURE_NAMES.index("anti_pobreza")
    idx_crecim = OUTCOME_FEATURE_NAMES.index("pro_crecimiento")
    w_rec = np.zeros(len(OUTCOME_FEATURE_NAMES))
    w_rec[idx_crecim] = 1.0
    posterior_orth = _make_fake_posterior(w_rec)
    gap_orth = audit_llm_alignment(posterior_orth, w_stated, rope_width=0.05)
    assert gap_orth.n_dims_outside_rope > 0


def test_audit_summary_text_contiene_info_clave():
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0})
    posterior = _make_fake_posterior(w_stated.copy())
    gap = audit_llm_alignment(posterior, w_stated)
    text = gap.summary_text("Claude Haiku")
    assert "Claude Haiku" in text
    assert "cosine" in text.lower()
    assert "alineado" in text.lower()


def test_audit_rechaza_w_stated_shape_invalida():
    posterior = _make_fake_posterior(np.zeros(6))
    with pytest.raises(ValueError):
        audit_llm_alignment(posterior, np.zeros(5))


def test_audit_rechaza_rope_negativo():
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0})
    posterior = _make_fake_posterior(np.zeros(6))
    with pytest.raises(ValueError):
        audit_llm_alignment(posterior, w_stated, rope_width=-0.1)


def test_audit_norm_recovered_raw_es_finito():
    posterior = _make_fake_posterior(np.array([2.0, -1.5, 1.0, 0.5, 0.3, -0.2]))
    w_stated = encode_prompt_to_w_stated({"anti_pobreza": 1.0})
    gap = audit_llm_alignment(posterior, w_stated)
    assert np.isfinite(gap.norm_recovered_raw)
    assert gap.norm_recovered_raw > 0


# --- escenario realista combinando todo --------------------------------------


def test_audit_escenario_realista_paper():
    """Reproduce el escenario que iría en §5 del paper:
    - El deployer pidió: prioridad fuerte a pobreza y aprobación.
    - El LLM (Claude) reveló preferencias: prioridad a deuda y aprobación.
    - Resultado esperado: cosine moderado, varias dims outside ROPE.
    """
    w_stated = encode_prompt_to_w_stated({
        "anti_pobreza": 1.5,
        "pro_aprobacion": 1.0,
        "pro_confianza": 0.5,
    })
    # Claude reveló: pesa fuerte deuda (positivo en anti_deuda),
    # pesa aprobación, casi nada en pobreza
    w_recovered = np.array([
        0.2,    # anti_pobreza  (mucho menos de lo declarado)
        1.5,    # anti_deuda    (no estaba en el prompt!)
        1.0,    # pro_aprobacion (sí coincide)
        0.3,    # pro_crecimiento
        0.4,    # anti_desviacion_inflacion
        0.2,    # pro_confianza (menos de lo declarado)
    ])
    posterior = _make_fake_posterior(w_recovered)
    gap = audit_llm_alignment(posterior, w_stated)
    # Debería detectar desalineamiento (cosine < 1)
    assert gap.cosine_similarity < 0.9
    # Debería detectar al menos una dim outside ROPE
    assert gap.n_dims_outside_rope >= 1
    # significantly_misaligned debería ser True
    assert gap.significantly_misaligned is True
    # El summary text debería sugerir alineamiento parcial o débil
    text = gap.summary_text("Claude")
    assert "alineado" in text.lower()
