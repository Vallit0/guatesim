"""Tests offline del módulo audit_sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from guatemala_sim.irl.audit_sensitivity import (
    AuditSensitivityReport,
    audit_sensitivity,
    generate_w_stated_variants,
)
from guatemala_sim.irl.features import OUTCOME_FEATURE_NAMES
from tests.test_irl_posterior_analysis import make_synthetic_posterior


# --- generate_w_stated_variants --------------------------------------------


def test_variants_incluye_base_y_uniform():
    base = {"anti_pobreza": 1.0, "pro_crecimiento": 0.5}
    out = generate_w_stated_variants(base)
    assert "base" in out
    assert "uniform" in out
    assert out["base"] == base
    # uniform: todas las features con peso 1.0
    for name in OUTCOME_FEATURE_NAMES:
        assert out["uniform"][name] == 1.0


def test_variants_incluye_perturbaciones_per_feature():
    base = {"anti_pobreza": 1.0}
    out = generate_w_stated_variants(base)
    # Para cada feature canónica debe haber plus_, minus_, emphasize_, deemphasize_
    for name in OUTCOME_FEATURE_NAMES:
        for prefix in ("plus_", "minus_", "emphasize_", "deemphasize_"):
            assert f"{prefix}{name}" in out, f"falta variante {prefix}{name}"


def test_plus_50_pct_realmente_aumenta():
    base = {"anti_pobreza": 1.0, "pro_crecimiento": 0.5}
    out = generate_w_stated_variants(base, perturbation=0.5)
    # plus_anti_pobreza: anti_pobreza × 1.5 = 1.5
    assert out["plus_anti_pobreza"]["anti_pobreza"] == 1.5
    # otras features inalteradas
    assert out["plus_anti_pobreza"]["pro_crecimiento"] == 0.5


def test_emphasize_amplifica_la_feature_objetivo():
    base = {"anti_pobreza": 1.0, "pro_crecimiento": 1.0}
    out = generate_w_stated_variants(base)
    emph = out["emphasize_anti_pobreza"]
    assert emph["anti_pobreza"] >= 3.0
    # Otras features se reducen a la mitad
    assert emph["pro_crecimiento"] == 0.5


def test_variants_con_base_zero_aun_da_perturbacion_efectiva():
    """Caso edge: si la base es 0 en una dim, plus_ no puede ser 0
    (sino no habría variante real)."""
    base = {}  # todo en 0
    out = generate_w_stated_variants(base, perturbation=0.5)
    plus = out["plus_anti_pobreza"]
    assert plus["anti_pobreza"] != 0.0


def test_variants_rechaza_features_invalidas():
    with pytest.raises(ValueError, match="inválidas"):
        generate_w_stated_variants({"feature_que_no_existe": 1.0})


def test_variants_perturbation_invalida():
    with pytest.raises(ValueError, match="perturbation"):
        generate_w_stated_variants({}, perturbation=2.0)


# --- audit_sensitivity -----------------------------------------------------


def test_audit_sensitivity_estructura():
    """Smoke test: corre sensitivity sobre un posterior alineado con
    el intent base. Debería ser misaligned NO en la variante base."""
    w_true = np.zeros(6)
    w_true[0] = 2.0  # anti_pobreza alto
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.05,
    )
    base_intent = {"anti_pobreza": 1.0}  # alineado con w_true en dirección
    report = audit_sensitivity(posterior, base_intent, rope_width=0.4)
    assert isinstance(report, AuditSensitivityReport)
    # Debe haber al menos: base + uniform + 4×6 per-feature = 26
    assert report.n_variants >= 20
    assert report.cosine_max >= report.cosine_min
    # Cosine en variante base debe ser > 0 (alineado)
    assert report.per_variant.loc["base", "cosine_similarity"] > 0


def test_audit_sensitivity_direction_stable_caso_extremo():
    """Si w_recovered está fuertemente concentrado en una dim,
    casi todas las variantes deberían dar cosine > 0 y direction
    debería ser estable."""
    w_true = np.zeros(6)
    w_true[0] = 5.0
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES, sample_sigma=0.02,
    )
    base_intent = {"anti_pobreza": 1.0}
    report = audit_sensitivity(posterior, base_intent, rope_width=0.4)
    # Mayoría de variantes deben tener cosine positivo
    pos_frac = (report.per_variant["cosine_similarity"] > 0).mean()
    assert pos_frac >= 0.7


def test_audit_sensitivity_extra_variants():
    w_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    posterior = make_synthetic_posterior(
        w_true, feature_names=OUTCOME_FEATURE_NAMES,
    )
    extra = {
        "minfin_anchored": {
            "anti_pobreza": 0.8,
            "anti_deuda": 0.2,
            "pro_confianza": 0.5,
        },
    }
    report = audit_sensitivity(
        posterior, {"anti_pobreza": 1.0},
        extra_variants=extra,
    )
    assert "minfin_anchored" in report.per_variant.index


def test_audit_sensitivity_extra_variants_rechaza_features_malas():
    posterior = make_synthetic_posterior(
        np.zeros(6), feature_names=OUTCOME_FEATURE_NAMES,
    )
    bad = {"weird": {"feature_que_no_existe": 1.0}}
    with pytest.raises(ValueError, match="inválidas"):
        audit_sensitivity(posterior, {"anti_pobreza": 1.0}, extra_variants=bad)


def test_audit_sensitivity_summary_text():
    posterior = make_synthetic_posterior(
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        feature_names=OUTCOME_FEATURE_NAMES,
    )
    report = audit_sensitivity(posterior, {"anti_pobreza": 1.0})
    txt = report.summary_text("Claude")
    assert "Claude" in txt
    assert "ROBUSTA" in txt or "VOLÁTIL" in txt
