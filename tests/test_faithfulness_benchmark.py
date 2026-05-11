"""Tests del benchmark sintético de faithfulness.

Verifican: (a) que los templates v3 tienen overlap controlado con v1/v2
(condición para que el benchmark sea fair), (b) que la generación de
samples es reproducible y bien-formada, (c) que los encoders v1, v2 y
v3 (con embedder mock) corren sin errores y producen métricas en
rango razonable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.faithfulness_benchmark import (
    PARAPHRASE_TEMPLATES_V3,
    SyntheticSample,
    compare_encoders,
    evaluate_encoder,
    generate_synthetic_samples,
    measure_lexical_overlap_with_v1_v2,
)
from guatemala_sim.irl.features import N_OUTCOME_FEATURES, OUTCOME_FEATURE_NAMES
from guatemala_sim.reasoning_consistency import encode_reasoning_to_w
from guatemala_sim.reasoning_consistency_v2 import (
    encode_reasoning_to_w_v2,
    fit_v2_encoder,
)
from guatemala_sim.reasoning_consistency_v3 import (
    fit_v3_encoder,
    encode_reasoning_to_w_v3,
)
from tests.test_reasoning_consistency_v3 import HashEmbedder


# --- templates: condición de fairness ---------------------------------------


def test_templates_v3_cubren_todas_las_features():
    assert set(PARAPHRASE_TEMPLATES_V3.keys()) == set(OUTCOME_FEATURE_NAMES)
    for name, phs in PARAPHRASE_TEMPLATES_V3.items():
        assert len(phs) >= 5, (
            f"feature {name} tiene solo {len(phs)} paraphrases; "
            f"recomendado ≥5 para diversidad léxica"
        )


def test_templates_v3_tienen_vocabulario_novel_suficiente():
    """Honestidad sobre fairness: en prosa de política en español el
    vocabulario sustantivo (pobreza, fiscal, inflación, instituciones)
    se REPITE entre encoders por la naturaleza del dominio. No es
    realista exigir disjoint.

    Lo que SÍ podemos exigir: que cada feature tenga una fracción
    suficiente de tokens de contenido NOVEL (no presentes en
    v1/v2). Eso prueba que los templates aportan paráfrasis nuevas
    y no son keyword stuffing.

    Umbral: ≥30 % del vocabulario de contenido por feature debe ser
    novel. Eso aísla la capacidad de generalización del encoder.
    """
    overlap = measure_lexical_overlap_with_v1_v2()
    for name, frac_overlap in overlap.items():
        novel_frac = 1.0 - frac_overlap
        assert novel_frac >= 0.30, (
            f"feature {name}: solo {novel_frac:.1%} del vocabulario de "
            f"contenido en los templates v3 es novel respecto de v1/v2. "
            f"Necesario ≥30 % para que el benchmark mida generalización, "
            f"no exact-match keyword detection."
        )


def test_ningun_template_v3_es_verbatim_de_v1_v2():
    """Un template idéntico a un keyword de v1 o anchor phrase de v2
    sería trampa. Debe no haber match exacto (case-insensitive,
    sin acentos)."""
    from guatemala_sim.faithfulness_benchmark import _normalize_for_overlap
    from guatemala_sim.reasoning_consistency import REASONING_KEYWORDS
    from guatemala_sim.reasoning_consistency_v2 import ANCHOR_PHRASES_V2

    v1_norm = {
        _normalize_for_overlap(kw)
        for kws in REASONING_KEYWORDS.values()
        for kw in kws
    }
    v2_norm = {
        _normalize_for_overlap(p)
        for phs in ANCHOR_PHRASES_V2.values()
        for p in phs
    }
    leaks: list[tuple[str, str]] = []
    for name, phs in PARAPHRASE_TEMPLATES_V3.items():
        for p in phs:
            n = _normalize_for_overlap(p)
            if n in v1_norm or n in v2_norm:
                leaks.append((name, p))
    assert not leaks, f"templates verbatim presentes en v1/v2: {leaks}"


# --- generación de samples ---------------------------------------------------


def test_generate_samples_estructura():
    samples = generate_synthetic_samples(
        n_pure_per_feature=3, n_mixed=5, seed=0,
    )
    assert len(samples) == 6 * 3 + 5
    n_pure = sum(1 for s in samples if s.sample_type == "pure")
    n_mixed = sum(1 for s in samples if s.sample_type == "mixed")
    assert n_pure == 18
    assert n_mixed == 5


def test_generate_samples_w_true_normalizado():
    samples = generate_synthetic_samples(
        n_pure_per_feature=2, n_mixed=10, seed=1,
    )
    for s in samples:
        # Cada w_true debe estar L2-normalizado
        nrm = float(np.linalg.norm(s.w_true))
        assert abs(nrm - 1.0) < 1e-9
        # Pure: exactamente una coordenada > 0
        if s.sample_type == "pure":
            n_nonzero = int((s.w_true > 0).sum())
            assert n_nonzero == 1
        # Mixed: exactamente dos coordenadas > 0
        if s.sample_type == "mixed":
            n_nonzero = int((s.w_true > 0).sum())
            assert n_nonzero == 2


def test_generate_samples_reproducible():
    a = generate_synthetic_samples(n_pure_per_feature=3, n_mixed=5, seed=99)
    b = generate_synthetic_samples(n_pure_per_feature=3, n_mixed=5, seed=99)
    assert len(a) == len(b)
    for s_a, s_b in zip(a, b):
        assert s_a.text == s_b.text
        np.testing.assert_array_equal(s_a.w_true, s_b.w_true)


def test_generate_samples_textos_no_vacios():
    samples = generate_synthetic_samples(
        n_pure_per_feature=2, n_mixed=5, seed=2,
    )
    for s in samples:
        assert len(s.text) > 20
        assert s.text.endswith(".")


# --- evaluate_encoder -------------------------------------------------------


def _v1_encode(text: str) -> np.ndarray:
    return encode_reasoning_to_w(text, normalize=False)


def _v2_encode_factory():
    enc = fit_v2_encoder()
    def _f(text: str) -> np.ndarray:
        return encode_reasoning_to_w_v2(text, encoder=enc, normalize=False)
    return _f


def _v3_mock_encode_factory():
    embedder = HashEmbedder(dim=32)
    enc = fit_v3_encoder(embedder=embedder, model_name="mock-hash-32")
    def _f(text: str) -> np.ndarray:
        return encode_reasoning_to_w_v3(text, encoder=enc, normalize=False)
    return _f


def test_evaluate_encoder_v1_no_crashea():
    """v1 (keyword counting) debe correr y producir métricas válidas
    aunque score bajo (templates v3 evitan sus keywords)."""
    samples = generate_synthetic_samples(
        n_pure_per_feature=3, n_mixed=10, seed=11,
    )
    ev = evaluate_encoder(samples, _v1_encode, encoder_name="v1-keywords")
    assert ev.n_samples == 28
    assert ev.n_pure == 18
    assert ev.n_mixed == 10
    # Métricas en rangos sensatos (NaN OK si todos los samples generan
    # encode cero, que no debería pasar acá)
    assert 0.0 <= ev.argmax_accuracy_pure <= 1.0 or np.isnan(ev.argmax_accuracy_pure)
    assert -1.0 <= ev.spearman_mean_mixed <= 1.0 or np.isnan(ev.spearman_mean_mixed)
    assert ev.l1_mean >= 0.0


def test_evaluate_encoder_v2_runs():
    samples = generate_synthetic_samples(
        n_pure_per_feature=3, n_mixed=5, seed=12,
    )
    ev = evaluate_encoder(samples, _v2_encode_factory(), encoder_name="v2-tfidf")
    assert ev.n_samples == 23
    assert ev.l1_mean >= 0.0


def test_evaluate_encoder_v3_mock_runs():
    samples = generate_synthetic_samples(
        n_pure_per_feature=3, n_mixed=5, seed=13,
    )
    ev = evaluate_encoder(samples, _v3_mock_encode_factory(), encoder_name="v3-mock")
    assert ev.n_samples == 23
    assert ev.l1_mean >= 0.0


def test_evaluate_encoder_falla_si_shape_invalida():
    samples = generate_synthetic_samples(
        n_pure_per_feature=2, n_mixed=2, seed=14,
    )
    def bad_encoder(text: str) -> np.ndarray:
        return np.zeros(3)  # shape incorrecto
    with pytest.raises(ValueError, match="shape"):
        evaluate_encoder(samples, bad_encoder, encoder_name="bad")


def test_compare_encoders_devuelve_tabla_ordenada():
    samples = generate_synthetic_samples(
        n_pure_per_feature=3, n_mixed=5, seed=15,
    )
    encoders = {
        "v1-keywords": _v1_encode,
        "v2-tfidf": _v2_encode_factory(),
        "v3-mock": _v3_mock_encode_factory(),
    }
    df = compare_encoders(samples, encoders)
    assert isinstance(df, pd.DataFrame)
    assert set(df["encoder"]) == {"v1-keywords", "v2-tfidf", "v3-mock"}
    assert "argmax_accuracy_pure" in df.columns
    # Orden descendente por argmax_accuracy_pure
    accs = df["argmax_accuracy_pure"].values
    # Si todos NaN o iguales el orden no importa, así que solo
    # validamos cuando hay variación
    valid = [a for a in accs if not np.isnan(a)]
    if len(set(valid)) > 1:
        assert valid == sorted(valid, reverse=True)


def test_per_sample_dataframe_columnas():
    samples = generate_synthetic_samples(
        n_pure_per_feature=2, n_mixed=3, seed=16,
    )
    ev = evaluate_encoder(samples, _v1_encode, encoder_name="v1")
    df = ev.per_sample
    expected_cols = {
        "sample_type", "dominant_features", "argmax_correct",
        "spearman", "l1_distance", "text_chars",
    }
    assert expected_cols.issubset(df.columns)
    # En pure samples, spearman es None / NaN; en mixed, argmax_correct es None
    pure = df[df["sample_type"] == "pure"]
    mixed = df[df["sample_type"] == "mixed"]
    assert pure["spearman"].isna().all()
    assert mixed["argmax_correct"].isna().all()
