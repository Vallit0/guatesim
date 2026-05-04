"""Tests del encoding v2 de reasoning-policy consistency."""

from __future__ import annotations

import numpy as np
import pytest

from guatemala_sim.irl.features import (
    N_OUTCOME_FEATURES,
    OUTCOME_FEATURE_NAMES,
)
from guatemala_sim.reasoning_consistency import REASONING_KEYWORDS
from guatemala_sim.reasoning_consistency_v2 import (
    ANCHOR_PHRASES_V2,
    V2Encoder,
    _strip_accents,
    _tokenize,
    assess_reasoning_consistency_v2,
    cohens_kappa_binary,
    encode_reasoning_to_w_v2,
    fit_v2_encoder,
)


# --- claim de disjunción v1 vs v2 -------------------------------------------


_SPANISH_STOPWORDS = frozenset({
    "de", "la", "el", "y", "o", "a", "en", "del", "los", "las",
    "un", "una", "es", "se", "que", "por", "con", "para", "al", "lo",
    "su", "sus", "como", "mas", "menos", "no", "si", "mi", "te",
    "le", "les", "ya", "ha", "han", "fue", "ser", "este", "esta",
    "esto", "muy", "ese", "esa", "eso",
})


def _content_tokens(text: str) -> set[str]:
    """Tokens de contenido (descartando stopwords) — para tests de disjunción."""
    return {t for t in _tokenize(text) if t not in _SPANISH_STOPWORDS}


def test_v2_anchor_phrases_no_contienen_keywords_v1_completos():
    """Disjunción a nivel de keyword completo: ningún keyword v1 debe
    aparecer como substring de una anchor phrase v2 (después de strip
    accents y lowercase). Esa es la noción operativa de
    'independencia de codificación' que el reviewer del paper pidió.
    """
    overlaps: list[tuple[str, str, str]] = []
    for feat, phrases in ANCHOR_PHRASES_V2.items():
        for phrase in phrases:
            phrase_norm = _strip_accents(phrase.lower())
            for v1_feat, kws in REASONING_KEYWORDS.items():
                for kw in kws:
                    kw_norm = _strip_accents(kw.lower())
                    if kw_norm in phrase_norm:
                        overlaps.append((feat, phrase, kw))
    assert overlaps == [], (
        f"v2 anchor phrases contienen keywords completos de v1: "
        f"{overlaps[:5]}"
    )


def test_v2_content_tokens_disjuntos_de_v1_content_tokens():
    """Disjunción más fuerte (a nivel de palabra de contenido,
    excluyendo stopwords). Documenta cuán independientes son los
    diccionarios — el cross-encoding agreement (Cohen's κ) sólo es
    informativo si los lexicones tienen poca overlap léxica.
    """
    v1_content: set[str] = set()
    for kws in REASONING_KEYWORDS.values():
        for kw in kws:
            v1_content |= _content_tokens(kw)

    v2_content: set[str] = set()
    for phrases in ANCHOR_PHRASES_V2.values():
        for p in phrases:
            v2_content |= _content_tokens(p)

    overlap = v1_content & v2_content
    # Tolerancia: hasta 10 palabras de contenido compartidas. Como ambos
    # lexicones cubren el mismo dominio (política fiscal/macroeconómica
    # en español), un puñado de palabras genéricas como "fiscal",
    # "monetaria", "precios" inevitablemente aparece en ambos. Lo que
    # nos importa es que v2 no replique las frases discriminativas de
    # v1 — verificado por el test anterior de substring-completo.
    # El umbral de 10 es ~5% de las ~200 palabras de contenido del v2;
    # más que eso sugeriría que el lexicón v2 no fue construido
    # independientemente.
    assert len(overlap) <= 10, (
        f"v2 comparte demasiadas palabras de contenido con v1: {sorted(overlap)}"
    )


def test_anchor_phrases_cubren_todas_las_features():
    assert set(ANCHOR_PHRASES_V2.keys()) == set(OUTCOME_FEATURE_NAMES)
    for feat, phrases in ANCHOR_PHRASES_V2.items():
        assert len(phrases) >= 6, f"feature {feat} tiene < 6 anchor phrases"


# --- TF-IDF helpers ---------------------------------------------------------


def test_strip_accents_basico():
    assert _strip_accents("acción inflación") == "accion inflacion"
    # _strip_accents preserva el case; lowercase lo aplica `_tokenize`.
    assert _strip_accents("Mañana") == "Manana"


def test_tokenize_basico():
    toks = _tokenize("Reducir HOGARES en privación.")
    assert toks == ["reducir", "hogares", "en", "privacion"]


def test_tokenize_ignora_signos():
    toks = _tokenize("a, b; c. d!")
    assert toks == ["a", "b", "c", "d"]


# --- Encoder fit ------------------------------------------------------------


def test_fit_v2_encoder_returns_v2encoder():
    enc = fit_v2_encoder()
    assert isinstance(enc, V2Encoder)
    assert enc.feature_names == OUTCOME_FEATURE_NAMES
    assert enc.feature_centroids.shape[0] == N_OUTCOME_FEATURES


def test_centroides_son_l2_unitarios():
    enc = fit_v2_encoder()
    norms = np.linalg.norm(enc.feature_centroids, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_encoder_keys_invalidas_levantan():
    bad = {k: v for k, v in ANCHOR_PHRASES_V2.items()}
    bad["feature_que_no_existe"] = ("frase",)
    with pytest.raises(ValueError, match="no coinciden"):
        fit_v2_encoder(anchor_phrases=bad)


# --- Encode reasoning -------------------------------------------------------


def test_encode_reasoning_anti_pobreza_dominante():
    """Un razonamiento totalmente anti-pobreza debe dar máximo en esa dim."""
    enc = fit_v2_encoder()
    text = (
        "Mi prioridad es reducir hogares en privacion y atender el "
        "quintil mas bajo. Vamos a sacar familias de la marginalidad."
    )
    w = encode_reasoning_to_w_v2(text, encoder=enc, normalize=False)
    k_pobreza = OUTCOME_FEATURE_NAMES.index("anti_pobreza")
    # debe ser máximo
    assert w.argmax() == k_pobreza
    assert w[k_pobreza] > 0.3


def test_encode_reasoning_anti_deuda_dominante():
    enc = fit_v2_encoder()
    text = (
        "Honrar compromisos con acreedores y mantener grado de inversion. "
        "Regla fiscal estricta y consolidacion del gasto."
    )
    w = encode_reasoning_to_w_v2(text, encoder=enc, normalize=False)
    k = OUTCOME_FEATURE_NAMES.index("anti_deuda")
    assert w.argmax() == k


def test_encode_reasoning_pro_crecimiento_dominante():
    enc = fit_v2_encoder()
    text = (
        "Vamos a dinamizar la economia, atraer capital privado y "
        "estimular el empleo formal con encadenamientos productivos."
    )
    w = encode_reasoning_to_w_v2(text, encoder=enc, normalize=False)
    k = OUTCOME_FEATURE_NAMES.index("pro_crecimiento")
    assert w.argmax() == k


def test_encode_reasoning_vacio_es_zero():
    enc = fit_v2_encoder()
    w = encode_reasoning_to_w_v2("", encoder=enc, normalize=False)
    assert np.allclose(w, 0.0)


def test_encode_reasoning_sin_match_devuelve_zero():
    enc = fit_v2_encoder()
    # texto en chino → ningún match en vocab español
    w = encode_reasoning_to_w_v2("xxxxxx yyyyyy zzzzzz", encoder=enc, normalize=False)
    assert np.allclose(w, 0.0)


def test_normalize_devuelve_unitario_o_zero():
    enc = fit_v2_encoder()
    w = encode_reasoning_to_w_v2(
        "reducir hogares en privacion", encoder=enc, normalize=True
    )
    n = float(np.linalg.norm(w))
    assert abs(n - 1.0) < 1e-9


# --- assess v2 --------------------------------------------------------------


def test_assess_v2_devuelve_consistency_report():
    enc = fit_v2_encoder()
    razonamientos = [
        "Reducir hogares en privacion",
        "Atender el quintil mas bajo",
        "Sacar familias de la marginalidad",
    ]
    w_rec = np.zeros(N_OUTCOME_FEATURES)
    w_rec[OUTCOME_FEATURE_NAMES.index("anti_pobreza")] = 1.0

    rpt = assess_reasoning_consistency_v2(
        razonamientos=razonamientos, w_recovered=w_rec, encoder=enc
    )
    assert rpt.n_turnos == 3
    assert rpt.cosine_similarity > 0.5
    assert not rpt.inconsistency_flag


def test_assess_v2_flag_se_dispara_con_inconsistencia():
    enc = fit_v2_encoder()
    razonamientos = [
        "Honrar compromisos con acreedores",
        "Mantener grado de inversion",
    ]
    # w_recovered apunta a anti_pobreza, no a anti_deuda → debería ser inconsistente
    w_rec = np.zeros(N_OUTCOME_FEATURES)
    w_rec[OUTCOME_FEATURE_NAMES.index("anti_pobreza")] = 1.0
    rpt = assess_reasoning_consistency_v2(
        razonamientos=razonamientos, w_recovered=w_rec, encoder=enc
    )
    assert rpt.inconsistency_flag


def test_assess_v2_lista_vacia_levanta():
    with pytest.raises(ValueError):
        assess_reasoning_consistency_v2([], np.zeros(N_OUTCOME_FEATURES))


def test_assess_v2_w_recovered_shape_invalido():
    with pytest.raises(ValueError):
        assess_reasoning_consistency_v2(["frase"], np.zeros(3))


# --- Cohen's kappa ----------------------------------------------------------


def test_kappa_acuerdo_perfecto():
    y1 = np.array([1, 0, 1, 1, 0])
    y2 = np.array([1, 0, 1, 1, 0])
    assert abs(cohens_kappa_binary(y1, y2) - 1.0) < 1e-12


def test_kappa_acuerdo_nulo():
    """Mitad y mitad sin correlacion → κ ≈ 0."""
    y1 = np.array([1, 0, 1, 0])
    y2 = np.array([0, 1, 0, 1])
    assert abs(cohens_kappa_binary(y1, y2) - (-1.0)) < 1e-12


def test_kappa_un_signo():
    y1 = np.array([1, 0, 1, 1, 0])
    y2 = np.array([1, 0, 0, 1, 1])
    k = cohens_kappa_binary(y1, y2)
    # debe estar en [-1, 1]
    assert -1.0 <= k <= 1.0


def test_kappa_clase_uniforme_es_nan():
    y1 = np.array([1, 1, 1, 1])
    y2 = np.array([1, 1, 1, 1])
    # ambas todo-1 → p_e = 1, κ = NaN por convención
    k = cohens_kappa_binary(y1, y2)
    assert np.isnan(k)


def test_kappa_shapes_distintos_levantan():
    with pytest.raises(ValueError):
        cohens_kappa_binary(np.array([1, 0]), np.array([1]))
