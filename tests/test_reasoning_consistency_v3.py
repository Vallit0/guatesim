"""Tests offline del encoder v3 (sentence embeddings, dependency-injected).

No tocan red ni descargan modelos: todos usan un embedder mock
determinístico (`HashEmbedder`) que respeta la firma del default.
Si `sentence-transformers` está instalado en CI, hay un test slow
adicional al final que valida el flujo end-to-end con el encoder real.
"""

from __future__ import annotations

import hashlib
import importlib.util
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.irl.features import N_OUTCOME_FEATURES, OUTCOME_FEATURE_NAMES
from guatemala_sim.reasoning_consistency_v2 import ANCHOR_PHRASES_V2
from guatemala_sim.reasoning_consistency_v3 import (
    ConsistencyReportV3,
    V3Encoder,
    assess_reasoning_consistency_v3,
    encode_reasoning_to_w_v3,
    fit_v3_encoder,
)


# --- Mock embedder determinístico --------------------------------------------


class HashEmbedder:
    """Embedder determinístico para tests offline.

    Tokeniza por whitespace, hashea cada token a un vector aleatorio
    seedeado por md5 del token (reproducible cross-proceso, a
    diferencia de `hash(str)` que tiene salt). Embedding del texto =
    promedio de vectores token. Texto vacío → vector cero.

    Garantía clave para los tests: los tokens compartidos entre dos
    textos los acercan en el espacio. Anchor phrases que comparten
    palabras entre sí pero no con otras features producen centroides
    bien separados — suficiente para validar la lógica de v3.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            tokens = t.lower().split()
            if not tokens:
                continue
            vecs = []
            for tk in tokens:
                # Seed determinístico cross-proceso desde md5
                seed = int(hashlib.md5(tk.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                vecs.append(rng.normal(size=self.dim))
            out[i] = np.mean(vecs, axis=0)
        return out


@pytest.fixture
def mock_embedder():
    return HashEmbedder(dim=32)


@pytest.fixture
def encoder(mock_embedder):
    return fit_v3_encoder(embedder=mock_embedder, model_name="mock-hash-32")


# --- fit_v3_encoder ---------------------------------------------------------


def test_fit_v3_encoder_shapes(encoder, mock_embedder):
    assert encoder.feature_names == OUTCOME_FEATURE_NAMES
    K = N_OUTCOME_FEATURES
    D = mock_embedder.dim
    assert encoder.feature_centroids.shape == (K, D)
    # Centroides L2-normalizados: norma cercana a 1 (excepto si todas las
    # phrases de una feature dieron embedding cero, cosa que no debería
    # pasar con HashEmbedder + las anchor phrases reales).
    norms = np.linalg.norm(encoder.feature_centroids, axis=1)
    assert np.all(norms > 0.9)
    assert np.all(norms < 1.1)


def test_fit_v3_encoder_rechaza_features_inconsistentes(mock_embedder):
    bad_phrases = {"feature_inexistente": ("frase",)}
    with pytest.raises(ValueError, match="no coinciden"):
        fit_v3_encoder(embedder=mock_embedder, anchor_phrases=bad_phrases)


def test_fit_v3_encoder_rechaza_phrases_vacias(mock_embedder):
    empty: dict = {name: () for name in OUTCOME_FEATURE_NAMES}
    with pytest.raises(ValueError):
        fit_v3_encoder(embedder=mock_embedder, anchor_phrases=empty)


def test_fit_v3_encoder_falla_si_embedder_devuelve_1d(mock_embedder):
    """Embedder mal implementado debe fallar temprano con error útil."""
    def bad_embedder(texts):
        return np.zeros(len(texts))  # 1D en vez de 2D
    with pytest.raises(ValueError, match="2D"):
        fit_v3_encoder(embedder=bad_embedder)


# --- encode_one / encode_batch ----------------------------------------------


def test_encode_one_shape(encoder):
    w = encoder.encode_one("reducir pobreza extrema")
    assert w.shape == (N_OUTCOME_FEATURES,)
    # Cosines dentro de [-1, 1]
    assert np.all(w >= -1.0 - 1e-9)
    assert np.all(w <= 1.0 + 1e-9)


def test_encode_batch_consistent_con_encode_one(encoder):
    texts = ["reducir pobreza", "honrar la deuda", "fortalecer instituciones"]
    batch = encoder.encode_batch(texts)
    one = np.stack([encoder.encode_one(t) for t in texts])
    np.testing.assert_allclose(batch, one, atol=1e-10)


def test_encode_batch_vacio_devuelve_empty(encoder):
    out = encoder.encode_batch([])
    assert out.shape == (0, N_OUTCOME_FEATURES)


def test_encode_one_texto_vacio_devuelve_ceros(encoder):
    """Texto vacío → embedding cero → cosines cero (no NaN)."""
    w = encoder.encode_one("")
    assert w.shape == (N_OUTCOME_FEATURES,)
    np.testing.assert_allclose(w, 0.0)


def test_encode_recupera_feature_dominante(encoder):
    """Sanity check: una anchor phrase de feature k debe activar más la
    coordenada k que cualquier otra. Si esto falla, el embedder mock
    no separa features — pero con HashEmbedder y phrases distintas
    suele cumplirse en la mayoría de las features.

    Verificamos en agregado: ≥4/6 features deben tener su anchor phrase
    activando su propia coordenada como argmax.
    """
    aciertos = 0
    for k, name in enumerate(OUTCOME_FEATURE_NAMES):
        phrase = ANCHOR_PHRASES_V2[name][0]
        w = encoder.encode_one(phrase)
        if int(np.argmax(w)) == k:
            aciertos += 1
    assert aciertos >= 4, (
        f"Solo {aciertos}/6 anchor phrases activaron su propia feature "
        f"como argmax con HashEmbedder. La lógica del módulo está OK pero "
        f"el mock no separa lo suficiente — esto no es bug del módulo, "
        f"pero el test no es informativo."
    )


# --- encode_reasoning_to_w_v3 (helper convenience) --------------------------


def test_encode_reasoning_normaliza_por_default(encoder):
    w = encode_reasoning_to_w_v3(
        "necesitamos reducir pobreza y proteger a los más vulnerables",
        encoder=encoder,
        normalize=True,
    )
    n = float(np.linalg.norm(w))
    # Normalizado a 1, salvo caso degenerado de cero (que no aplica acá)
    assert abs(n - 1.0) < 1e-9 or n == 0.0


def test_encode_reasoning_sin_normalizar_preserva_magnitudes(encoder):
    w = encode_reasoning_to_w_v3(
        "deuda servicio amortización",
        encoder=encoder,
        normalize=False,
    )
    # Las magnitudes son cosines crudos; pueden ser <1
    assert np.any(w != 0.0)


# --- assess_reasoning_consistency_v3 ----------------------------------------


def test_assess_consistency_basico(encoder):
    razonamientos = [
        "priorizamos la reducción de pobreza y la transferencia social",
        "expandir cobertura de bonos y atender vulnerabilidad",
        "garantizar la cobertura de necesidades básicas",
    ]
    # w_recovered apuntando hacia anti_pobreza
    w_rec = np.zeros(N_OUTCOME_FEATURES)
    w_rec[OUTCOME_FEATURE_NAMES.index("anti_pobreza")] = 1.0

    rep = assess_reasoning_consistency_v3(
        razonamientos, w_rec, threshold=0.3, encoder=encoder,
    )
    assert isinstance(rep, ConsistencyReportV3)
    assert rep.n_turnos == 3
    assert rep.threshold == 0.3
    assert rep.model_name == "mock-hash-32"
    assert rep.per_turn.shape[0] == 3
    assert "cos_per_turn" in rep.per_turn.columns
    # No NaN en este caso (textos no vacíos, w_rec no nulo)
    assert not np.isnan(rep.cosine_similarity)
    assert not np.isnan(rep.angle_degrees)


def test_assess_consistency_w_recovered_invalid_shape(encoder):
    with pytest.raises(ValueError, match="shape"):
        assess_reasoning_consistency_v3(
            ["texto"], np.zeros(3), encoder=encoder,
        )


def test_assess_consistency_razonamientos_vacios(encoder):
    with pytest.raises(ValueError):
        assess_reasoning_consistency_v3(
            [], np.zeros(N_OUTCOME_FEATURES), encoder=encoder,
        )


def test_assess_consistency_w_recovered_cero(encoder):
    """w_recovered = 0 → cosine indeterminado → NaN, sin crash."""
    w_rec = np.zeros(N_OUTCOME_FEATURES)
    rep = assess_reasoning_consistency_v3(
        ["texto"], w_rec, encoder=encoder,
    )
    assert np.isnan(rep.cosine_similarity)
    assert np.isnan(rep.angle_degrees)
    # Flag explícito: no se dispara cuando es indeterminado
    assert rep.inconsistency_flag is False


def test_assess_consistency_textos_vacios_no_crashea(encoder):
    """Razonamientos con string vacío → embedding cero → cos_t NaN."""
    w_rec = np.ones(N_OUTCOME_FEATURES)
    rep = assess_reasoning_consistency_v3(
        ["", "", ""], w_rec, encoder=encoder,
    )
    assert rep.n_turnos == 3
    # Todos los cos_per_turn deben ser NaN (embeddings cero)
    cos_col = rep.per_turn["cos_per_turn"].values
    assert all(np.isnan(c) for c in cos_col)


def test_assess_consistency_threshold_separa_consistentes(encoder):
    """Threshold alto debe disparar el flag aunque la cosine sea positiva."""
    razonamientos = [
        "reducir pobreza y proteger vulnerables",
        "atender el quintil más bajo",
    ]
    w_rec = np.zeros(N_OUTCOME_FEATURES)
    w_rec[0] = 1.0  # alguna dirección
    rep_low = assess_reasoning_consistency_v3(
        razonamientos, w_rec, threshold=-2.0, encoder=encoder,
    )
    rep_high = assess_reasoning_consistency_v3(
        razonamientos, w_rec, threshold=2.0, encoder=encoder,
    )
    assert rep_low.inconsistency_flag is False  # cualquier cos < 2.0 cae
    # Wait, threshold negativo: cos < -2.0 nunca, entonces no flag
    # threshold positivo grande (2.0): cos < 2.0 siempre, entonces SÍ flag
    assert rep_high.inconsistency_flag is True


# --- consistencia API con v1/v2 ---------------------------------------------


def test_v3_report_tiene_campos_paralelos_a_v2(encoder):
    """Smoke test: los nombres de campos clave deben existir, alineados
    con `ConsistencyReportV2` en `reasoning_consistency_v2`."""
    razonamientos = ["test uno", "test dos"]
    w_rec = np.ones(N_OUTCOME_FEATURES)
    rep = assess_reasoning_consistency_v3(
        razonamientos, w_rec, encoder=encoder,
    )
    expected_attrs = {
        "n_turnos", "cosine_similarity", "angle_degrees",
        "inconsistent_turns", "inconsistency_flag", "threshold",
        "w_razonamiento_avg", "w_recovered_normalized", "per_turn",
        "model_name",
    }
    actual = set(vars(rep).keys())
    assert expected_attrs.issubset(actual), f"faltan: {expected_attrs - actual}"


# --- end-to-end con sentence-transformers real (slow, opt-in) ----------------


_HAS_ST = importlib.util.find_spec("sentence_transformers") is not None


@pytest.mark.skipif(
    not _HAS_ST,
    reason="sentence-transformers no instalado; instalar [embeddings] extra",
)
@pytest.mark.slow
def test_default_embedder_e2e_smoke():
    """Sólo corre si sentence-transformers + modelo están disponibles.
    Carga el default, codifica un par de textos, valida shape.
    """
    from guatemala_sim.reasoning_consistency_v3 import make_default_embedder
    embedder = make_default_embedder()
    enc = fit_v3_encoder(embedder=embedder)
    razonamientos = [
        "priorizo la reducción de pobreza extrema",
        "necesito mantener la sostenibilidad fiscal",
    ]
    w_rec = np.ones(N_OUTCOME_FEATURES)
    rep = assess_reasoning_consistency_v3(razonamientos, w_rec, encoder=enc)
    assert rep.n_turnos == 2
    assert not np.isnan(rep.cosine_similarity)
