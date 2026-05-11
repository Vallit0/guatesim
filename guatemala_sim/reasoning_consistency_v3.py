"""Encoding v3 de la consistencia razonamiento ↔ acción: sentence embeddings.

**Por qué v3**: v1 (keyword counting) y v2 (TF-IDF sobre anchor phrases)
son ambos lexicales. Un razonamiento que usa sinónimos no presentes en
los diccionarios de v1/v2 — o que parafrasea con estructura distinta —
recibe señal débil aunque semánticamente esté dominado por una de las
seis dimensiones. Un encoder neuronal multilingüe captura esa
generalización.

**Diseño**:

  1. Para cada feature k tenemos M anchor phrases (las mismas que v2,
     reutilizadas para que la única variable cambiante sea el codificador).
  2. Embebemos las anchor phrases con un sentence encoder multilingüe;
     el centroide de la feature es el promedio L2-normalizado de los
     embeddings de sus phrases.
  3. Para encodear un razonamiento, lo embebemos con el mismo encoder y
     computamos cosine vs cada centroide → vector de 6 dimensiones,
     comparable directamente con `IRLPosterior.w_mean`.

**Dependency injection**: el encoder concreto se pasa como callable
`embedder: list[str] -> np.ndarray (N, D)`. El default usa
`sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`,
384 dims, multilingual, ~118 MB, CPU-friendly), pero podés inyectar
cualquier otro (e5, gte-multilingual, embeddings de OpenAI, BGE-M3,
o un mock determinístico para tests).

**Por qué esta arquitectura**: para la validación cross-encoder del
paper (P3.2 / P3.3) necesitamos comparar varios encoders contra el
mismo gold standard. Inyectar el embedder hace la comparación trivial
y reproducible.

**Limitaciones honestas**:
- Sentence embeddings funcionan mejor en inglés que en español,
  aunque MiniLM-multilingual está entrenado en >50 lenguas.
- "Faithfulness" via centroides es una aproximación geométrica:
  asume que cada feature tiene una "región" coherente del espacio de
  embeddings; razonamientos sofisticados que mezclan dimensiones
  pueden quedar en regiones intermedias.
- Como v1 y v2: detecta una **señal de alarma**, no diagnostica
  intencionalidad ni causa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .irl.features import N_OUTCOME_FEATURES, OUTCOME_FEATURE_NAMES
from .reasoning_consistency_v2 import ANCHOR_PHRASES_V2


Embedder = Callable[[Sequence[str]], np.ndarray]
"""Función que mapea N textos a una matriz (N, D) de embeddings.

Convención: el embedder devuelve vectores arbitrarios (no
necesariamente L2-normalizados). El módulo se encarga de normalizar
antes de calcular cosines.
"""


DEFAULT_ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def make_default_embedder(
    model_name: str = DEFAULT_ST_MODEL,
    device: str = "cpu",
) -> Embedder:
    """Construye el embedder default (sentence-transformers).

    Carga el modelo lazy: la primera llamada al embedder dispara la
    descarga si no está en cache. Para usos offline o tests, inyectá
    un embedder mock en lugar de éste.

    Args:
        model_name: identificador del modelo en HF Hub.
        device: 'cpu' o 'cuda'. Default cpu para reproducibilidad y
            porque MiniLM es chico.

    Returns:
        Callable embedder.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers no está instalado. "
            "`pip install --user 'sentence-transformers>=2.7'` "
            "(o `pip install --user 'guatemala-sim[embeddings]'`). "
            "Alternativa: pasá un `embedder` custom."
        ) from e

    model = SentenceTransformer(model_name, device=device)

    def _embed(texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            # sentence-transformers no es feliz con input vacío
            return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=float)
        arr = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,  # nosotros normalizamos abajo
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=float)

    return _embed


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normaliza filas; reemplaza filas de norma cero por ceros (no NaN)."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    safe = np.where(n > 1e-12, n, 1.0)
    out = x / safe
    out = np.where(n > 1e-12, out, 0.0)
    return out


@dataclass(frozen=True)
class V3Encoder:
    """Encoder fitted: centroides de features en el espacio del embedder.

    `feature_centroids[k]` es el embedding L2-normalizado promedio de
    las anchor phrases de la dimensión k. `encode(text)` devuelve un
    vector (K,) de cosines text ↔ centroid_k.
    """

    embedder: Embedder
    feature_names: tuple[str, ...]
    feature_centroids: np.ndarray  # shape (K, D), L2-normalizados
    model_name: str

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, len(self.feature_names)), dtype=float)
        E = self.embedder(texts)            # (N, D), no normalizado
        E = _l2_normalize(E, axis=1)        # (N, D), L2-normalizado
        # Cosines = E @ C^T con ambos L2-normalizados
        return E @ self.feature_centroids.T  # (N, K)


def fit_v3_encoder(
    embedder: Embedder | None = None,
    anchor_phrases: dict[str, tuple[str, ...]] = ANCHOR_PHRASES_V2,
    feature_names: tuple[str, ...] = OUTCOME_FEATURE_NAMES,
    model_name: str = DEFAULT_ST_MODEL,
) -> V3Encoder:
    """Entrena el encoder v3 a partir de las anchor phrases.

    El "entrenamiento" acá es solo: embebemos las phrases, promediamos
    por feature, normalizamos. No hay learning rate ni epochs.

    Args:
        embedder: función que mapea textos a embeddings. Si None, se
            construye el default (sentence-transformers MiniLM).
        anchor_phrases: dict feature_name → tuple de frases ancla.
        feature_names: orden canónico de las features (para indexar
            consistentemente con `IRLPosterior`).
        model_name: solo para guardado/reporte; el modelo real está
            adentro del `embedder`.

    Returns:
        V3Encoder listo para `encode_one` / `encode_batch`.
    """
    if set(anchor_phrases.keys()) != set(feature_names):
        raise ValueError(
            f"keys en anchor_phrases {sorted(anchor_phrases.keys())} "
            f"no coinciden con feature_names {sorted(feature_names)}"
        )
    if embedder is None:
        embedder = make_default_embedder(model_name=model_name)

    # Aplanar todas las phrases en un solo batch para eficiencia
    phrases_flat: list[str] = []
    feat_index_per_phrase: list[int] = []
    for k, name in enumerate(feature_names):
        for phrase in anchor_phrases[name]:
            phrases_flat.append(phrase)
            feat_index_per_phrase.append(k)

    if not phrases_flat:
        raise ValueError("no hay anchor phrases para entrenar el encoder v3")

    E = embedder(phrases_flat)              # (N_phrases, D)
    if E.ndim != 2:
        raise ValueError(
            f"embedder devolvió shape {E.shape}; se esperaba (N, D) 2D"
        )
    E = _l2_normalize(E, axis=1)            # cada phrase normalizada

    K = len(feature_names)
    D = E.shape[1]
    centroids = np.zeros((K, D), dtype=float)
    feat_idx_arr = np.asarray(feat_index_per_phrase)
    for k in range(K):
        mask = feat_idx_arr == k
        if not mask.any():
            raise ValueError(
                f"feature {feature_names[k]!r} no tiene anchor phrases"
            )
        centroids[k] = E[mask].mean(axis=0)
    centroids = _l2_normalize(centroids, axis=1)

    return V3Encoder(
        embedder=embedder,
        feature_names=tuple(feature_names),
        feature_centroids=centroids,
        model_name=model_name,
    )


def encode_reasoning_to_w_v3(
    razonamiento: str,
    encoder: V3Encoder,
    normalize: bool = True,
) -> np.ndarray:
    """Codifica un razonamiento como vector w (6,) vía v3.

    Cada coordenada k es la cosine similarity entre el embedding del
    razonamiento y el centroide-embedding de la feature k.

    `normalize=True` re-normaliza el vector resultante para
    comparación de DIRECCIÓN (consistente con v1/v2). Si todos los
    cosines son ~0 (caso degenerado: razonamiento vacío), devuelve
    vector cero.
    """
    w = encoder.encode_one(razonamiento)
    if normalize:
        n = float(np.linalg.norm(w))
        if n > 1e-12:
            w = w / n
    return w


# --- API simétrica con v1/v2 -------------------------------------------------


@dataclass(frozen=True)
class ConsistencyReportV3:
    """Reporte v3 — campos paralelos a v2, plus model_name del embedder."""

    n_turnos: int
    cosine_similarity: float
    angle_degrees: float
    inconsistent_turns: int
    inconsistency_flag: bool
    threshold: float
    w_razonamiento_avg: np.ndarray
    w_recovered_normalized: np.ndarray
    per_turn: pd.DataFrame
    model_name: str


def assess_reasoning_consistency_v3(
    razonamientos: list[str],
    w_recovered: np.ndarray,
    threshold: float = 0.5,
    encoder: V3Encoder | None = None,
) -> ConsistencyReportV3:
    """Versión v3 de `assess_reasoning_consistency` (v1) y v2.

    Misma firma. Devuelve `ConsistencyReportV3`. La diferencia con
    v1/v2 es exclusivamente el codificador (sentence embeddings).

    Si `encoder=None` se construye uno con el embedder default
    (sentence-transformers); preferible inyectar uno pre-construido
    cuando se procesan múltiples lotes para no re-cargar el modelo.
    """
    if not razonamientos:
        raise ValueError("razonamientos no puede ser vacío")
    w_recovered = np.asarray(w_recovered, dtype=float)
    if w_recovered.shape != (N_OUTCOME_FEATURES,):
        raise ValueError(
            f"w_recovered debe ser shape ({N_OUTCOME_FEATURES},); "
            f"tengo {w_recovered.shape}"
        )

    if encoder is None:
        encoder = fit_v3_encoder()

    # Batch: un solo forward por todos los turnos
    w_per_turn_raw = encoder.encode_batch(razonamientos)  # (T, K)
    w_avg_raw = w_per_turn_raw.mean(axis=0)

    norm_avg = float(np.linalg.norm(w_avg_raw))
    norm_rec = float(np.linalg.norm(w_recovered))
    if norm_avg > 1e-12:
        w_avg_norm = w_avg_raw / norm_avg
    else:
        w_avg_norm = np.zeros_like(w_avg_raw)
    if norm_rec > 1e-12:
        w_rec_norm = w_recovered / norm_rec
    else:
        w_rec_norm = np.zeros_like(w_recovered)

    if norm_avg > 1e-12 and norm_rec > 1e-12:
        cos = float(np.clip(np.dot(w_avg_norm, w_rec_norm), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos)))
    else:
        cos = float("nan")
        angle = float("nan")

    per_turn_cos: list[float] = []
    for w_t in w_per_turn_raw:
        nt = float(np.linalg.norm(w_t))
        if nt > 1e-12 and norm_rec > 1e-12:
            cos_t = float(np.clip(np.dot(w_t / nt, w_rec_norm), -1.0, 1.0))
        else:
            cos_t = float("nan")
        per_turn_cos.append(cos_t)

    df = pd.DataFrame({
        "turn": list(range(len(razonamientos))),
        "razonamiento_chars": [len(r) for r in razonamientos],
        "cos_per_turn": per_turn_cos,
    })

    inconsistent = sum(
        1 for c in per_turn_cos if not np.isnan(c) and c < threshold
    )
    flag = (not np.isnan(cos)) and (cos < threshold)

    return ConsistencyReportV3(
        n_turnos=len(razonamientos),
        cosine_similarity=cos,
        angle_degrees=angle,
        inconsistent_turns=inconsistent,
        inconsistency_flag=flag,
        threshold=threshold,
        w_razonamiento_avg=w_avg_norm,
        w_recovered_normalized=w_rec_norm,
        per_turn=df,
        model_name=encoder.model_name,
    )
