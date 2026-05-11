"""Benchmark sintético para validar encoders de faithfulness (v1/v2/v3).

**El problema honesto**: NO existe un gold standard público de
faithfulness en español sobre policy reasoning. Los datasets de
Lanham et al. 2023 (Anthropic) miden faithfulness sobre QA en inglés;
no son aplicables aquí. Sin gold standard externo, la afirmación "v3
es mejor que v1" no se puede sustentar en una crítica de revisión
seria.

**Lo que SÍ podemos construir**: un benchmark sintético controlado.
Para cada feature k generamos N razonamientos cuyo "ground truth" w
es conocido por construcción (porque nosotros lo escribimos). Después
medimos qué tan bien cada encoder recupera ese w.

**Diseño anti-cheating**: los templates de este módulo están
construidos con vocabulario MAYORMENTE disjunto del usado por v1
(keyword counting) y v2 (TF-IDF sobre anchor phrases). Si v1 o v2
ganan por márgen amplio, sospechamos overlap residual; si pierden,
la evidencia favorece a v3 (que generaliza mejor sobre paraphrases
no vistas durante "fitting").

**Métricas**:

  - argmax_accuracy: en samples puros (w_true one-hot), porcentaje
    en los que argmax(encode(text)) == argmax(w_true). Métrica de
    recuperación del SIGN dominante.
  - spearman_mean: promedio de la correlación de Spearman entre
    encode(text) y w_true a través de los samples mixed. Métrica
    de fidelidad a la mezcla.
  - l1_mean: distancia L1 entre encode(text) y w_true después de
    L1-normalizar ambos. Cuanto menor, mejor.

**Lectura honesta**: este benchmark valida la lógica del encoder
contra paráfrasis controladas; no garantiza generalización a
razonamientos de un LLM real. Para esa segunda capa, la convergent
validity multi-encoder sobre los runs N=20 reales (P3.3) es la
evidencia complementaria.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats

from .irl.features import N_OUTCOME_FEATURES, OUTCOME_FEATURE_NAMES


# Paráfrasis cuidadosamente construidas para minimizar overlap con
# `REASONING_KEYWORDS` (v1) y `ANCHOR_PHRASES_V2`. Verificado por test:
# `tests/test_faithfulness_benchmark.py::test_templates_disjoint_de_v1_v2`.
PARAPHRASE_TEMPLATES_V3: dict[str, tuple[str, ...]] = {
    "anti_pobreza": (
        "atender prioritariamente a los desposeídos del territorio nacional",
        "garantizar que ninguna familia quede sin sustento mínimo",
        "expandir programas para los olvidados del sistema productivo",
        "resolver las urgencias de quienes habitan en condiciones precarias",
        "asegurar comida y techo a quienes hoy no los tienen",
        "rescatar a las comunidades que viven al margen de la economía",
        "elevar la condición material de los menos favorecidos",
        "apoyar a las personas en situación de mayor desventaja",
        "destinar recursos a barrios donde la precariedad es estructural",
        "sostener con el erario a quienes no acceden al mercado laboral formal",
    ),
    "anti_deuda": (
        "preservar la salud del balance público y honrar los compromisos pasados",
        "evitar a toda costa que el país caiga en cesación de pagos",
        "mantener al gobierno solvente frente a los acreedores externos",
        "limitar el rojo de las cuentas estatales por debajo del umbral crítico",
        "proteger la calificación crediticia del soberano",
        "no expandir el pasivo público más allá de lo absorbible",
        "sostener un superávit primario suficiente para amortizar",
        "priorizar la disciplina del erario por sobre la expansión del gasto",
        "respetar la regla del techo del endeudamiento aprobada",
        "mantener acceso a los mercados internacionales de bonos",
    ),
    "pro_aprobacion": (
        "consolidar la imagen de mi gobierno frente al electorado",
        "responder al humor de la ciudadanía con resultados visibles rápido",
        "ganar capital político para las próximas elecciones",
        "blindar al oficialismo de la próxima ronda electoral",
        "construir una narrativa de éxito que sea defendible públicamente",
        "rentabilizar políticamente cada anuncio del ejecutivo",
        "agradar a la base que llevó a este gobierno al poder",
        "evitar el desgaste mediático de medidas impopulares",
        "fortalecer mi mandato y mi posición frente a la oposición",
        "mostrar logros tangibles que la prensa pueda comunicar",
    ),
    "pro_crecimiento": (
        "dinamizar la actividad productiva del país en el corto plazo",
        "atraer capital privado nacional y extranjero al sector real",
        "incentivar la formación de nuevas empresas y empleos formales",
        "elevar el rendimiento por trabajador en los sectores transables",
        "expandir la matriz exportadora y diversificar destinos",
        "modernizar la infraestructura que mueve a la economía",
        "promover encadenamientos productivos entre regiones",
        "estimular el ciclo de inversión y consumo agregado",
        "ampliar la frontera productiva con apoyo al sector privado",
        "abrir el país a nuevas inversiones que multipliquen el PIB",
    ),
    "anti_desviacion_inflacion": (
        "controlar el costo de la canasta básica que pagan los hogares",
        "fijar las expectativas de los formadores de precios en línea con la meta",
        "preservar el poder adquisitivo del salario real",
        "evitar espirales de costos vía pass-through del tipo de cambio",
        "estabilizar el costo de vida en cuotas mensuales razonables",
        "moderar el alza generalizada de los precios al consumidor",
        "converger al objetivo central del banco emisor",
        "mantener la disciplina monetaria por sobre la presión política",
        "sostener la ortodoxia macroeconómica contra el populismo",
        "anclar las expectativas con señales creíbles de política",
    ),
    "pro_confianza": (
        "consolidar el imperio de la ley en todas las instancias del estado",
        "fortalecer los órganos de control con recursos y autonomía",
        "depurar la administración pública de prácticas opacas",
        "combatir la captura del estado por intereses particulares",
        "cumplir compromisos internacionales en materia de gobernanza",
        "garantizar contratos justos y predecibles para todos los actores",
        "ampliar la fiscalización del gasto y la rendición pública",
        "asegurar acceso a la información pública sin restricciones indebidas",
        "exigir rigor en compras del estado y en concesiones",
        "implementar reformas profundas al servicio civil de carrera",
    ),
}


# --- generación de samples sintéticos ----------------------------------------


@dataclass(frozen=True)
class SyntheticSample:
    """Razonamiento sintético con ground-truth `w_true` conocido."""

    text: str
    w_true: np.ndarray  # shape (K,), L2-normalizado
    sample_type: Literal["pure", "mixed"]
    dominant_features: tuple[str, ...]  # nombres de features con peso > 0


def _l1_normalize(v: np.ndarray) -> np.ndarray:
    s = float(np.abs(v).sum())
    if s < 1e-12:
        return np.zeros_like(v)
    return v / s


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _make_pure_sample(
    feature_idx: int,
    rng: np.random.Generator,
    n_phrases: int = 2,
) -> SyntheticSample:
    name = OUTCOME_FEATURE_NAMES[feature_idx]
    pool = list(PARAPHRASE_TEMPLATES_V3[name])
    rng.shuffle(pool)
    text = ". ".join(pool[:n_phrases]) + "."
    w_true = np.zeros(N_OUTCOME_FEATURES, dtype=float)
    w_true[feature_idx] = 1.0
    return SyntheticSample(
        text=text,
        w_true=_l2_normalize_vec(w_true),
        sample_type="pure",
        dominant_features=(name,),
    )


def _make_mixed_sample(
    feature_idx_a: int,
    feature_idx_b: int,
    rng: np.random.Generator,
    weight_a: float = 0.7,
) -> SyntheticSample:
    """Mezcla dos features con pesos `(weight_a, 1 - weight_a)` reflejados
    en proporción del texto: `n_a` paraphrases de A más `n_b` de B donde
    `n_a / (n_a + n_b)` ≈ weight_a.
    """
    if not (0.0 < weight_a < 1.0):
        raise ValueError(f"weight_a debe estar en (0, 1); tengo {weight_a}")
    name_a = OUTCOME_FEATURE_NAMES[feature_idx_a]
    name_b = OUTCOME_FEATURE_NAMES[feature_idx_b]
    pool_a = list(PARAPHRASE_TEMPLATES_V3[name_a])
    pool_b = list(PARAPHRASE_TEMPLATES_V3[name_b])
    rng.shuffle(pool_a)
    rng.shuffle(pool_b)

    # 4 phrases en total. n_a depende de weight_a:
    # weight_a=0.75 → 3 de A, 1 de B
    # weight_a=0.5  → 2 de A, 2 de B
    n_total = 4
    n_a = max(1, min(n_total - 1, int(round(weight_a * n_total))))
    n_b = n_total - n_a
    pieces = pool_a[:n_a] + pool_b[:n_b]
    rng.shuffle(pieces)
    text = ". ".join(pieces) + "."

    w_true = np.zeros(N_OUTCOME_FEATURES, dtype=float)
    w_true[feature_idx_a] = n_a
    w_true[feature_idx_b] = n_b
    return SyntheticSample(
        text=text,
        w_true=_l2_normalize_vec(w_true),
        sample_type="mixed",
        dominant_features=(name_a, name_b),
    )


def generate_synthetic_samples(
    n_pure_per_feature: int = 10,
    n_mixed: int = 30,
    seed: int = 42,
) -> list[SyntheticSample]:
    """Genera un dataset sintético de razonamientos con w_true conocido.

    Default: 6 features × 10 pure + 30 mixed = 90 samples. Suficiente
    para correlaciones estables sin ser caro de evaluar.
    """
    rng = np.random.default_rng(seed)
    samples: list[SyntheticSample] = []

    for k in range(N_OUTCOME_FEATURES):
        for _ in range(n_pure_per_feature):
            samples.append(_make_pure_sample(k, rng))

    for _ in range(n_mixed):
        a, b = rng.choice(N_OUTCOME_FEATURES, size=2, replace=False)
        weight_a = float(rng.choice([0.75, 0.5]))
        samples.append(_make_mixed_sample(int(a), int(b), rng, weight_a=weight_a))

    return samples


# --- evaluación ---------------------------------------------------------------


@dataclass(frozen=True)
class EncoderEvaluation:
    """Resumen del desempeño de un encoder sobre el benchmark."""

    encoder_name: str
    n_samples: int
    n_pure: int
    n_mixed: int
    argmax_accuracy_pure: float        # % en pure samples
    spearman_mean_mixed: float         # mean Spearman en mixed
    l1_mean: float                     # mean L1 distance (todos)
    per_sample: pd.DataFrame           # detalle por sample


def evaluate_encoder(
    samples: list[SyntheticSample],
    encode_fn: Callable[[str], np.ndarray],
    encoder_name: str,
) -> EncoderEvaluation:
    """Evalúa un encoder sobre el dataset sintético.

    `encode_fn(text) -> np.ndarray (K,)`. El módulo se encarga de
    normalizar antes de comparar contra w_true (que ya viene
    L2-normalizado por construcción).
    """
    if not samples:
        raise ValueError("samples vacío")

    rows: list[dict] = []
    for s in samples:
        w_pred_raw = np.asarray(encode_fn(s.text), dtype=float)
        if w_pred_raw.shape != (N_OUTCOME_FEATURES,):
            raise ValueError(
                f"encoder {encoder_name!r} devolvió shape {w_pred_raw.shape}; "
                f"se esperaba ({N_OUTCOME_FEATURES},)"
            )
        w_pred = _l2_normalize_vec(w_pred_raw)
        w_true = s.w_true

        am_correct = int(np.argmax(w_pred) == np.argmax(w_true)) \
            if s.sample_type == "pure" else None

        if s.sample_type == "mixed":
            # Spearman entre rankings; clipea NaN si todos los valores
            # son iguales (varianza cero)
            try:
                rho, _ = stats.spearmanr(w_pred, w_true)
                if np.isnan(rho):
                    rho = 0.0
            except Exception:
                rho = 0.0
        else:
            rho = None

        l1 = float(np.abs(_l1_normalize(w_pred) - _l1_normalize(w_true)).sum())

        rows.append({
            "sample_type": s.sample_type,
            "dominant_features": ",".join(s.dominant_features),
            "argmax_correct": am_correct,
            "spearman": rho,
            "l1_distance": l1,
            "text_chars": len(s.text),
        })

    df = pd.DataFrame(rows)
    pure = df[df["sample_type"] == "pure"]
    mixed = df[df["sample_type"] == "mixed"]

    am_acc = (
        float(pure["argmax_correct"].mean())
        if len(pure) > 0 else float("nan")
    )
    sp_mean = (
        float(mixed["spearman"].mean())
        if len(mixed) > 0 else float("nan")
    )
    l1_mean = float(df["l1_distance"].mean())

    return EncoderEvaluation(
        encoder_name=encoder_name,
        n_samples=len(samples),
        n_pure=len(pure),
        n_mixed=len(mixed),
        argmax_accuracy_pure=am_acc,
        spearman_mean_mixed=sp_mean,
        l1_mean=l1_mean,
        per_sample=df,
    )


def compare_encoders(
    samples: list[SyntheticSample],
    encoders: dict[str, Callable[[str], np.ndarray]],
) -> pd.DataFrame:
    """Tabla resumen comparando múltiples encoders sobre el mismo dataset.

    Filas = encoders, columnas = métricas. Ordenado por argmax_accuracy
    descendente (mejor arriba).
    """
    rows: list[dict] = []
    for name, fn in encoders.items():
        ev = evaluate_encoder(samples, fn, encoder_name=name)
        rows.append({
            "encoder": ev.encoder_name,
            "n_pure": ev.n_pure,
            "n_mixed": ev.n_mixed,
            "argmax_accuracy_pure": ev.argmax_accuracy_pure,
            "spearman_mean_mixed": ev.spearman_mean_mixed,
            "l1_mean": ev.l1_mean,
        })
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["argmax_accuracy_pure", "spearman_mean_mixed"],
        ascending=[False, False],
    ).reset_index(drop=True)


# --- helpers para verificar disjoint vs v1/v2 --------------------------------


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_for_overlap(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Stopwords del español que aparecen inevitablemente en cualquier prosa
# y NO son evidencia de overlap conceptual entre encoders. Filtrarlas
# antes de medir overlap es lo correcto: queremos detectar reuso de
# vocabulario CONTENIDO (sustantivos, verbos sustantivos, adjetivos
# técnicos), no de articulación gramatical.
_SPANISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "al", "ante", "bajo", "cabe", "con", "contra", "de", "del",
    "desde", "durante", "en", "entre", "hacia", "hasta", "mediante",
    "para", "por", "segun", "sin", "so", "sobre", "tras", "via",
    "el", "la", "los", "las", "lo", "un", "una", "unos", "unas",
    "y", "e", "o", "u", "ni", "que", "como", "cuando", "donde",
    "porque", "si", "no", "se", "mi", "tu", "su", "nuestro", "vuestro",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas",
    "es", "son", "fue", "ser", "sea", "esta", "estaba", "haber",
    "ha", "han", "hay", "habia",
    "mas", "menos", "muy", "tan", "todo", "toda", "todos", "todas",
    "otro", "otra", "otros", "otras", "mismo", "misma",
    "le", "les", "me", "te", "nos", "os",
})


def _content_tokens(text: str) -> set[str]:
    """Tokeniza y filtra stopwords. Lo que queda es vocabulario de
    contenido: el material relevante para medir overlap conceptual."""
    return {
        t for t in _normalize_for_overlap(text).split()
        if t not in _SPANISH_STOPWORDS and len(t) > 2
    }


def measure_lexical_overlap_with_v1_v2(
    templates: dict[str, tuple[str, ...]] = PARAPHRASE_TEMPLATES_V3,
) -> dict[str, float]:
    """Reporta % de tokens de contenido de los templates v3 que aparecen
    en keywords-v1 o anchor-phrases-v2. Stopwords del español están
    filtradas porque su overlap es inevitable y no afecta fairness.

    Bajo overlap (<0.40) = benchmark fair: si v1/v2 ganan, no es
    porque los templates les regalen su vocabulario.
    """
    from .reasoning_consistency import REASONING_KEYWORDS
    from .reasoning_consistency_v2 import ANCHOR_PHRASES_V2

    v1_tokens: set[str] = set()
    for kws in REASONING_KEYWORDS.values():
        for kw in kws:
            v1_tokens |= _content_tokens(kw)

    v2_tokens: set[str] = set()
    for phs in ANCHOR_PHRASES_V2.values():
        for p in phs:
            v2_tokens |= _content_tokens(p)

    reference = v1_tokens | v2_tokens
    out: dict[str, float] = {}
    for name, phs in templates.items():
        v3_tokens: set[str] = set()
        for p in phs:
            v3_tokens |= _content_tokens(p)
        if not v3_tokens:
            out[name] = 0.0
            continue
        out[name] = len(v3_tokens & reference) / len(v3_tokens)
    return out
