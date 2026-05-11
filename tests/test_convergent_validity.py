"""Tests del módulo de convergent validity (faithfulness multi-encoder)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.convergent_validity import (
    compute_kappa_table,
    compute_per_seed_flags,
    extract_razonamientos,
    get_w_recovered,
    parse_seed_model_from_jsonl_name,
    write_robustness_report,
)
from guatemala_sim.irl.features import OUTCOME_FEATURE_NAMES
from guatemala_sim.reasoning_consistency_v3 import fit_v3_encoder
from tests.test_reasoning_consistency_v3 import HashEmbedder


# --- helpers para fabricar inputs sintéticos --------------------------------


def _make_jsonl(path: Path, razonamientos: list[str]) -> None:
    """Crea un JSONL minimal con campo decision.razonamiento por turno."""
    with path.open("w", encoding="utf-8") as f:
        for t, raz in enumerate(razonamientos):
            rec = {
                "t": t,
                "decision": {"razonamiento": raz},
                "indicadores": {},
                "shocks": [],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_posteriors_csv(path: Path, entries: list[tuple[int, str, np.ndarray]]) -> None:
    """entries: [(seed, model, w_mean[6]), ...]"""
    rows = []
    for seed, model, w in entries:
        for k, name in enumerate(OUTCOME_FEATURE_NAMES):
            rows.append({
                "seed": seed,
                "replica": 0,
                "model": model,
                "dim": name,
                "w_mean": float(w[k]),
                "hdi_lo": float(w[k]) - 1.0,
                "hdi_hi": float(w[k]) + 1.0,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


# --- parse_seed_model_from_jsonl_name ---------------------------------------


def test_parse_jsonl_name_basico():
    assert parse_seed_model_from_jsonl_name("seed007_claude") == (7, "claude")
    assert parse_seed_model_from_jsonl_name("seed020_openai") == (20, "openai")


def test_parse_jsonl_name_invalido():
    assert parse_seed_model_from_jsonl_name("foo") is None
    assert parse_seed_model_from_jsonl_name("seed_no_digits") is None


# --- extract_razonamientos --------------------------------------------------


def test_extract_razonamientos_lee_jsonl(tmp_path: Path):
    p = tmp_path / "seed001_test.jsonl"
    _make_jsonl(p, ["razon 1", "razon 2", "razon 3"])
    out = extract_razonamientos(p)
    assert out == ["razon 1", "razon 2", "razon 3"]


def test_extract_razonamientos_vacio_si_no_hay_decision(tmp_path: Path):
    p = tmp_path / "vacio.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"t": 0, "indicadores": {}}) + "\n")
    out = extract_razonamientos(p)
    assert out == []


# --- get_w_recovered --------------------------------------------------------


def test_get_w_recovered_lee_csv(tmp_path: Path):
    p = tmp_path / "post.csv"
    w_true = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    _make_posteriors_csv(p, [(7, "claude", w_true)])
    df = pd.read_csv(p)
    w = get_w_recovered(df, 7, "claude")
    assert w is not None
    np.testing.assert_array_equal(w, w_true)


def test_get_w_recovered_devuelve_none_si_falta(tmp_path: Path):
    p = tmp_path / "post.csv"
    _make_posteriors_csv(p, [(1, "claude", np.zeros(6))])
    df = pd.read_csv(p)
    assert get_w_recovered(df, 999, "claude") is None
    assert get_w_recovered(df, 1, "ghost-model") is None


# --- compute_per_seed_flags --------------------------------------------------


@pytest.fixture
def synthetic_batch(tmp_path: Path):
    """Mini-batch sintético con 2 seeds × 2 modelos."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _make_jsonl(runs / "seed001_claude.jsonl", [
        "priorizo reducir la pobreza extrema y la migración",
        "expandir el bono de protección social",
        "atender la pobreza más severa primero",
    ])
    _make_jsonl(runs / "seed001_openai.jsonl", [
        "necesito mantener la sostenibilidad fiscal de la deuda",
        "honrar los compromisos con acreedores",
        "sostener la calificación crediticia",
    ])
    _make_jsonl(runs / "seed002_claude.jsonl", [
        "controlar la inflación es prioridad",
        "anclaje de expectativas inflacionarias",
        "estabilidad de precios sobre todo",
    ])
    _make_jsonl(runs / "seed002_openai.jsonl", [
        "fortalecer las instituciones del estado",
        "transparencia y anticorrupción",
        "rendición de cuentas",
    ])
    posteriors = tmp_path / "posteriors.csv"
    # w apunta a la feature dominante de cada conjunto de razonamientos
    _make_posteriors_csv(posteriors, [
        (1, "claude", np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])),  # anti_pobreza
        (1, "openai", np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])),  # anti_deuda
        (2, "claude", np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])),  # anti_desviacion_inflacion
        (2, "openai", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),  # pro_confianza
    ])
    return runs, posteriors


def test_compute_per_seed_flags_estructura(synthetic_batch):
    runs, posteriors = synthetic_batch
    v3_enc = fit_v3_encoder(embedder=HashEmbedder(dim=32), model_name="mock-32")
    df = compute_per_seed_flags(runs, posteriors, v3_encoder=v3_enc)
    assert len(df) == 4  # 2 seeds × 2 models
    expected_cols = {
        "seed", "model", "n_turnos",
        "cos_v1", "flag_v1", "inconsistent_v1",
        "cos_v2", "flag_v2", "inconsistent_v2",
        "cos_v3", "flag_v3", "inconsistent_v3",
        "v3_model",
    }
    assert expected_cols.issubset(df.columns)
    # Ordenado por (model, seed)
    assert list(df["model"].values) == sorted(df["model"].values)


def test_compute_per_seed_flags_v1_detecta_features_obvias(synthetic_batch):
    """Razonamientos llenos de keywords v1 con w apuntando a la misma
    feature → cos_v1 alto, flag_v1=0 (consistente)."""
    runs, posteriors = synthetic_batch
    v3_enc = fit_v3_encoder(embedder=HashEmbedder(dim=32), model_name="mock-32")
    df = compute_per_seed_flags(runs, posteriors, v3_encoder=v3_enc, threshold=0.5)
    # Para seed 1 claude: razonamientos sobre pobreza, w=anti_pobreza → consistente v1
    s1c = df[(df["seed"] == 1) & (df["model"] == "claude")].iloc[0]
    assert s1c["cos_v1"] > 0.5
    assert s1c["flag_v1"] == 0


# --- compute_kappa_table ----------------------------------------------------


def test_kappa_table_estructura(synthetic_batch):
    runs, posteriors = synthetic_batch
    v3_enc = fit_v3_encoder(embedder=HashEmbedder(dim=32), model_name="mock-32")
    per_seed = compute_per_seed_flags(runs, posteriors, v3_encoder=v3_enc)
    kappa = compute_kappa_table(per_seed)
    # 2 modelos × 3 pares = 6 filas
    assert len(kappa) == 6
    assert {"v1", "v2", "v3"} == set(kappa["encoder_a"]) | set(kappa["encoder_b"])
    # Tabla de contingencia debe sumar a n
    for _, row in kappa.iterrows():
        total = row["both_flag"] + row["only_a_flag"] + row["only_b_flag"] + row["neither_flag"]
        assert total == row["n"]


# --- write_robustness_report -----------------------------------------------


def test_write_robustness_report_genera_archivos(synthetic_batch, tmp_path):
    runs, posteriors = synthetic_batch
    v3_enc = fit_v3_encoder(embedder=HashEmbedder(dim=32), model_name="mock-32")
    per_seed = compute_per_seed_flags(runs, posteriors, v3_encoder=v3_enc)
    kappa = compute_kappa_table(per_seed)
    out = tmp_path / "out"
    paths = write_robustness_report(per_seed, kappa, out)
    assert paths["per_seed"].exists()
    assert paths["kappa"].exists()
    assert paths["report"].exists()
    md = paths["report"].read_text(encoding="utf-8")
    assert "Cohen's κ" in md
    assert "v1" in md and "v2" in md and "v3" in md
