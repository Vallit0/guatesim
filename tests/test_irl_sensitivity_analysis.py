"""Tests del orquestador de robustness checks.

Sólo testea las funciones puras (no toca el batch real ni NUTS).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.irl import OUTCOME_FEATURE_NAMES

from irl_sensitivity_analysis import (
    _aggregate_r1,
    _aggregate_r2,
    _aggregate_r3,
    _aggregate_r4,
    w_recovered_for,
)


# --- w_recovered_for --------------------------------------------------------


def _fake_posteriors(seeds=(1, 2), models=("claude", "openai")) -> pd.DataFrame:
    rows = []
    for s in seeds:
        for m in models:
            for k, name in enumerate(OUTCOME_FEATURE_NAMES):
                rows.append({
                    "seed": s, "replica": 0, "model": m, "dim": name,
                    "w_mean": float(s + k * 0.1 + (0.5 if m == "openai" else 0)),
                    "hdi_lo": -1.0, "hdi_hi": +1.0,
                })
    return pd.DataFrame(rows)


def test_w_recovered_for_devuelve_orden_canonico():
    df = _fake_posteriors(seeds=(1,), models=("claude",))
    w = w_recovered_for(df, 1, "claude")
    assert w.shape == (6,)
    # debe estar en el orden de OUTCOME_FEATURE_NAMES
    expected = np.array([1.0 + k * 0.1 for k in range(6)])
    np.testing.assert_allclose(w, expected)


def test_w_recovered_for_distingue_modelos():
    df = _fake_posteriors(seeds=(1,), models=("claude", "openai"))
    w_c = w_recovered_for(df, 1, "claude")
    w_o = w_recovered_for(df, 1, "openai")
    # openai shifted +0.5 en mi fake
    np.testing.assert_allclose(w_o - w_c, np.full(6, 0.5))


def test_w_recovered_for_seed_inexistente_raises():
    df = _fake_posteriors(seeds=(1,), models=("claude",))
    with pytest.raises(KeyError, match="No hay posterior"):
        w_recovered_for(df, 99, "claude")


# --- _aggregate_r1 ----------------------------------------------------------


def test_aggregate_r1_basico():
    df = pd.DataFrame([
        {"model": "a", "seed": 1, "rho": 0.1, "n_perturb": 200, "n_misaligned": 200, "pct_misaligned": 1.0},
        {"model": "a", "seed": 2, "rho": 0.1, "n_perturb": 200, "n_misaligned": 180, "pct_misaligned": 0.9},
        {"model": "a", "seed": 1, "rho": 0.5, "n_perturb": 200, "n_misaligned": 100, "pct_misaligned": 0.5},
    ])
    agg = _aggregate_r1(df)
    assert len(agg) == 2
    row_01 = agg[(agg["model"] == "a") & (agg["rho"] == 0.1)].iloc[0]
    assert row_01["n_seeds"] == 2
    assert row_01["pct_misaligned_max"] == 1.0
    assert row_01["pct_misaligned_min"] == 0.9
    assert row_01["n_seeds_always_misaligned"] == 1


def test_aggregate_r1_vacio():
    df = pd.DataFrame()
    out = _aggregate_r1(df)
    assert out.empty


# --- _aggregate_r2 ----------------------------------------------------------


def test_aggregate_r2_basico():
    df = pd.DataFrame([
        {"model": "a", "seed": 1, "tau": 0.5, "cosine_v1": 0.4, "n_inconsistent_v1": 3, "n_turnos": 8, "flag_v1": 1},
        {"model": "a", "seed": 2, "tau": 0.5, "cosine_v1": 0.6, "n_inconsistent_v1": 1, "n_turnos": 8, "flag_v1": 0},
        {"model": "a", "seed": 1, "tau": 0.7, "cosine_v1": 0.4, "n_inconsistent_v1": 5, "n_turnos": 8, "flag_v1": 1},
    ])
    agg = _aggregate_r2(df)
    row = agg[(agg["model"] == "a") & (agg["tau"] == 0.5)].iloc[0]
    assert row["flag_count"] == 1
    assert row["n_seeds"] == 2


# --- _aggregate_r3 ----------------------------------------------------------


def test_aggregate_r3_basico():
    df = pd.DataFrame([
        {"model": "a", "seed": 1, "n_turnos": 8,
         "cosine_v1": 0.4, "n_inconsistent_v1": 3, "flag_v1": 1,
         "cosine_v2": 0.45, "n_inconsistent_v2": 2, "flag_v2": 1,
         "kappa_per_turn": 0.6, "flags_concur": 1},
        {"model": "a", "seed": 2, "n_turnos": 8,
         "cosine_v1": 0.7, "n_inconsistent_v1": 0, "flag_v1": 0,
         "cosine_v2": 0.65, "n_inconsistent_v2": 1, "flag_v2": 0,
         "kappa_per_turn": 0.4, "flags_concur": 1},
    ])
    agg = _aggregate_r3(df)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["flag_v1_count"] == 1
    assert row["flag_v2_count"] == 1
    assert row["flags_concur_pct"] == 100.0


# --- _aggregate_r4 ----------------------------------------------------------


def test_aggregate_r4_basico():
    df = pd.DataFrame([
        {"model": "a", "seed": 1, "drop_idx": 0, "cosine_dir_K4_vs_K5": 0.95, "n_turnos_kept": 7, "skipped": ""},
        {"model": "a", "seed": 2, "drop_idx": 0, "cosine_dir_K4_vs_K5": 0.92, "n_turnos_kept": 6, "skipped": ""},
        {"model": "a", "seed": 1, "drop_idx": 1, "cosine_dir_K4_vs_K5": float("nan"), "n_turnos_kept": 0, "skipped": "too_few_turns"},
    ])
    agg = _aggregate_r4(df)
    valid = agg[(agg["model"] == "a") & (agg["drop_idx"] == 0)]
    assert len(valid) == 1
    assert valid.iloc[0]["n"] == 2
    assert abs(valid.iloc[0]["median_cos"] - 0.935) < 1e-9


def test_aggregate_r4_solo_nans_devuelve_vacio():
    df = pd.DataFrame([
        {"model": "a", "seed": 1, "drop_idx": 0, "cosine_dir_K4_vs_K5": float("nan"),
         "n_turnos_kept": 0, "skipped": "too_few_turns"},
    ])
    agg = _aggregate_r4(df)
    assert agg.empty
