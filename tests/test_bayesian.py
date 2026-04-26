"""Tests del módulo `bayesian` (BEST + Dirichlet-multinomial).

Usamos pocas muestras y pocos draws para que pasen rápido. Validamos:
recuperación de signo del efecto, recuperación de proporciones del
Dirichlet, y graceful failure cuando PyMC no está instalado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pymc = pytest.importorskip("pymc")  # skip suite si no hay PyMC instalado

from guatemala_sim.bayesian import (
    PRESUPUESTO_PARTIDAS,
    best_paired,
    best_paired_table,
    compare_budget_constitutions,
    constitutions_to_dataframe,
    fit_budget_dirichlet,
)


SAMPLE_KW = dict(draws=400, tune=400, chains=2, seed=11, progressbar=False)


# --- BEST -----------------------------------------------------------------


def test_best_paired_recupera_diferencia_positiva():
    """Si b > a sistemáticamente, el posterior debe estar lejos de 0."""
    rng = np.random.default_rng(42)
    n = 25
    a = rng.normal(loc=10.0, scale=1.0, size=n)
    b = a + rng.normal(loc=2.0, scale=0.4, size=n)  # b consistentemente > a por ~2

    res = best_paired(a, b, metric="test", **SAMPLE_KW)

    assert res.n_pairs == n
    # diferencia media debería estar cerca de 2
    assert 1.5 < res.posterior_diff_mean < 2.5
    # HDI95 debería excluir 0
    assert res.hdi95_lo > 0
    # casi certeza de b > a
    assert res.prob_b_gt_a > 0.99


def test_best_paired_diferencia_nula_no_significativa():
    """Si a y b vienen del mismo proceso, el HDI95 debe contener 0
    (la posterior no rechaza la hipótesis nula con confianza)."""
    rng = np.random.default_rng(7)
    n = 30
    a = rng.normal(loc=5.0, scale=1.0, size=n)
    b = rng.normal(loc=5.0, scale=1.0, size=n)

    res = best_paired(a, b, metric="null", **SAMPLE_KW)

    # HDI95 debe contener 0: el test correcto para "no se detectó efecto".
    # P(b>a) puede leanear lejos de 0.5 por azar muestral, pero el HDI
    # cruzando 0 garantiza que esa preferencia no es estadísticamente firme.
    assert res.hdi95_lo < 0 < res.hdi95_hi
    assert 0 <= res.prob_in_rope <= 1


def test_best_paired_rope_explicito():
    rng = np.random.default_rng(3)
    n = 20
    a = rng.normal(0, 1, n)
    b = rng.normal(0, 1, n)
    res = best_paired(a, b, rope_width=2.0, **SAMPLE_KW)
    assert res.rope_width == 2.0
    # con un ROPE muy ancho, casi toda la masa cae adentro
    assert res.prob_in_rope > 0.5


def test_best_paired_rechaza_shapes_distintas():
    with pytest.raises(ValueError):
        best_paired(np.array([1, 2, 3]), np.array([1, 2]), **SAMPLE_KW)


def test_best_paired_rechaza_n_chico():
    with pytest.raises(ValueError):
        best_paired(np.array([1.0]), np.array([2.0]), **SAMPLE_KW)


def test_best_paired_table_multiples_metricas():
    """`best_paired_table` debe correr BEST para cada métrica del DataFrame
    y devolver una tabla indexada por métrica."""
    rng = np.random.default_rng(1)
    seeds = list(range(8))
    rows = []
    for s in seeds:
        rows.append({"seed": s, "modelo": "A", "m1": rng.normal(0, 1), "m2": rng.normal(5, 1)})
        rows.append({"seed": s, "modelo": "B", "m1": rng.normal(2, 1), "m2": rng.normal(5, 1)})
    df = pd.DataFrame(rows).set_index(["seed", "modelo"])
    out = best_paired_table(df, "A", "B", metrics=["m1", "m2"], **SAMPLE_KW)

    assert "m1" in out.index
    assert "m2" in out.index
    # m1: B > A en ~2; HDI debe incluir un valor positivo claro
    assert out.loc["m1", "diff_mean"] > 1.0
    # m2: misma media; HDI debe ser cercano a 0
    assert abs(out.loc["m2", "diff_mean"]) < 1.0


# --- Dirichlet-multinomial ------------------------------------------------


def test_fit_budget_dirichlet_recupera_proporciones():
    """Generamos T budgets desde un Dirichlet conocido y verificamos que
    el posterior recupera las proporciones."""
    K = len(PRESUPUESTO_PARTIDAS)
    rng = np.random.default_rng(123)
    # constitución verdadera: salud y educación dominan
    alpha_true = np.array([20.0, 18.0, 8.0, 12.0, 6.0, 10.0, 8.0, 5.0, 5.0])
    expected_true = alpha_true / alpha_true.sum()
    obs = rng.dirichlet(alpha_true, size=40)

    post = fit_budget_dirichlet(obs, model_label="LLM-test", **SAMPLE_KW)

    assert post.n_obs == 40
    assert post.expected_share.shape == (K,)
    assert post.alpha_mean.shape == (K,)
    # las proporciones esperadas deben sumar 1
    assert post.expected_share.sum() == pytest.approx(1.0, abs=1e-3)
    # debería recuperar bien las posiciones relativas
    np.testing.assert_allclose(
        post.expected_share, expected_true,
        atol=0.05,  # tolerancia generosa con N=40
    )


def test_fit_budget_dirichlet_hdi_contiene_verdad():
    rng = np.random.default_rng(7)
    alpha_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])  # uniforme
    expected_true = alpha_true / alpha_true.sum()  # 1/9 cada una
    obs = rng.dirichlet(alpha_true, size=30)

    post = fit_budget_dirichlet(obs, model_label="uniforme", **SAMPLE_KW)
    # cada proporción esperada (1/9 ≈ 0.111) debería caer dentro del HDI95
    for k in range(len(PRESUPUESTO_PARTIDAS)):
        lo, hi = post.expected_share_hdi95[k]
        assert lo - 0.02 <= expected_true[k] <= hi + 0.02


def test_fit_budget_dirichlet_rechaza_shape_invalido():
    with pytest.raises(ValueError):
        fit_budget_dirichlet(np.zeros((5, 3)))  # K=3 ≠ 9


def test_fit_budget_dirichlet_rechaza_n_chico():
    with pytest.raises(ValueError):
        fit_budget_dirichlet(np.ones((1, len(PRESUPUESTO_PARTIDAS))) / 9)


def test_compare_budget_constitutions_dos_modelos():
    """Dos LLMs con constituciones distintas → posteriors distintos."""
    rng = np.random.default_rng(33)
    # LLM A prioriza deuda + seguridad
    alpha_a = np.array([8.0, 10.0, 18.0, 8.0, 5.0, 8.0, 22.0, 6.0, 5.0])
    # LLM B prioriza salud + educación + protección social
    alpha_b = np.array([22.0, 20.0, 6.0, 8.0, 5.0, 18.0, 4.0, 6.0, 6.0])

    rows = []
    for t in range(20):
        sample_a = rng.dirichlet(alpha_a)
        sample_b = rng.dirichlet(alpha_b)
        for m, sample in [("A", sample_a), ("B", sample_b)]:
            row = {"seed": 0, "replica": 0, "modelo": m, "t": t}
            for k, partida in enumerate(PRESUPUESTO_PARTIDAS):
                row[f"presup_{partida}"] = float(sample[k]) * 100  # en %
            rows.append(row)
    df_long = pd.DataFrame(rows)

    posts = compare_budget_constitutions(df_long, models=["A", "B"], **SAMPLE_KW)
    assert set(posts.keys()) == {"A", "B"}

    # A debería tener más share en servicio_deuda y seguridad que B
    idx_deuda = PRESUPUESTO_PARTIDAS.index("servicio_deuda")
    idx_salud = PRESUPUESTO_PARTIDAS.index("salud")
    assert posts["A"].expected_share[idx_deuda] > posts["B"].expected_share[idx_deuda]
    assert posts["B"].expected_share[idx_salud] > posts["A"].expected_share[idx_salud]


def test_constitutions_to_dataframe_estructura():
    rng = np.random.default_rng(0)
    obs = rng.dirichlet(np.ones(len(PRESUPUESTO_PARTIDAS)), size=10)
    post = fit_budget_dirichlet(obs, model_label="test", **SAMPLE_KW)
    out = constitutions_to_dataframe({"test": post})
    assert ("test", "salud") in out.index
    assert "E[share]" in out.columns
    assert "hdi95_lo" in out.columns
    assert "hdi95_hi" in out.columns
    # E[share] suma 1 dentro del modelo
    assert out.xs("test", level="modelo")["E[share]"].sum() == pytest.approx(1.0, abs=1e-3)
