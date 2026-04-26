"""Tests offline del módulo `guatemala_sim.multiseed`.

Cubre Capa 1 (correcciones múltiples + tamaños de efecto + power) y
Capa 2 (mixed-effects + ICC con réplicas). Usa `_FixedBudgetMaker` con
presupuesto deterministico para que los tests no dependan de ninguna API.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from guatemala_sim.actions import (
    DecisionTurno,
    Fiscal,
    PoliticaExterior,
    PresupuestoAnual,
)
from guatemala_sim.agents import (
    AgentesModel,
    CACIF,
    CongresoOposicion,
    PartidoOficialista,
    ProtestaSocial,
)
from guatemala_sim.bootstrap import initial_state
from guatemala_sim.engine import run_turn
from guatemala_sim.logging_ import JsonlLogger
from guatemala_sim.multiseed import (
    SeedRun,
    _benjamini_hochberg,
    _holm_bonferroni,
    aggregate_by_model,
    analyze,
    cliffs_delta,
    cohens_d_paired,
    collapse_replicas,
    collect_metrics,
    collect_turn_metrics,
    compute_icc,
    fit_mixed_effects_one,
    paired_tests,
    power_post_hoc_paired,
)
from guatemala_sim.world.territory import Territory


class _FixedBudgetMaker:
    """Decisor con presupuesto custom y resto uniforme.

    Si `jitter` > 0 introduce ruido gaussiano pequeño en cada llamada
    (simula la estocasticidad del sampler de Boltzmann del LLM, útil para
    ICC < 1 en los tests).
    """

    def __init__(
        self,
        salud_pp: float,
        deuda_pp: float,
        alineamiento: str = "multilateral",
        jitter: float = 0.0,
        rng_seed: int = 0,
    ):
        self.salud_pp = salud_pp
        self.deuda_pp = deuda_pp
        self.alineamiento = alineamiento
        self.jitter = jitter
        self._rng = np.random.default_rng(rng_seed)

    def decide(self, state):
        salud = self.salud_pp + self.jitter * float(self._rng.normal())
        deuda = self.deuda_pp + self.jitter * float(self._rng.normal())
        salud = float(np.clip(salud, 0.5, 95.0))
        deuda = float(np.clip(deuda, 0.5, 95.0))
        resto = (100.0 - salud - deuda) / 7.0
        return DecisionTurno(
            razonamiento="fixed test maker",
            presupuesto=PresupuestoAnual(
                salud=salud,
                educacion=resto,
                seguridad=resto,
                infraestructura=resto,
                agro_desarrollo_rural=resto,
                proteccion_social=resto,
                servicio_deuda=deuda,
                justicia=resto,
                otros=resto,
            ),
            fiscal=Fiscal(delta_iva_pp=0.0, delta_isr_pp=0.0),
            exterior=PoliticaExterior(
                alineamiento_priorizado=self.alineamiento,
            ),
            mensaje_al_pueblo="mensaje fijo",
        )


def _corrida(tmp_path: Path, seed: int, label: str, decisor, turnos: int = 4) -> Path:
    rng = np.random.default_rng(seed)
    state = initial_state()
    territory = Territory.load_default()
    agentes = AgentesModel(
        [PartidoOficialista, CongresoOposicion, CACIF, ProtestaSocial], seed=seed
    )
    path = tmp_path / f"{label}.jsonl"
    with JsonlLogger(path) as lg:
        for _ in range(turnos):
            state, _ = run_turn(
                state, decisor, rng, hooks=[lg.log],
                agentes=agentes, territorio=territory,
            )
    return path


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def runs_simple(tmp_path: Path) -> list[SeedRun]:
    """3 seeds × 2 modelos, 1 réplica cada uno (presupuestos opuestos)."""
    runs: list[SeedRun] = []
    for s in [1, 2, 3]:
        runs.append(SeedRun(seed=s, model_label="ModelA", log_path=_corrida(
            tmp_path, s, f"a_s{s}", _FixedBudgetMaker(25.0, 5.0))))
        runs.append(SeedRun(seed=s, model_label="ModelB", log_path=_corrida(
            tmp_path, s, f"b_s{s}", _FixedBudgetMaker(5.0, 25.0))))
    return runs


@pytest.fixture
def runs_with_replicas(tmp_path: Path) -> list[SeedRun]:
    """3 seeds × 2 modelos × 3 réplicas (con jitter para ICC < 1)."""
    runs: list[SeedRun] = []
    for s in [1, 2, 3]:
        for rep in range(3):
            runs.append(SeedRun(seed=s, model_label="ModelA",
                                replica=rep,
                                log_path=_corrida(
                tmp_path, s, f"a_s{s}_r{rep}",
                _FixedBudgetMaker(25.0, 5.0, jitter=1.5, rng_seed=100*s+rep))))
            runs.append(SeedRun(seed=s, model_label="ModelB",
                                replica=rep,
                                log_path=_corrida(
                tmp_path, s, f"b_s{s}_r{rep}",
                _FixedBudgetMaker(5.0, 25.0, jitter=1.5, rng_seed=200*s+rep))))
    return runs


# --- recolección -----------------------------------------------------------


def test_collect_metrics_indice_triple(runs_simple):
    df = collect_metrics(runs_simple)
    assert df.index.names == ["seed", "replica", "modelo"]
    assert len(df) == 6
    assert "presup_salud" in df.columns


def test_collapse_replicas_promedia(runs_with_replicas):
    df = collect_metrics(runs_with_replicas)
    assert len(df) == 18  # 3 seeds × 2 modelos × 3 réplicas
    collapsed = collapse_replicas(df)
    assert collapsed.index.names == ["seed", "modelo"]
    assert len(collapsed) == 6


def test_collect_turn_metrics_long_format(runs_simple):
    df_long = collect_turn_metrics(runs_simple)
    # 6 corridas × 4 turnos = 24 filas
    assert len(df_long) == 24
    for col in ("seed", "replica", "modelo", "t",
                "pib_usd_mm", "pobreza_general",
                "presup_salud", "ind_bienestar"):
        assert col in df_long.columns


# --- agregación ------------------------------------------------------------


def test_aggregate_recupera_diferencia_conocida(runs_simple):
    df = collect_metrics(runs_simple)
    agg = aggregate_by_model(df, n_boot=500)
    assert {"mean", "std", "ic95_lo", "ic95_hi", "n"} <= set(agg.columns)
    a_salud = agg.loc[("ModelA", "presup_salud"), "mean"]
    b_salud = agg.loc[("ModelB", "presup_salud"), "mean"]
    assert a_salud > 20.0 and b_salud < 10.0


# --- correcciones múltiples (Capa 1) --------------------------------------


def test_holm_bonferroni_basico():
    p = np.array([0.001, 0.01, 0.04, 0.50])
    adj = _holm_bonferroni(p)
    # ordenado ascendente: 0.001*4=0.004, 0.01*3=0.03, 0.04*2=0.08, 0.50*1=0.50
    assert adj[0] == pytest.approx(0.004)
    assert adj[1] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.08)
    assert adj[3] == pytest.approx(0.50)


def test_bh_fdr_basico():
    p = np.array([0.001, 0.01, 0.04, 0.50])
    adj = _benjamini_hochberg(p)
    # m=4: rango i devuelve p*m/i, con monotonización descendente
    # p=0.50 (i=4): 0.50; p=0.04 (i=3): 0.04*4/3 ≈ 0.0533; p=0.01 (i=2): 0.02;
    # p=0.001 (i=1): 0.004 → min(0.004, 0.02) = 0.004
    assert adj[0] == pytest.approx(0.004, rel=0.01)
    assert adj[3] == pytest.approx(0.50, rel=0.01)
    # debe ser monótono no-decreciente cuando p está ordenado
    p_sorted = np.sort(p)
    adj_sorted = _benjamini_hochberg(p_sorted)
    assert all(adj_sorted[i] <= adj_sorted[i+1] + 1e-9 for i in range(len(adj_sorted)-1))


def test_corrections_handle_nan():
    p = np.array([0.001, np.nan, 0.04])
    h = _holm_bonferroni(p)
    bh = _benjamini_hochberg(p)
    assert np.isnan(h[1]) and np.isnan(bh[1])
    assert not np.isnan(h[0]) and not np.isnan(bh[0])


# --- tamaños de efecto -----------------------------------------------------


def test_cohens_d_paired_signo_y_magnitud():
    a = np.array([10, 11, 12, 13, 14], dtype=float)
    b = np.array([5, 6, 7, 8, 9], dtype=float)
    d = cohens_d_paired(a, b)
    # diferencia constante de 5, std 0 → debería ser inf, pero ddof=1 sobre [5,5,5,5,5] da 0
    # → nan. Cubre el edge case de "perfectamente correlacionado".
    assert np.isnan(d)
    # ahora una diferencia con varianza
    a2 = np.array([10, 12, 14, 16, 18], dtype=float)
    b2 = np.array([5, 8, 7, 11, 9], dtype=float)
    d2 = cohens_d_paired(a2, b2)
    assert d2 > 0  # a > b en promedio
    assert not np.isnan(d2)


def test_cliffs_delta_extremos():
    a = np.array([10, 11, 12])
    b = np.array([1, 2, 3])
    assert cliffs_delta(a, b) == pytest.approx(1.0)  # a domina completamente
    assert cliffs_delta(b, a) == pytest.approx(-1.0)
    # solapamiento total
    c = np.array([1, 2, 3])
    assert cliffs_delta(c, c) == pytest.approx(0.0)


def test_power_post_hoc_creciente_en_n_y_d():
    # mayor d → mayor power
    p_chico = power_post_hoc_paired(d=0.2, n=10)
    p_grande = power_post_hoc_paired(d=0.8, n=10)
    assert p_grande > p_chico
    # mayor N → mayor power
    p_n10 = power_post_hoc_paired(d=0.5, n=10)
    p_n50 = power_post_hoc_paired(d=0.5, n=50)
    assert p_n50 > p_n10
    # power ∈ [0, 1]
    assert 0.0 <= p_n10 <= 1.0


# --- tests pareados con correcciones --------------------------------------


def test_paired_tests_columnas_completas(tmp_path: Path):
    runs = []
    for s in range(1, 9):
        runs.append(SeedRun(seed=s, model_label="A", log_path=_corrida(
            tmp_path, s, f"a_{s}", _FixedBudgetMaker(25.0, 5.0))))
        runs.append(SeedRun(seed=s, model_label="B", log_path=_corrida(
            tmp_path, s, f"b_{s}", _FixedBudgetMaker(5.0, 25.0))))
    df = collect_metrics(runs)
    tests = paired_tests(df, "A", "B")
    for col in ("p_value", "p_holm", "p_bh", "cohens_d", "cliffs_delta",
                "rank_biserial", "power_post_hoc", "sig", "sig_bh"):
        assert col in tests.columns
    # p_holm y p_bh nunca deben ser menores que p_value crudo
    valid = ~tests["p_value"].isna()
    assert (tests.loc[valid, "p_holm"] >= tests.loc[valid, "p_value"] - 1e-9).all()
    assert (tests.loc[valid, "p_bh"] >= tests.loc[valid, "p_value"] - 1e-9).all()


def test_paired_tests_significativo_sobrevive_correccion(tmp_path: Path):
    """Con presupuestos opuestos extremos, la diferencia en presup_salud debe
    sobrevivir la corrección BH-FDR."""
    runs = []
    for s in range(1, 11):
        runs.append(SeedRun(seed=s, model_label="A", log_path=_corrida(
            tmp_path, s, f"a_{s}", _FixedBudgetMaker(25.0, 5.0))))
        runs.append(SeedRun(seed=s, model_label="B", log_path=_corrida(
            tmp_path, s, f"b_{s}", _FixedBudgetMaker(5.0, 25.0))))
    df = collect_metrics(runs)
    tests = paired_tests(df, "A", "B")
    p_bh_salud = tests.loc["presup_salud", "p_bh"]
    assert not pd.isna(p_bh_salud) and p_bh_salud < 0.05


# --- mixed-effects (Capa 2) -----------------------------------------------


def test_fit_mixed_effects_recupera_efecto_conocido(tmp_path: Path):
    """Con heterogeneidad seed-a-seed (offset variable) y jitter intra-corrida,
    el MixedLM tiene un efecto aleatorio identificable y debe recuperar el
    efecto verdadero `B − A ≈ -20pp` en `presup_salud`."""
    runs = []
    for s in range(1, 11):
        # cada seed comparte un offset entre A y B (eso produce var(seed) > 0
        # y al mismo tiempo deja la diferencia A−B en ≈ 20pp)
        offset = (s - 5) * 0.6
        runs.append(SeedRun(seed=s, model_label="A", log_path=_corrida(
            tmp_path, s, f"a_{s}",
            _FixedBudgetMaker(25.0 + offset, 5.0,
                              jitter=0.8, rng_seed=10*s))))
        runs.append(SeedRun(seed=s, model_label="B", log_path=_corrida(
            tmp_path, s, f"b_{s}",
            _FixedBudgetMaker(5.0 + offset, 25.0,
                              jitter=0.8, rng_seed=20*s))))
    df_long = collect_turn_metrics(runs)
    res = fit_mixed_effects_one(df_long, "presup_salud", "A", "B")
    assert not np.isnan(res["fixed_effect_b_minus_a"]), \
        f"mixed-effects no convergió: {res.get('error', '')}"
    assert res["fixed_effect_b_minus_a"] == pytest.approx(-20.0, abs=2.0)
    assert res["ci95_lo"] <= -18.0
    assert res["ci95_hi"] >= -22.0
    assert res["p_value"] < 0.001
    # n_obs = 10 seeds × 2 modelos × 4 turnos = 80
    assert res["n_obs"] == 80
    assert res["n_seeds"] == 10


# --- ICC (test-retest, Capa 2) --------------------------------------------


def test_icc_sin_replicas_devuelve_error(runs_simple):
    df_long = collect_turn_metrics(runs_simple)
    res = compute_icc(df_long, "presup_salud", "ModelA")
    assert np.isnan(res["icc"])
    assert "réplica" in res.get("error", "") or "replica" in res.get("error", "")


def test_icc_con_replicas_es_numerico(runs_with_replicas):
    df_long = collect_turn_metrics(runs_with_replicas)
    res = compute_icc(df_long, "presup_salud", "ModelA")
    # con jitter chico relativo a la diferencia entre seeds (que es 0 acá:
    # todas las seeds tienen el mismo presupuesto medio porque el jitter es
    # ruido), el ICC puede ser cualquier valor en [0,1]; sólo verificamos
    # que se computa sin error.
    assert "icc" in res
    if not np.isnan(res["icc"]):
        assert 0.0 <= res["icc"] <= 1.0
        assert res["n_seeds"] == 3


# --- pipeline completo ----------------------------------------------------


def test_analyze_simple_produce_archivos(runs_simple, tmp_path: Path):
    out_dir = tmp_path / "analysis"
    paths = analyze(runs_simple, out_dir, model_a="ModelA", model_b="ModelB")
    for k in ("summary", "metrics_per_seed", "aggregate",
              "turn_metrics_long",
              "presupuesto_plot", "outcomes_plot"):
        assert paths[k].exists() and paths[k].stat().st_size > 0
    md = paths["summary"].read_text(encoding="utf-8")
    assert "ModelA" in md and "ModelB" in md
    assert "IC95" in md


def test_analyze_con_replicas_incluye_icc(runs_with_replicas, tmp_path: Path):
    out_dir = tmp_path / "analysis_replicas"
    paths = analyze(runs_with_replicas, out_dir,
                    model_a="ModelA", model_b="ModelB")
    # ICC sólo aparece cuando hay réplicas
    assert "icc" in paths
    assert paths["icc"].exists() and paths["icc"].stat().st_size > 0
    md = paths["summary"].read_text(encoding="utf-8")
    assert "ICC" in md
    assert "Boltzmann" in md or "sampler" in md
