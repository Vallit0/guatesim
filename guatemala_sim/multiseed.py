"""Análisis multi-seed: agregación, IC95 bootstrap, tests pareados,
correcciones múltiples, tamaños de efecto, mixed-effects e ICC.

Cada "corrida" sigue siendo un JSONL producido por `JsonlLogger`. Acá tomamos
una grilla `(seed, réplica, modelo) → ruta_jsonl`, computamos métricas y
reportamos:

  * tabla agregada por modelo: media, std, IC95 bootstrap por métrica.
  * tests pareados Wilcoxon (signed-rank) entre dos modelos, con p-values
    crudos y corregidos (Holm-Bonferroni y Benjamini-Hochberg FDR).
  * tamaños de efecto: rank-biserial, Cohen's d (paramétrico), Cliff's δ
    (no-paramétrico independiente de N).
  * power post-hoc dado el efecto observado y N.
  * mixed-effects (statsmodels) sobre datos turn-level: `metric ~ modelo +
    (1|seed)` para aprovechar las 8×N observaciones en vez de colapsar a N.
  * ICC (intraclass correlation) sobre réplicas dentro de modelo: cuánto de
    la varianza es entre-seed (sustantivo) vs. dentro-seed (sampler del LLM).
  * plots con error bars + boxplots + reporte markdown.

Las métricas escalares por corrida (columnas del DataFrame) son las mismas
que devuelve `comparison.tabla_comparativa`. Las turn-level se generan acá
recorriendo cada record JSONL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .comparison import CorridaEtiquetada, tabla_comparativa
from .logging_ import read_run


# --- recolección ------------------------------------------------------------


@dataclass
class SeedRun:
    seed: int
    model_label: str
    log_path: Path
    replica: int = 0  # 0 = corrida única; >0 si hay test-retest


def collect_metrics(runs: Sequence[SeedRun]) -> pd.DataFrame:
    """DataFrame con multi-índice (seed, replica, modelo) → métricas escalares
    de fin-de-horizonte (las mismas que `tabla_comparativa`).
    """
    rows: list[pd.DataFrame] = []
    for r in runs:
        c = CorridaEtiquetada.from_path(r.model_label, r.log_path)
        df = tabla_comparativa([c]).reset_index()
        df.insert(0, "seed", r.seed)
        df.insert(1, "replica", r.replica)
        df = df.rename(columns={"corrida": "modelo"})
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out.set_index(["seed", "replica", "modelo"]).sort_index()
    return out


def collapse_replicas(df: pd.DataFrame) -> pd.DataFrame:
    """Promedia sobre réplicas dentro de cada (seed, modelo).

    Útil para tests pareados que quieren un valor por par. Si no hay
    réplicas (`replica` único = 0), es identidad.
    """
    numeric = df.select_dtypes(include=[np.number])
    return numeric.groupby(level=["seed", "modelo"]).mean()


def collect_turn_metrics(runs: Sequence[SeedRun]) -> pd.DataFrame:
    """Long-format turn-level: una fila por (seed, replica, modelo, t).

    Aprovecha las 8 × N observaciones que `collect_metrics` colapsa a N.
    Crítico para los modelos de efectos mixtos.
    """
    rows: list[dict] = []
    for r in runs:
        records = read_run(Path(r.log_path))
        for rec in records:
            sa = rec["state_after"]
            dec = rec["decision"]
            indicadores = rec.get("indicadores", {})
            row: dict = {
                "seed": r.seed,
                "replica": r.replica,
                "modelo": r.model_label,
                "t": rec["t"],
                "n_shocks": len(rec.get("shocks", []) or []),
                # outcomes macro/social
                "pib_usd_mm": sa["macro"]["pib_usd_mm"],
                "crecimiento_pib": sa["macro"]["crecimiento_pib"],
                "inflacion": sa["macro"]["inflacion"],
                "deuda_pib": sa["macro"]["deuda_pib"],
                "balance_fiscal_pib": sa["macro"]["balance_fiscal_pib"],
                "pobreza_general": sa["social"]["pobreza_general"],
                "homicidios_100k": sa["social"]["homicidios_100k"],
                "aprobacion_presidencial": sa["politico"]["aprobacion_presidencial"],
                "indice_protesta": sa["politico"]["indice_protesta"],
                "confianza_institucional": sa["politico"]["confianza_institucional"],
                # decisión: presupuesto turn-by-turn
                **{f"presup_{k}": float(v) for k, v in dec["presupuesto"].items()},
                "delta_iva_pp": dec["fiscal"]["delta_iva_pp"],
                "delta_isr_pp": dec["fiscal"]["delta_isr_pp"],
                "n_reformas": len(dec.get("reformas", []) or []),
                # índices compuestos
                **{f"ind_{k}": float(v) for k, v in indicadores.items() if k != "t"},
            }
            rows.append(row)
    return pd.DataFrame(rows)


# --- agregación ------------------------------------------------------------


def _bootstrap_ic95(values: np.ndarray, n_boot: int = 5_000, rng=None) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = rng or np.random.default_rng(0)
    boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate_by_model(df: pd.DataFrame, n_boot: int = 5_000) -> pd.DataFrame:
    """Media, std, n, IC95 bootstrap por (modelo × métrica).

    Acepta DataFrame con índice (seed, modelo) o (seed, replica, modelo).
    Si hay réplicas, primero las colapsa con `collapse_replicas`.
    """
    if "replica" in df.index.names and df.index.get_level_values("replica").nunique() > 1:
        df = collapse_replicas(df)
    elif "replica" in df.index.names:
        df = df.droplevel("replica")

    rng = np.random.default_rng(0)
    rows: list[dict] = []
    numeric = df.select_dtypes(include=[np.number])
    for modelo, g in numeric.groupby(level="modelo"):
        for col in g.columns:
            v = g[col].to_numpy()
            mean = float(np.nanmean(v)) if len(v) else float("nan")
            std = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
            lo, hi = _bootstrap_ic95(v, n_boot=n_boot, rng=rng)
            rows.append({
                "modelo": modelo, "metrica": col,
                "n": int(np.sum(~np.isnan(v))),
                "mean": mean, "std": std,
                "ic95_lo": lo, "ic95_hi": hi,
            })
    return pd.DataFrame(rows).set_index(["modelo", "metrica"]).sort_index()


# --- tamaños de efecto ----------------------------------------------------


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d para datos pareados: mean(a-b) / sd(a-b).

    Convención de magnitud (Cohen 1988): |d| < 0.2 trivial, 0.2–0.5 chico,
    0.5–0.8 medio, > 0.8 grande.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(d) / sd)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ no-paramétrico: P(a > b) − P(a < b), rango [-1, 1].

    Convención (Romano et al. 2006): |δ| < 0.147 trivial, 0.147–0.33 chico,
    0.33–0.474 medio, > 0.474 grande.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n_gt = float(np.sum(a[:, None] > b[None, :]))
    n_lt = float(np.sum(a[:, None] < b[None, :]))
    return (n_gt - n_lt) / (len(a) * len(b))


def power_post_hoc_paired(d: float, n: int, alpha: float = 0.05) -> float:
    """Potencia post-hoc de un t-test pareado dado Cohen's d y N de pares."""
    if n < 2 or np.isnan(d):
        return float("nan")
    df = n - 1
    nc = d * np.sqrt(n)  # parámetro de no-centralidad
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = (
        1.0
        - stats.nct.cdf(t_crit, df, nc)
        + stats.nct.cdf(-t_crit, df, nc)
    )
    return float(power)


# --- tests pareados con correcciones --------------------------------------


def _holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down. Devuelve p-values ajustados."""
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return p
    idx_valid = np.where(valid)[0]
    sub = p[valid]
    order = np.argsort(sub)
    m = len(sub)
    adj = np.empty_like(sub)
    running = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * sub[i]
        running = max(running, v)
        adj[i] = min(running, 1.0)
    out = np.full_like(p, np.nan)
    out[idx_valid] = adj
    return out


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR. Devuelve p-values ajustados (q-values)."""
    p = np.asarray(pvals, dtype=float)
    valid = ~np.isnan(p)
    if valid.sum() == 0:
        return p
    idx_valid = np.where(valid)[0]
    sub = p[valid]
    order = np.argsort(sub)
    m = len(sub)
    adj = np.empty_like(sub)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        v = sub[i] * m / (rank + 1)
        running = min(running, v)
        adj[i] = min(running, 1.0)
    out = np.full_like(p, np.nan)
    out[idx_valid] = adj
    return out


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def paired_tests(df: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Tests pareados Wilcoxon + correcciones múltiples + tamaños de efecto.

    Si `df` tiene réplicas, las colapsa primero (media por seed, modelo).

    Columnas:
      n_pares, median_diff, p_value, p_holm, p_bh,
      rank_biserial, cohens_d, cliffs_delta, power_post_hoc,
      sig (sobre p crudo), sig_bh (sobre BH-FDR).
    """
    if "replica" in df.index.names:
        df = collapse_replicas(df)

    a = df.xs(model_a, level="modelo")
    b = df.xs(model_b, level="modelo")
    seeds_comunes = a.index.intersection(b.index)
    if len(seeds_comunes) == 0:
        return pd.DataFrame()
    a = a.loc[seeds_comunes].select_dtypes(include=[np.number])
    b = b.loc[seeds_comunes].select_dtypes(include=[np.number])

    rows: list[dict] = []
    for col in a.columns:
        va = a[col].to_numpy(dtype=float)
        vb = b[col].to_numpy(dtype=float)
        mask = ~(np.isnan(va) | np.isnan(vb))
        va, vb = va[mask], vb[mask]
        n = len(va)
        if n < 2 or np.allclose(va, vb):
            rows.append({
                "metrica": col, "n_pares": n,
                "median_diff": float(np.median(va - vb)) if n else float("nan"),
                "p_value": float("nan"),
                "rank_biserial": float("nan"),
                "cohens_d": float("nan"),
                "cliffs_delta": float("nan"),
                "power_post_hoc": float("nan"),
            })
            continue
        try:
            res = stats.wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided")
            p = float(res.pvalue)
            diffs = va - vb
            ranks = stats.rankdata(np.abs(diffs[diffs != 0]))
            signs = np.sign(diffs[diffs != 0])
            if len(ranks) > 0:
                W_pos = float(np.sum(ranks[signs > 0]))
                W_neg = float(np.sum(ranks[signs < 0]))
                r_rb = (W_pos - W_neg) / (W_pos + W_neg) if (W_pos + W_neg) > 0 else 0.0
            else:
                r_rb = 0.0
        except Exception:
            p = float("nan")
            r_rb = float("nan")
        d = cohens_d_paired(va, vb)
        delta = cliffs_delta(va, vb)
        power = power_post_hoc_paired(d, n) if not np.isnan(d) else float("nan")
        rows.append({
            "metrica": col, "n_pares": n,
            "median_diff": float(np.median(va - vb)),
            "p_value": p,
            "rank_biserial": float(r_rb),
            "cohens_d": d,
            "cliffs_delta": delta,
            "power_post_hoc": power,
        })

    out = pd.DataFrame(rows).set_index("metrica")
    p_raw = out["p_value"].to_numpy()
    out["p_holm"] = _holm_bonferroni(p_raw)
    out["p_bh"] = _benjamini_hochberg(p_raw)
    out["sig"] = out["p_value"].map(_stars)
    out["sig_bh"] = out["p_bh"].map(_stars)
    return out


# --- mixed effects (Capa 2) -----------------------------------------------


def fit_mixed_effects_one(
    df_long: pd.DataFrame, metric: str, model_a: str, model_b: str,
) -> dict:
    """Ajusta `metric ~ modelo_b + (1|seed)` sobre los datos turn-level.

    `modelo_b` es indicadora de pertenecer al modelo B (vs. A). Con N seeds
    × 8 turnos = 8N observaciones por modelo, en vez de las N del análisis
    end-of-horizon. Captura correctamente la correlación intra-seed.
    """
    import statsmodels.formula.api as smf

    sub = df_long[df_long["modelo"].isin([model_a, model_b])].copy()
    if sub.empty:
        return {"error": "sin filas para esos modelos"}
    sub["modelo_b"] = (sub["modelo"] == model_b).astype(int)
    if sub["seed"].nunique() < 2:
        return {"error": "necesita ≥ 2 seeds para efecto aleatorio"}

    formula = f"Q('{metric}') ~ modelo_b"
    try:
        md = smf.mixedlm(formula, sub, groups=sub["seed"])
        result = md.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
        coef = float(result.fe_params.get("modelo_b", float("nan")))
        se = float(result.bse_fe.get("modelo_b", float("nan")))
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        p = float(result.pvalues.get("modelo_b", float("nan")))
        return {
            "fixed_effect_b_minus_a": coef,
            "se": se,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "p_value": p,
            "n_obs": int(len(sub)),
            "n_seeds": int(sub["seed"].nunique()),
        }
    except Exception as e:
        return {
            "fixed_effect_b_minus_a": float("nan"),
            "se": float("nan"),
            "ci95_lo": float("nan"),
            "ci95_hi": float("nan"),
            "p_value": float("nan"),
            "n_obs": int(len(sub)),
            "n_seeds": int(sub["seed"].nunique()),
            "error": str(e)[:200],
        }


def fit_mixed_effects_all(
    df_long: pd.DataFrame, metrics: Sequence[str], model_a: str, model_b: str,
) -> pd.DataFrame:
    """Itera mixed-effects sobre una lista de métricas. Aplica BH-FDR a los
    p-values del efecto fijo."""
    rows: list[dict] = []
    for m in metrics:
        if m not in df_long.columns:
            continue
        r = fit_mixed_effects_one(df_long, m, model_a, model_b)
        r["metric"] = m
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("metric")
    p_raw = out["p_value"].to_numpy()
    out["p_bh"] = _benjamini_hochberg(p_raw)
    out["sig_bh"] = out["p_bh"].map(_stars)
    return out


# --- ICC sobre réplicas (test-retest) -------------------------------------


def compute_icc(df: pd.DataFrame, metric: str, model: str) -> dict:
    """ICC(1) para `metric` dentro de un modelo, usando réplicas como
    repeticiones del mismo seed.

    Modelo: `metric ~ 1 + (1 | seed)` ajustado sólo a las filas de `model`.
    ICC = var(seed) / (var(seed) + var(residual)).

    Interpretación: ICC alto → la varianza está dominada por diferencias
    entre seeds (sustantivo, mismos shocks → misma respuesta). ICC bajo →
    réplicas del mismo seed difieren tanto como seeds distintos: el sampler
    de Boltzmann del LLM domina, y las "diferencias entre modelos" son
    sospechosas.
    """
    import statsmodels.formula.api as smf

    sub = df[df["modelo"] == model].copy()
    n_replicas = sub.groupby("seed")["replica"].nunique()
    if (n_replicas < 2).all():
        return {
            "modelo": model, "metric": metric,
            "icc": float("nan"),
            "var_seed": float("nan"),
            "var_resid": float("nan"),
            "n_obs": int(len(sub)),
            "n_seeds": int(sub["seed"].nunique()) if len(sub) else 0,
            "error": "sin réplicas múltiples (todas las seeds tienen 1 réplica)",
        }
    if metric not in sub.columns:
        return {"modelo": model, "metric": metric, "icc": float("nan"),
                "error": f"metric '{metric}' no presente"}
    formula = f"Q('{metric}') ~ 1"
    try:
        md = smf.mixedlm(formula, sub, groups=sub["seed"])
        result = md.fit(reml=True, method="lbfgs", maxiter=200, disp=False)
        var_seed = float(result.cov_re.iloc[0, 0])
        var_resid = float(result.scale)
        denom = var_seed + var_resid
        icc = (var_seed / denom) if denom > 0 else float("nan")
        return {
            "modelo": model, "metric": metric,
            "icc": icc,
            "var_seed": var_seed,
            "var_resid": var_resid,
            "n_obs": int(len(sub)),
            "n_seeds": int(sub["seed"].nunique()),
        }
    except Exception as e:
        return {
            "modelo": model, "metric": metric,
            "icc": float("nan"),
            "var_seed": float("nan"),
            "var_resid": float("nan"),
            "n_obs": int(len(sub)),
            "n_seeds": int(sub["seed"].nunique()),
            "error": str(e)[:200],
        }


def compute_icc_all(
    df_long: pd.DataFrame, metrics: Sequence[str], modelos: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for modelo in modelos:
        for m in metrics:
            if m in df_long.columns:
                rows.append(compute_icc(df_long, m, modelo))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["modelo", "metric"])


# --- plots -----------------------------------------------------------------


_COLORES = {"Claude": "#d97757", "OpenAI": "#10a37f"}
_FALLBACK = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _color_for(label: str, idx: int) -> str:
    for k, v in _COLORES.items():
        if k.lower() in label.lower():
            return v
    return _FALLBACK[idx % len(_FALLBACK)]


def plot_budget_with_ci(df: pd.DataFrame, agg: pd.DataFrame, out_path: Path) -> Path:
    """Bar chart del presupuesto promedio por partida con IC95 errorbars."""
    if "replica" in df.index.names:
        df = collapse_replicas(df)
    partidas = [c for c in df.columns if c.startswith("presup_")]
    nombres = [p.replace("presup_", "") for p in partidas]
    modelos = sorted(df.index.get_level_values("modelo").unique())
    x = np.arange(len(partidas))
    n_mod = len(modelos)
    w = 0.8 / n_mod

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, modelo in enumerate(modelos):
        means, errs_lo, errs_hi = [], [], []
        for p in partidas:
            try:
                row = agg.loc[(modelo, p)]
                means.append(row["mean"])
                errs_lo.append(row["mean"] - row["ic95_lo"])
                errs_hi.append(row["ic95_hi"] - row["mean"])
            except KeyError:
                means.append(float("nan"))
                errs_lo.append(0.0)
                errs_hi.append(0.0)
        offset = (i - (n_mod - 1) / 2) * w
        ax.bar(
            x + offset, means, w,
            yerr=[errs_lo, errs_hi],
            capsize=3, color=_color_for(modelo, i),
            label=modelo, edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=30, ha="right")
    ax.set_ylabel("% del presupuesto (promedio sobre turnos y seeds)")
    ax.set_title("Presupuesto revelado: media ± IC95 (bootstrap, multi-seed)",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_outcomes_box(df: pd.DataFrame, out_path: Path) -> Path:
    if "replica" in df.index.names:
        df = collapse_replicas(df)
    metricas = [
        ("PIB_delta", "Δ PIB"),
        ("pobreza_fin", "pobreza final (%)"),
        ("aprobacion_fin", "aprobación final"),
        ("deuda_fin", "deuda/PIB final"),
        ("bienestar_fin", "bienestar (0-100)"),
        ("gobernabilidad_fin", "gobernabilidad (0-100)"),
    ]
    modelos = sorted(df.index.get_level_values("modelo").unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, (col, titulo) in zip(axes, metricas):
        data = [df.xs(m, level="modelo")[col].dropna().to_numpy() for m in modelos]
        bp = ax.boxplot(data, tick_labels=modelos, patch_artist=True, widths=0.55)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_color_for(modelos[i], i))
            patch.set_alpha(0.55)
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Outcomes por modelo (multi-seed)", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_mixed_effects_forest(me_table: pd.DataFrame, out_path: Path,
                              top_n: int | None = 20) -> Path:
    """Forest plot del efecto fijo (B − A) ± IC95 por métrica.

    Ordena por magnitud absoluta del efecto y opcionalmente recorta a top_n.
    """
    df = me_table.copy().dropna(subset=["fixed_effect_b_minus_a"])
    df["abs_eff"] = df["fixed_effect_b_minus_a"].abs()
    df = df.sort_values("abs_eff", ascending=True)
    if top_n is not None:
        df = df.tail(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(df))))
    y = np.arange(len(df))
    eff = df["fixed_effect_b_minus_a"].to_numpy()
    lo = df["ci95_lo"].to_numpy()
    hi = df["ci95_hi"].to_numpy()
    sig = df["sig_bh"].to_numpy()
    colors = ["#d62728" if e < 0 else "#1f77b4" for e in eff]
    ax.errorbar(eff, y, xerr=[eff - lo, hi - eff], fmt="o",
                ecolor="gray", elinewidth=1.5, capsize=3,
                markersize=7,
                markerfacecolor="white", markeredgewidth=1.5)
    for i, (e, s, c) in enumerate(zip(eff, sig, colors)):
        ax.plot(e, i, "o", color=c, markersize=7)
        if s:
            ax.text(hi[i] + 0.02 * (max(hi) - min(lo) + 1e-9), i, s,
                    va="center", fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Efecto fijo  (Modelo B − Modelo A)")
    ax.set_title("Mixed-effects: efecto del modelo por métrica\n"
                 "errorbar = IC95;  *=BH-FDR<0.05  **<0.01  ***<0.001",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# --- reporte ---------------------------------------------------------------


def write_report(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    tests: pd.DataFrame,
    me_table: pd.DataFrame | None,
    icc_table: pd.DataFrame | None,
    out_dir: Path,
    model_a: str = "Claude",
    model_b: str = "OpenAI",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = sorted(df.index.get_level_values("seed").unique())
    modelos = sorted(df.index.get_level_values("modelo").unique())
    has_replicas = (
        "replica" in df.index.names
        and df.index.get_level_values("replica").nunique() > 1
    )
    n_replicas = (
        df.index.get_level_values("replica").nunique()
        if "replica" in df.index.names else 1
    )

    md_path = out_dir / "summary.md"
    lines: list[str] = []
    lines.append("# Multi-seed: comparativa Anthropic vs. OpenAI")
    lines.append("")
    lines.append(f"- **Seeds**: {len(seeds)} ({seeds[0]}–{seeds[-1]})")
    lines.append(f"- **Modelos**: {', '.join(modelos)}")
    lines.append(f"- **Réplicas por (seed, modelo)**: {n_replicas}")
    lines.append("")

    # 1. Outcomes con IC95
    lines.append("## 1. Outcomes — media ± IC95 (bootstrap N=5000)")
    lines.append("")
    metricas_outcomes = [
        "PIB_delta", "pobreza_fin", "aprobacion_fin", "deuda_fin",
        "bienestar_fin", "gobernabilidad_fin", "estabilidad_fin",
        "idh_fin", "estres_fin",
    ]
    rows = []
    for m in metricas_outcomes:
        row: dict = {"métrica": m}
        for modelo in modelos:
            try:
                r = agg.loc[(modelo, m)]
                row[modelo] = f"{r['mean']:.2f} [{r['ic95_lo']:.2f}, {r['ic95_hi']:.2f}]"
            except KeyError:
                row[modelo] = "—"
        rows.append(row)
    lines.append(pd.DataFrame(rows).set_index("métrica").to_markdown())
    lines.append("")

    # 2. Constitución revelada
    lines.append("## 2. Métricas constitucionales — media ± IC95")
    lines.append("")
    metricas_const = [
        "coherencia_temporal", "diversidad_valores",
        "reformas_totales", "reformas_radicales",
        "delta_iva_medio", "delta_isr_medio",
    ]
    rows = []
    for m in metricas_const:
        row = {"métrica": m}
        for modelo in modelos:
            try:
                r = agg.loc[(modelo, m)]
                row[modelo] = f"{r['mean']:.2f} [{r['ic95_lo']:.2f}, {r['ic95_hi']:.2f}]"
            except KeyError:
                row[modelo] = "—"
        rows.append(row)
    lines.append(pd.DataFrame(rows).set_index("métrica").to_markdown())
    lines.append("")

    # 3. Presupuesto revelado
    lines.append("## 3. Presupuesto revelado por partida — media ± IC95 (%)")
    lines.append("")
    df_collapsed = collapse_replicas(df) if "replica" in df.index.names else df
    partidas = [c for c in df_collapsed.columns if c.startswith("presup_")]
    rows = []
    for p in partidas:
        row = {"partida": p.replace("presup_", "")}
        for modelo in modelos:
            try:
                r = agg.loc[(modelo, p)]
                row[modelo] = f"{r['mean']:.2f} [{r['ic95_lo']:.2f}, {r['ic95_hi']:.2f}]"
            except KeyError:
                row[modelo] = "—"
        rows.append(row)
    lines.append(pd.DataFrame(rows).set_index("partida").to_markdown())
    lines.append("")

    # 4. Tests pareados con correcciones
    if not tests.empty:
        lines.append(
            f"## 4. Tests pareados Wilcoxon: {model_a} vs. {model_b}"
        )
        lines.append("")
        lines.append(
            f"Pares por seed (mismos shocks → comparación válida). "
            f"`median_diff` = mediana({model_a} − {model_b}). "
            f"`p_holm` y `p_bh` son p-values corregidos por comparaciones "
            f"múltiples (Holm-Bonferroni y Benjamini-Hochberg FDR). "
            f"`sig_bh` marca significancia tras FDR. Tamaños de efecto: "
            f"rank-biserial, Cohen's d (paramétrico), Cliff's δ (no-paramétrico)."
        )
        lines.append("")
        t = tests.copy()
        for col in ("p_value", "p_holm", "p_bh"):
            t[col] = t[col].map(lambda x: f"{x:.4g}" if not pd.isna(x) else "—")
        for col in ("median_diff", "rank_biserial", "cohens_d", "cliffs_delta"):
            t[col] = t[col].map(lambda x: f"{x:+.3f}" if not pd.isna(x) else "—")
        t["power_post_hoc"] = t["power_post_hoc"].map(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "—"
        )
        cols_show = [
            "n_pares", "median_diff",
            "p_value", "p_holm", "p_bh",
            "cohens_d", "cliffs_delta", "rank_biserial",
            "power_post_hoc", "sig", "sig_bh",
        ]
        t = t.sort_values(by="sig_bh", ascending=False)
        lines.append(t[cols_show].to_markdown())
        lines.append("")
        lines.append(
            "Convención de significancia: `*` p<0.05, `**` p<0.01, `***` p<0.001. "
            "Magnitud Cohen's d: 0.2 chico, 0.5 medio, 0.8 grande. "
            "Magnitud Cliff's δ: 0.147 chico, 0.33 medio, 0.474 grande."
        )
        lines.append("")

    # 5. Mixed-effects sobre datos turn-level
    if me_table is not None and not me_table.empty:
        lines.append(
            f"## 5. Mixed-effects (turn-level): `metric ~ {model_b} + (1|seed)`"
        )
        lines.append("")
        lines.append(
            f"Aprovecha las {n_replicas * 8} × N obs por modelo en vez de "
            f"colapsar a N. El efecto fijo de modelo es la diferencia esperada "
            f"`{model_b} − {model_a}` controlando por la correlación intra-seed. "
            f"Más datos efectivos → IC95 más apretado y p-values más pequeños "
            f"que el Wilcoxon end-of-horizon."
        )
        lines.append("")
        m = me_table.copy()
        for col in ("fixed_effect_b_minus_a", "se", "ci95_lo", "ci95_hi"):
            if col in m.columns:
                m[col] = m[col].map(lambda x: f"{x:+.3f}" if not pd.isna(x) else "—")
        for col in ("p_value", "p_bh"):
            if col in m.columns:
                m[col] = m[col].map(lambda x: f"{x:.4g}" if not pd.isna(x) else "—")
        cols_show = [
            "fixed_effect_b_minus_a", "ci95_lo", "ci95_hi",
            "p_value", "p_bh", "n_obs", "n_seeds", "sig_bh",
        ]
        cols_show = [c for c in cols_show if c in m.columns]
        m = m.sort_values(by="sig_bh", ascending=False)
        lines.append(m[cols_show].to_markdown())
        lines.append("")

    # 6. ICC sobre réplicas
    if icc_table is not None and not icc_table.empty:
        lines.append("## 6. ICC (test-retest dentro de modelo)")
        lines.append("")
        lines.append(
            "ICC alto (→ 1) significa que repetir el mismo seed con el mismo "
            "modelo produce resultados parecidos: la varianza está dominada "
            "por la diferencia entre seeds (mismos shocks → misma respuesta). "
            "ICC bajo (→ 0) significa que el sampler de Boltzmann del LLM "
            "(temperatura ~1) introduce tanta varianza intra-modelo como la que "
            "vemos entre modelos — y entonces las diferencias inter-modelo "
            "deben interpretarse con cuidado."
        )
        lines.append("")
        ic = icc_table.copy()
        for col in ("icc", "var_seed", "var_resid"):
            if col in ic.columns:
                ic[col] = ic[col].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
        cols_show = [c for c in ("icc", "var_seed", "var_resid", "n_obs", "n_seeds")
                     if c in ic.columns]
        lines.append(ic[cols_show].to_markdown())
        lines.append("")

    # 7. Datos crudos
    lines.append("## 7. Datos crudos")
    lines.append("")
    lines.append("- `metrics_per_seed.csv` — fin-de-horizonte por (seed, replica, modelo).")
    lines.append("- `aggregate_by_model.csv` — media, std, IC95 por modelo×métrica.")
    lines.append("- `paired_tests.csv` — Wilcoxon + correcciones + tamaños de efecto.")
    if me_table is not None and not me_table.empty:
        lines.append("- `mixed_effects.csv` — coeficientes y CI95 del efecto del modelo.")
    if icc_table is not None and not icc_table.empty:
        lines.append("- `icc.csv` — ICC por (modelo, métrica).")
    lines.append("- `turn_metrics_long.csv` — long-format turn-level (input de mixed-effects).")
    lines.append("- `presupuesto_ic95.png`, `outcomes_box.png`,"
                 " `mixed_effects_forest.png`.")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# --- entrypoint -----------------------------------------------------------


def analyze(
    runs: Sequence[SeedRun],
    out_dir: Path,
    model_a: str = "Claude",
    model_b: str = "OpenAI",
) -> dict[str, Path]:
    """Pipeline completo: agregación + tests pareados + mixed-effects + ICC.

    Genera CSVs, plots y `summary.md` en `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # End-of-horizon
    df = collect_metrics(runs)
    agg = aggregate_by_model(df)
    try:
        tests = paired_tests(df, model_a, model_b)
    except Exception:
        tests = pd.DataFrame()

    # Turn-level
    df_long = collect_turn_metrics(runs)
    metrics_for_me = [
        "pib_usd_mm", "deuda_pib", "pobreza_general",
        "aprobacion_presidencial", "indice_protesta",
        "ind_bienestar", "ind_gobernabilidad", "ind_estabilidad_macro",
        "ind_desarrollo_humano", "ind_estres_social",
        "delta_iva_pp", "delta_isr_pp",
    ]
    metrics_for_me += [c for c in df_long.columns if c.startswith("presup_")]
    try:
        me_table = fit_mixed_effects_all(df_long, metrics_for_me, model_a, model_b)
    except Exception:
        me_table = pd.DataFrame()

    # ICC (sólo si hay réplicas múltiples)
    has_replicas = df_long["replica"].nunique() > 1
    if has_replicas:
        try:
            icc_table = compute_icc_all(df_long, metrics_for_me, [model_a, model_b])
        except Exception:
            icc_table = pd.DataFrame()
    else:
        icc_table = pd.DataFrame()

    # Plots
    plot_budget_with_ci(df, agg, out_dir / "presupuesto_ic95.png")
    plot_outcomes_box(df, out_dir / "outcomes_box.png")
    if not me_table.empty:
        plot_mixed_effects_forest(me_table, out_dir / "mixed_effects_forest.png")

    # Reporte
    md = write_report(df, agg, tests, me_table, icc_table, out_dir,
                      model_a=model_a, model_b=model_b)

    # CSVs
    df.to_csv(out_dir / "metrics_per_seed.csv")
    agg.to_csv(out_dir / "aggregate_by_model.csv")
    df_long.to_csv(out_dir / "turn_metrics_long.csv", index=False)
    if not tests.empty:
        tests.to_csv(out_dir / "paired_tests.csv")
    if not me_table.empty:
        me_table.to_csv(out_dir / "mixed_effects.csv")
    if not icc_table.empty:
        icc_table.to_csv(out_dir / "icc.csv")

    paths = {
        "summary": md,
        "metrics_per_seed": out_dir / "metrics_per_seed.csv",
        "aggregate": out_dir / "aggregate_by_model.csv",
        "turn_metrics_long": out_dir / "turn_metrics_long.csv",
        "presupuesto_plot": out_dir / "presupuesto_ic95.png",
        "outcomes_plot": out_dir / "outcomes_box.png",
    }
    if not tests.empty:
        paths["tests"] = out_dir / "paired_tests.csv"
    if not me_table.empty:
        paths["mixed_effects"] = out_dir / "mixed_effects.csv"
        paths["mixed_effects_plot"] = out_dir / "mixed_effects_forest.png"
    if not icc_table.empty:
        paths["icc"] = out_dir / "icc.csv"
    return paths
