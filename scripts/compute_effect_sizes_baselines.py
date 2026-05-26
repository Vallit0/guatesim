"""T1.5 — Effect sizes against B1/B2/B3 baselines, per model.

Pre-registered in paper/TUNING_PREREG.md as descriptive enrichment of
Table V and §V.B. No new claim, no decision rule.

Inputs (artifact CSVs already on disk):
  - figures/20260503_181558_dceacd_multiseed_baselines/baselines_per_seed.csv
    with score_llm, score_b1_constrained_optimum, score_b2_random_uniform
    per (seed, model).
  - figures/20260503_181558_dceacd_multiseed_b3_anchor/per_seed.csv
    with l1_vs_minfin, cos_vs_minfin per (seed, model).

Output: figures/20260503_181558_dceacd_multiseed_irl_multiseed/
        paired_effect_sizes_baselines.csv

Run from repo root:
  python scripts/compute_effect_sizes_baselines.py

What it reports, per (model, baseline_metric):
  - n: number of seeds.
  - median_diff: median per-seed (score_llm - score_baseline) with sign
    convention "positive means LLM outperforms baseline on the metric's
    natural direction".
  - cliffs_delta_within: sign-dominance of the LLM vs the baseline,
    paired by seed.  delta = (#pos - #neg) / n.
  - rank_biserial_within: same data as Wilcoxon r_rb, paired by seed.
  - wilcoxon_p: paired Wilcoxon signed-rank p-value (zero ties).

And, per metric, the cross-model Cliff's delta on the per-seed score
itself (Claude - OpenAI, paired by seed).  This second block mirrors
the existing paired_effect_sizes.csv but for the B1/B2/B3 axes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT_DIR = Path("figures/20260503_181558_dceacd_multiseed_irl_multiseed")
BASELINES_CSV = Path(
    "figures/20260503_181558_dceacd_multiseed_baselines/baselines_per_seed.csv"
)
B3_CSV = Path(
    "figures/20260503_181558_dceacd_multiseed_b3_anchor/per_seed.csv"
)


def cliffs_delta(diff: np.ndarray) -> float:
    n = len(diff)
    if n == 0:
        return 0.0
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    return (pos - neg) / n


def rank_biserial(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom else 0.0


def magnitude(delta: float) -> str:
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def wilcoxon_p(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if len(d) < 2:
        return float("nan")
    try:
        _, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except ValueError:
        return float("nan")


def within_model_block(df: pd.DataFrame) -> pd.DataFrame:
    """For each (model, baseline_axis), paired (seed-level)
    sign-dominance of LLM vs baseline.

    Sign convention: positive means LLM is on the *favored* side of the
    metric (higher score for B1/B2; lower L1 for B3; higher cosine for B3).
    """
    rows = []

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].sort_values("seed")
        n = len(sub)

        # B1 regret: ideal is score_llm == score_b1; negative diff means
        # the LLM under-performs B1.  We report (score_llm - score_b1).
        diff_b1 = (sub["score_llm"] - sub["score_b1_constrained_optimum"]).to_numpy()
        # B2 lift: positive means LLM outperforms random uniform.
        diff_b2 = (sub["score_llm"] - sub["score_b2_random_uniform"]).to_numpy()
        # B3 L1: smaller is better; we sign-flip so "positive = LLM is
        # closer to MINFIN" by computing -(l1 - reference).  Here we
        # report just l1 itself for the within-model block; the
        # cross-model block below uses paired model-vs-model.
        # For within-model, we use cosine_vs_minfin as the natural
        # "higher is closer" metric.
        diff_b3_cos = sub["cos_vs_minfin"].to_numpy()  # raw cosine; not a diff
        # No within-model paired test for B3 (there is no per-seed
        # reference point), so this row reports the cosine distribution
        # summary only.

        rows.append(
            {
                "model": model,
                "axis": "B1_regret (score_llm - score_b1)",
                "n_seeds": n,
                "median": float(np.median(diff_b1)),
                "iqr_low": float(np.quantile(diff_b1, 0.25)),
                "iqr_high": float(np.quantile(diff_b1, 0.75)),
                "cliffs_delta_within": cliffs_delta(diff_b1),
                "rank_biserial_within": rank_biserial(diff_b1),
                "magnitude": magnitude(cliffs_delta(diff_b1)),
                "wilcoxon_p_within": wilcoxon_p(diff_b1),
            }
        )
        rows.append(
            {
                "model": model,
                "axis": "B2_lift (score_llm - score_b2)",
                "n_seeds": n,
                "median": float(np.median(diff_b2)),
                "iqr_low": float(np.quantile(diff_b2, 0.25)),
                "iqr_high": float(np.quantile(diff_b2, 0.75)),
                "cliffs_delta_within": cliffs_delta(diff_b2),
                "rank_biserial_within": rank_biserial(diff_b2),
                "magnitude": magnitude(cliffs_delta(diff_b2)),
                "wilcoxon_p_within": wilcoxon_p(diff_b2),
            }
        )
        rows.append(
            {
                "model": model,
                "axis": "B3_cosine_vs_MINFIN (raw)",
                "n_seeds": n,
                "median": float(np.median(diff_b3_cos)),
                "iqr_low": float(np.quantile(diff_b3_cos, 0.25)),
                "iqr_high": float(np.quantile(diff_b3_cos, 0.75)),
                "cliffs_delta_within": float("nan"),
                "rank_biserial_within": float("nan"),
                "magnitude": "n/a",
                "wilcoxon_p_within": float("nan"),
            }
        )

    return pd.DataFrame(rows)


def cross_model_block(df: pd.DataFrame) -> pd.DataFrame:
    """For each baseline-axis metric, paired Cliff's delta on
    (claude_metric - openai_metric) per seed.  Mirrors the existing
    paired_effect_sizes.csv layout but for the baseline axes.
    """
    metrics = {
        "score_b1_regret": ("score_llm", "score_b1_constrained_optimum", "diff"),
        "score_b2_lift":   ("score_llm", "score_b2_random_uniform", "diff"),
        "agreement_llm_vs_b1": ("agreement_llm_vs_b1", None, "raw"),
        "l1_vs_minfin":    ("l1_vs_minfin", None, "raw"),
        "cos_vs_minfin":   ("cos_vs_minfin", None, "raw"),
    }

    rows = []
    for metric, (col, ref, mode) in metrics.items():
        cdf = df[df["model"] == "claude"].sort_values("seed").reset_index(drop=True)
        odf = df[df["model"] == "openai"].sort_values("seed").reset_index(drop=True)

        if mode == "diff":
            c_series = cdf[col].to_numpy() - cdf[ref].to_numpy()
            o_series = odf[col].to_numpy() - odf[ref].to_numpy()
        else:
            if col not in cdf.columns:
                continue
            c_series = cdf[col].to_numpy()
            o_series = odf[col].to_numpy()

        diff = c_series - o_series
        rows.append(
            {
                "metric": metric,
                "n_pairs": len(diff),
                "median_diff_claude_minus_openai": float(np.median(diff)),
                "cliffs_delta": cliffs_delta(diff),
                "rank_biserial_r_rb": rank_biserial(diff),
                "magnitude_cliffs": magnitude(cliffs_delta(diff)),
                "pvalue_wilcoxon": wilcoxon_p(diff),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    bl = pd.read_csv(BASELINES_CSV)
    b3 = pd.read_csv(B3_CSV)

    df = bl.merge(b3, on=["seed", "model"], how="inner")
    if len(df) < len(bl):
        print(
            f"WARN: baselines has {len(bl)} rows, b3 has {len(b3)} rows, "
            f"merged inner = {len(df)} rows.  Some seeds dropped."
        )

    within = within_model_block(df)
    cross = cross_model_block(df)

    print("=" * 70)
    print("WITHIN-MODEL (per-seed sign-dominance of LLM vs baseline)")
    print("=" * 70)
    print(within.to_string(index=False))
    print()
    print("=" * 70)
    print("CROSS-MODEL (Claude vs OpenAI per-seed, per baseline axis)")
    print("=" * 70)
    print(cross.to_string(index=False))

    out_within = OUT_DIR / "paired_effect_sizes_baselines_within.csv"
    out_cross = OUT_DIR / "paired_effect_sizes_baselines_cross.csv"
    out_within.parent.mkdir(parents=True, exist_ok=True)
    within.to_csv(out_within, index=False)
    cross.to_csv(out_cross, index=False)
    print()
    print(f"Wrote {out_within}")
    print(f"Wrote {out_cross}")


if __name__ == "__main__":
    main()
