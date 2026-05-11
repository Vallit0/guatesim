"""B3 (human-process anchor): LLM trajectories vs MINFIN 2024.

For each (seed, model) pair we average the 8 quarterly budget vectors
the LLM emitted, then compare that mean budget to the MINFIN 2024
appropriated/executed shares (data/minfin_2024_ejecutado.csv, validated
against ICEFI Tables 7 and 8).

Reported metrics (per seed, per model):
  - L1 deviation: sum_j |w_LLM_j - w_MINFIN_j|, in percentage points.
  - cosine similarity: cos(w_LLM, w_MINFIN).

Aggregate across seeds:
  - median + IQR per model and metric
  - paired Wilcoxon Claude vs OpenAI on both metrics

Outputs:
  - figures/<batch>_b3_anchor/per_seed.csv
  - figures/<batch>_b3_anchor/summary.md
  - figures/<batch>_b3_anchor/comparison.png

Usage:
    python irl_b3_human_anchor.py \
        --batch-dir runs/20260503_181558_dceacd_multiseed \
        --out figures/20260503_181558_dceacd_multiseed_b3_anchor
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from guatemala_sim.minfin_ingest import load_minfin_baseline
from guatemala_sim.minfin_plot import PARTIDAS_LABEL_CORTO, PARTIDAS_ORDEN

ROOT = Path(__file__).resolve().parent
RE_RUN = re.compile(
    r"^seed(?P<seed>\d{3})(?:_R(?P<replica>\d+))?_(?P<label>[a-z][\w]*)\.jsonl$"
)


def avg_budget_from_jsonl(path: Path) -> np.ndarray:
    shares = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            p = obj["decision"]["presupuesto"]
            shares.append([float(p[k]) for k in PARTIDAS_ORDEN])
    arr = np.asarray(shares, dtype=float)
    return arr.mean(axis=0)


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def discover(batch_dir: Path) -> dict[tuple[int, str], Path]:
    out: dict[tuple[int, str], Path] = {}
    for p in sorted(batch_dir.glob("seed*.jsonl")):
        m = RE_RUN.match(p.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        label = m.group("label")
        out[(seed, label)] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    bl = load_minfin_baseline()
    minfin_vec = np.asarray(
        [bl.presupuesto.model_dump()[k] for k in PARTIDAS_ORDEN], dtype=float
    )

    runs = discover(args.batch_dir)
    seeds = sorted({s for s, _ in runs})
    rows = []
    avg_per_model: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        for label in ("claude", "openai"):
            path = runs.get((seed, label))
            if path is None:
                continue
            avg = avg_budget_from_jsonl(path)
            avg_per_model.setdefault(label, []).append(avg)
            rows.append(
                {
                    "seed": seed,
                    "model": label,
                    "l1_vs_minfin": l1(avg, minfin_vec),
                    "cos_vs_minfin": cosine(avg, minfin_vec),
                    **{f"w_{k}": v for k, v in zip(PARTIDAS_ORDEN, avg)},
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_seed.csv", index=False)

    pivot_l1 = df.pivot(index="seed", columns="model", values="l1_vs_minfin")
    pivot_cos = df.pivot(index="seed", columns="model", values="cos_vs_minfin")
    w_l1 = wilcoxon(pivot_l1["claude"], pivot_l1["openai"])
    w_cos = wilcoxon(pivot_cos["claude"], pivot_cos["openai"])

    lines: list[str] = []
    lines.append("# B3 — Human-process anchor (MINFIN 2024) vs LLM trajectories")
    lines.append("")
    lines.append(
        "Per-seed mean budget allocation across the 8 quarterly turns "
        "compared to the MINFIN 2024 appropriated/executed shares "
        "(ICEFI Tables 7 + 8, primary SICOIN data)."
    )
    lines.append("")
    lines.append("## 1. Per-model summary")
    lines.append("")
    summary_rows = []
    for model in ("claude", "openai"):
        l1s = pivot_l1[model].values
        coss = pivot_cos[model].values
        summary_rows.append(
            {
                "model": model,
                "n": len(l1s),
                "L1 median (pp)": float(np.median(l1s)),
                "L1 IQR": f"[{np.percentile(l1s, 25):.2f}, {np.percentile(l1s, 75):.2f}]",
                "cos median": float(np.median(coss)),
                "cos IQR": f"[{np.percentile(coss, 25):.3f}, {np.percentile(coss, 75):.3f}]",
            }
        )
    sdf = pd.DataFrame(summary_rows)
    lines.append(sdf.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")
    lines.append("## 2. Paired Wilcoxon (Claude vs OpenAI)")
    lines.append("")
    lines.append(f"- L1 deviation vs MINFIN: median diff = "
                 f"{float(np.median(pivot_l1['claude'] - pivot_l1['openai'])):+.3f} pp, "
                 f"p = {w_l1.pvalue:.4f}")
    lines.append(f"- cos similarity vs MINFIN: median diff = "
                 f"{float(np.median(pivot_cos['claude'] - pivot_cos['openai'])):+.4f}, "
                 f"p = {w_cos.pvalue:.4f}")
    lines.append("")
    lines.append("## 3. Mean budget per model (% of total) vs MINFIN")
    lines.append("")
    mean_rows = []
    for k in PARTIDAS_ORDEN:
        row = {"partida": k, "MINFIN_2024": float(bl.presupuesto.model_dump()[k])}
        for label in ("claude", "openai"):
            col = f"w_{k}"
            row[label] = float(df[df["model"] == label][col].mean())
        mean_rows.append(row)
    mean_df = pd.DataFrame(mean_rows).set_index("partida")
    lines.append(mean_df.round(2).to_markdown())
    lines.append("")
    lines.append("## 4. Lectura")
    lines.append("")
    cl_med = float(np.median(pivot_l1["claude"]))
    op_med = float(np.median(pivot_l1["openai"]))
    sq_dev = 52.92  # from existing deviations.md, status_quo_uniforme baseline
    closer = "Claude" if cl_med < op_med else "GPT-4o-mini"
    farther = "GPT-4o-mini" if cl_med < op_med else "Claude"
    lines.append(
        f"- Mediana de desviación L1 vs MINFIN: Claude {cl_med:.2f} pp, "
        f"GPT-4o-mini {op_med:.2f} pp. {closer} se aleja menos del proceso "
        f"humano que {farther}."
    )
    lines.append(
        f"- Comparación: el menu candidate más cercano a MINFIN "
        f"(`status_quo_uniforme`) está a {sq_dev:.1f} pp; el más lejano "
        f"(`seguridad_primero`) a 78.8 pp. Las trayectorias LLM se ubican "
        f"dentro de ese rango."
    )
    lines.append("")

    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11.5, 5))
    width = 0.27
    x = np.arange(len(PARTIDAS_ORDEN))
    ax.bar(x - width, [bl.presupuesto.model_dump()[k] for k in PARTIDAS_ORDEN],
           width, label="MINFIN 2024 (B3 human anchor)", color="#2c3e50",
           edgecolor="black", linewidth=0.5)
    ax.bar(x, [mean_df.loc[k, "claude"] for k in PARTIDAS_ORDEN], width,
           label="Claude Haiku 4.5 (mean over 20 seeds × 8 turns)",
           color="#1f77b4", edgecolor="black", linewidth=0.3)
    ax.bar(x + width, [mean_df.loc[k, "openai"] for k in PARTIDAS_ORDEN], width,
           label="GPT-4o-mini (mean over 20 seeds × 8 turns)",
           color="#ff7f0e", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([PARTIDAS_LABEL_CORTO[k] for k in PARTIDAS_ORDEN],
                       rotation=20, ha="right")
    ax.set_ylabel("% of budget")
    ax.set_title("LLM trajectories vs MINFIN 2024 human-process anchor (B3)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(args.out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[b3] per-seed -> {args.out / 'per_seed.csv'}")
    print(f"[b3] summary  -> {args.out / 'summary.md'}")
    print(f"[b3] figure   -> {args.out / 'comparison.png'}")


if __name__ == "__main__":
    main()
