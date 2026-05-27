"""Three method-illustration figures (real data), single-column for IEEE.

  fig_llm_choice  : Stage 3 -- menu of K=5 real candidate budgets, the
                    Boltzmann softmax P(k) under the recovered theta, the
                    chosen item highlighted, and a real CoT snippet.
  fig_bayesian_irl: Stage 4 -- graphical model (prior -> theta -> Boltzmann
                    -> observed choices) + the real recovered posterior
                    (6 dims, mean +/- IC95) from posterior_pooled.csv.
  fig_ird_radar   : Stage 5 -- radar of theta_stated vs theta_rec (Claude,
                    GPT) over the 6 dims, with the cosine annotated.

Usage: python scripts/fig_methods.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures"
POOLED = OUT / "20260503_181558_dceacd_multiseed_irl_multiseed" / "posterior_pooled.csv"

from guatemala_sim.irl import OUTCOME_FEATURE_NAMES, parse_menu_run  # noqa: E402
from irl_audit_real_run import DEFAULT_W_STATED_INTENT  # noqa: E402

DIMS = list(OUTCOME_FEATURE_NAMES)
SHORT = {
    "anti_pobreza": "anti-pob",
    "anti_deuda": "anti-debt",
    "pro_aprobacion": "pro-appr",
    "pro_crecimiento": "pro-grow",
    "anti_desviacion_inflacion": "anti-infl",
    "pro_confianza": "pro-conf",
}
EDGE = "#34495e"
C_CLAUDE = "#2c7fb8"
C_GPT = "#d95f0e"
C_STATED = "#555555"


def pooled_theta(model: str) -> dict:
    df = pd.read_csv(POOLED)
    sub = df[df.model == model]
    return {r["dim"]: (r["w_mean"], r["ic95_lo"], r["ic95_hi"]) for _, r in sub.iterrows()}


# ---------------------------------------------------------------- Fig 1
def fig_llm_choice() -> None:
    theta = np.array([pooled_theta("claude")[d][0] for d in DIMS])
    base = ROOT / "runs" / "20260503_181558_dceacd_multiseed"

    def softmax(u):
        e = np.exp(u - u.max()); return e / e.sum()

    # Scan seeds for a clean, representative turn: the LLM's actual choice
    # coincides with the mode of the Boltzmann policy under the recovered
    # theta, with the largest such P(chosen).
    best = None  # (P_chosen, seed, t)
    for s in range(1, 9):
        run = base / f"seed{s:03d}_claude.jsonl"
        if not run.exists():
            continue
        pr = parse_menu_run(run, feature_seed=0, n_samples=20)
        feats = np.asarray(pr.features); ch = np.asarray(pr.chosen)
        for t in range(feats.shape[0]):
            P = softmax(feats[t] @ theta)
            k = int(ch[t])
            if k == int(np.argmax(P)) and (best is None or P[k] > best[0]):
                best = (float(P[k]), s, t)
    p_best, seed_sel, t_sel = best
    run = base / f"seed{seed_sel:03d}_claude.jsonl"
    lines = run.read_text(encoding="utf-8", errors="replace").splitlines()
    obj = json.loads(lines[t_sel])
    mc = obj["menu_choice"]
    cand = mc["candidates"]
    chosen = int(mc["chosen_index"])
    phi = np.asarray(parse_menu_run(run, feature_seed=0, n_samples=20).features)[t_sel]
    P = softmax(phi @ theta)
    raz = str(obj["decision"].get("razonamiento", "")).replace("\n", " ").strip()
    raz = raz.encode("ascii", "ignore").decode()[:88]
    cot_text = f"CoT (seed {seed_sel}, turn {t_sel+1}): «{raz}…»"

    partidas = list(cand[0]["presupuesto"].keys())
    budgets = np.array([[c["presupuesto"][p] for p in partidas] for c in cand])  # (K,9)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(partidas))]

    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    ax.set_xlim(0, 1.42); ax.set_ylim(-0.6, len(cand) - 0.3); ax.axis("off")
    for k in range(len(cand)):
        y = len(cand) - 1 - k
        left = 0.0
        for j, p in enumerate(partidas):
            w = budgets[k, j] / 100.0
            ax.add_patch(Rectangle((left, y - 0.28), w, 0.56,
                                   facecolor=colors[j], edgecolor="white", linewidth=0.3))
            left += w
        if k == chosen:
            ax.add_patch(Rectangle((-0.005, y - 0.31), 1.01, 0.62, fill=False,
                                   edgecolor="#c0392b", linewidth=2.0, zorder=5))
        ax.text(-0.03, y, f"c{k}", ha="right", va="center", fontsize=7, fontweight="bold")
        # softmax probability bar to the right
        ax.barh(y, P[k] * 0.34, left=1.05, height=0.5,
                color=("#c0392b" if k == chosen else "#95a5a6"), edgecolor=EDGE, linewidth=0.5)
        ax.text(1.05 + P[k] * 0.34 + 0.01, y, f"{P[k]:.2f}", ha="left", va="center", fontsize=6.5)
    ax.text(0.5, len(cand) - 0.15, "candidate budgets (9 partidas)",
            ha="center", fontsize=7.0, color=EDGE)
    ax.text(1.22, len(cand) - 0.15, r"$P(k)$", ha="center", fontsize=7.5, color=EDGE)
    ax.annotate(r"softmax$(\beta\,\theta\cdot\phi)$", xy=(1.04, 1.6), xytext=(0.62, 1.6),
                fontsize=7, va="center",
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0))
    ax.text(0.0, -0.5, cot_text, ha="left", va="center", fontsize=5.7, style="italic",
            color="#566573", wrap=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.99, bottom=0.02)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_llm_choice.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] llm_choice  chosen=c{chosen}  P={np.round(P,3)}")


# ---------------------------------------------------------------- Fig 2
def fig_bayesian_irl() -> None:
    fig, (axg, axp) = plt.subplots(1, 2, figsize=(3.45, 2.5),
                                   gridspec_kw={"width_ratios": [0.9, 1.25]})
    # --- left: graphical model
    axg.set_xlim(0, 1); axg.set_ylim(0, 1); axg.axis("off")
    axg.add_patch(Circle((0.5, 0.86), 0.075, facecolor="white", edgecolor=EDGE, lw=1.2))
    axg.text(0.5, 0.86, r"$\sigma$", ha="center", va="center", fontsize=8)
    axg.add_patch(Circle((0.5, 0.58), 0.095, facecolor="#d6eaf8", edgecolor=EDGE, lw=1.2))
    axg.text(0.5, 0.58, r"$\theta$", ha="center", va="center", fontsize=9)
    axg.add_patch(Circle((0.5, 0.24), 0.085, facecolor="#dddddd", edgecolor=EDGE, lw=1.2))
    axg.text(0.5, 0.24, r"$k_t$", ha="center", va="center", fontsize=8.5)
    axg.add_patch(FancyBboxPatch((0.27, 0.08), 0.46, 0.33,
                  boxstyle="round,pad=0.005", fill=False, edgecolor="#999999", lw=0.9))
    axg.text(0.70, 0.115, r"$t{=}1..T$", fontsize=6.5, color="#777777")
    for y0, y1 in [(0.785, 0.675), (0.485, 0.325)]:
        axg.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle="-|>",
                      mutation_scale=10, color=EDGE, lw=1.1))
    axg.text(0.62, 0.42, "Boltzmann", fontsize=6.0, color="#777777", rotation=90, va="center")
    axg.text(0.5, 0.005, "prior $\\to\\theta\\to$ choices", ha="center", fontsize=6.3, color=EDGE)

    # --- right: recovered posterior (Claude), 6 dims, mean + IC95
    th = pooled_theta("claude")
    y = np.arange(len(DIMS))[::-1]
    for i, d in enumerate(DIMS):
        m, lo, hi = th[d]
        yy = y[i]
        axp.plot([lo, hi], [yy, yy], color=C_CLAUDE, lw=1.6)
        axp.plot(m, yy, "o", color=C_CLAUDE, ms=4.5)
    axp.axvline(0, color="#bbbbbb", lw=0.8, ls="--")
    axp.set_yticks(y); axp.set_yticklabels([SHORT[d] for d in DIMS], fontsize=6.5)
    axp.set_xlabel(r"$\theta_{\mathrm{rec}}$ posterior (mean, IC95)", fontsize=6.8)
    axp.tick_params(axis="x", labelsize=6)
    axp.set_title("recovered (Claude, 20 seeds)", fontsize=6.8)
    for s in ("top", "right"):
        axp.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.9, bottom=0.17, wspace=0.55)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_bayesian_irl.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[fig] bayesian_irl  ok")


# ---------------------------------------------------------------- Fig 3
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fig_ird_radar() -> None:
    stated = np.array([DEFAULT_W_STATED_INTENT[d] for d in DIMS], dtype=float)
    cl = np.array([pooled_theta("claude")[d][0] for d in DIMS])
    gp = np.array([pooled_theta("openai")[d][0] for d in DIMS])
    cos_cl, cos_gp = _cos(stated, cl), _cos(stated, gp)

    def prof(v):  # L1-normalize the positive part for a comparable radar profile
        p = np.clip(v, 0, None)
        s = p.sum()
        return p / s if s > 0 else p
    s_n, cl_n, gp_n = prof(stated), prof(cl), prof(gp)

    ang = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    fig = plt.figure(figsize=(3.45, 3.1))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(ang); ax.set_xticklabels([SHORT[d] for d in DIMS], fontsize=6.8)
    ax.tick_params(axis="y", labelsize=5.5)
    for vals, col, lab, ls in [(s_n, C_STATED, r"$\theta_{\mathrm{stated}}$", "--"),
                               (cl_n, C_CLAUDE, "Claude", "-"),
                               (gp_n, C_GPT, "GPT-4o-mini", ":")]:
        vc = np.concatenate([vals, vals[:1]])
        ax.plot(ang_c, vc, ls, color=col, lw=1.6, label=lab)
        ax.fill(ang_c, vc, color=col, alpha=0.08)
    ax.set_title("Stated vs recovered constitution", fontsize=7.5, pad=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.16), fontsize=6.0, frameon=False)
    ax.text(0.5, -0.22,
            f"cos(stated, rec): Claude {cos_cl:.2f}  |  GPT {cos_gp:.2f}  (both misaligned)",
            transform=ax.transAxes, ha="center", fontsize=6.3, color=EDGE)
    fig.subplots_adjust(left=0.08, right=0.86, top=0.82, bottom=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_ird_radar.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] ird_radar  cos_claude={cos_cl:.3f}  cos_gpt={cos_gp:.3f}")


if __name__ == "__main__":
    fig_llm_choice()
    fig_bayesian_irl()
    fig_ird_radar()
