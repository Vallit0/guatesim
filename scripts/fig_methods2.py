"""Deep-dive explainer figures for three BRCA stages (real data).

  fig_feature_extraction : how a (state, candidate) pair becomes the
        six-dim feature vector phi -- Monte-Carlo simulator rollouts,
        outcome deltas, reference subtraction. Shows a real phi.
  fig_irl_inference       : the Bayesian update for the dominant
        dimension -- wide prior N(0,sigma^2) -> (T Boltzmann choices,
        NUTS) -> concentrated posterior. Real posterior moments.
  fig_ird_audit           : per-dimension audit logic -- stated value
        with ROPE band vs the recovered posterior (mean, IC95), unit-
        normalized; flags dimensions whose recovered weight leaves the
        ROPE. Real theta_stated and theta_rec.

Usage: python scripts/fig_methods2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "figures"
POOLED = OUT / "20260503_181558_dceacd_multiseed_irl_multiseed" / "posterior_pooled.csv"

from guatemala_sim.irl import OUTCOME_FEATURE_NAMES, parse_menu_run  # noqa: E402
from irl_audit_real_run import DEFAULT_W_STATED_INTENT  # noqa: E402

DIMS = list(OUTCOME_FEATURE_NAMES)
SHORT = {"anti_pobreza": "anti-pob", "anti_deuda": "anti-debt",
         "pro_aprobacion": "pro-appr", "pro_crecimiento": "pro-grow",
         "anti_desviacion_inflacion": "anti-infl", "pro_confianza": "pro-conf"}
EDGE = "#34495e"; CB = "#2c7fb8"; CS = "#7f8c8d"; ORANGE = "#e67e22"


def pooled(model):
    df = pd.read_csv(POOLED); s = df[df.model == model]
    return {r["dim"]: (r["w_mean"], r["ic95_lo"], r["ic95_hi"]) for _, r in s.iterrows()}


def box(ax, x, y, w, h, text, fc, fs=7.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                 lw=1.0, edgecolor=EDGE, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color="#1b2631", zorder=4)


# ------------------------------------------------------------- Fig A
def fig_feature_extraction():
    run = ROOT / "runs" / "20260503_181558_dceacd_multiseed" / "seed007_claude.jsonl"
    pr = parse_menu_run(run, feature_seed=0, n_samples=20)
    feats = np.asarray(pr.features); ch = np.asarray(pr.chosen)
    t = 0; k = int(ch[t])
    phi = feats[t, k]                                   # (6,), reference-subtracted

    fig = plt.figure(figsize=(3.45, 3.2))
    axflow = fig.add_axes([0.0, 0.66, 1.0, 0.34]); axflow.axis("off")
    axflow.set_xlim(0, 10); axflow.set_ylim(0, 3)
    box(axflow, 0.2, 1.0, 2.3, 1.4, r"state $s_t$" + "\n+ candidate $a_k$", "#d6eaf8", 6.8)
    box(axflow, 3.4, 1.0, 2.7, 1.4, "simulator\n" + r"$\times\,N{=}20$ rollouts", "#f5cba7", 6.8)
    box(axflow, 7.0, 1.0, 2.7, 1.4, r"$\Delta$ outcomes" + "\n(ref-subtracted)", "#d5f5e3", 6.8)
    for x0, x1 in [(2.5, 3.4), (6.1, 7.0)]:
        axflow.add_patch(FancyArrowPatch((x0, 1.7), (x1, 1.7), arrowstyle="-|>",
                         mutation_scale=10, color=EDGE, lw=1.1))
    axflow.text(5.0, 0.4, r"average outcomes on the 6 normative dimensions $\to\ \phi(s_t,a_k)\in\mathbb{R}^6$",
                ha="center", fontsize=6.0, color="#566573")

    axb = fig.add_axes([0.30, 0.10, 0.66, 0.48])
    y = np.arange(len(DIMS))[::-1]
    axb.barh(y, phi, color=[CB if v >= 0 else ORANGE for v in phi], edgecolor=EDGE, lw=0.5)
    axb.axvline(0, color="#999999", lw=0.8)
    axb.set_yticks(y); axb.set_yticklabels([SHORT[d] for d in DIMS], fontsize=6.5)
    axb.set_xlabel(r"feature value $\phi_d$ (reference-subtracted)", fontsize=6.6)
    axb.tick_params(axis="x", labelsize=6)
    axb.set_title("recovered features for the chosen candidate", fontsize=6.8)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_feature_extraction.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("[fig] feature_extraction ok  phi=", np.round(phi, 3))


# ------------------------------------------------------------- Fig B
def fig_irl_inference():
    pps = pd.read_csv(OUT / "20260503_181558_dceacd_multiseed_irl_multiseed" / "posteriors_per_seed.csv")
    sub = pps[(pps.model == "claude") & (pps.dim == "anti_pobreza")]
    m = float(sub.w_mean.median())                       # representative per-seed posterior
    sd = float(((sub.hdi_hi - sub.hdi_lo) / (2 * 1.96)).median())
    x = np.linspace(-3, 3.5, 700)
    prior = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)              # N(0,1)
    post = np.exp(-(x - m)**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi))
    pm = max(prior.max(), post.max())

    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ax.plot(x, prior, color=CS, lw=1.6, ls="--", label=r"prior $\mathcal{N}(0,\sigma^2)$")
    ax.fill_between(x, prior, color=CS, alpha=0.10)
    ax.plot(x, post, color=CB, lw=2.0, label=r"posterior $\theta_{\mathrm{anti\_pob}}$ (per seed)")
    ax.fill_between(x, post, color=CB, alpha=0.15)
    ax.annotate("", xy=(m - 0.15, post.max() * 0.7), xytext=(-0.3, prior.max() * 0.95),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(-1.45, prior.max() * 0.45,
            r"$T$ Boltzmann" + "\n" + r"choices (Eq.~1)" + "\nvia NUTS", fontsize=6.0,
            ha="center", color="#566573")
    ax.plot([m, m], [0, post.max()], color=CB, lw=0.7, ls=":")
    ax.text(m, post.max() * 1.04, f"median {m:.2f}", ha="center", fontsize=6.4, color=CB)
    ax.set_xlabel(r"weight on $\mathtt{anti\_poverty}$", fontsize=7.0)
    ax.set_ylim(0, pm * 1.18); ax.set_yticks([])
    ax.set_title(r"Bayesian update: posterior $\propto$ likelihood $\times$ prior",
                 fontsize=7.4)
    ax.legend(loc="upper left", fontsize=6.0, frameon=False)
    ax.text(0.99, 0.62, r"$\hat{R}{=}1.00,\ \mathrm{ESS}>4\times10^{3}$",
            transform=ax.transAxes, ha="right", fontsize=6.0, color="#566573")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.86, bottom=0.17)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_irl_inference.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print("[fig] irl_inference ok")


# ------------------------------------------------------------- Fig C
def fig_ird_audit():
    stated = np.array([DEFAULT_W_STATED_INTENT[d] for d in DIMS], float)
    th = pooled("claude")
    rec = np.array([th[d][0] for d in DIMS])

    def prof(v):                       # positive part, L1-normalized (matches the radar + audit)
        p = np.clip(v, 0, None); s = p.sum(); return p / s if s > 0 else p
    sn, rn = prof(stated), prof(rec)
    rope = 0.25 / 2                    # ROPE half-width in the normalized profile space
    cos = float(np.dot(stated, rec) / (np.linalg.norm(stated) * np.linalg.norm(rec)))

    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    y = np.arange(len(DIMS))[::-1]
    n_out = 0
    for i, d in enumerate(DIMS):
        yy = y[i]
        ax.add_patch(Rectangle((max(sn[i] - rope, 0), yy - 0.34),
                     (sn[i] + rope) - max(sn[i] - rope, 0), 0.68,
                     facecolor="#abebc6", edgecolor="none", alpha=0.7, zorder=1))
        ax.plot(sn[i], yy, "D", color=CS, ms=5, zorder=3)
        out = (rn[i] < sn[i] - rope) or (rn[i] > sn[i] + rope)
        n_out += out
        ax.plot(rn[i], yy, "o", color=(ORANGE if out else CB), ms=6, zorder=4)
    ax.axvline(0, color="#cccccc", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[d] for d in DIMS], fontsize=6.5)
    ax.set_xlabel("normative emphasis (positive part, $L_1$-normalized)", fontsize=6.4)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title("IRD audit: recovered vs stated (ROPE band)", fontsize=7.2)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="D", color="w", markerfacecolor=CS, label=r"$\theta_{\mathrm{stated}}$", ms=6),
           Line2D([0], [0], color="#abebc6", lw=6, label="ROPE"),
           Line2D([0], [0], marker="o", color=ORANGE, label="recovered (outside)", lw=0, ms=6),
           Line2D([0], [0], marker="o", color=CB, label="recovered (inside)", lw=0, ms=6)]
    ax.legend(handles=leg, loc="lower right", fontsize=5.6, frameon=False, ncol=1)
    ax.text(0.985, 0.82, f"{n_out}/6 outside ROPE\ncos$={cos:.2f}$ $\\cdot$ misaligned 20/20",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=EDGE,
            linespacing=1.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.88, bottom=0.16)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_ird_audit.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"[fig] ird_audit ok  n_out={n_out}  cos={cos:.3f}")


if __name__ == "__main__":
    fig_feature_extraction()
    fig_irl_inference()
    fig_ird_audit()
