"""BRCA system architecture (transformer-paper style schematic).

Two phases: (A) online trajectory generation -- the simulator<->LLM loop
over T turns; (B) offline black-box audit -- one Bayesian-IRL posterior
fanning out into the three audit heads (IRD, harm, RPC), with the
normative baselines feeding the IRD comparison. Complements the linear
7-stage pipeline figure by showing the loop and the branching.

Usage: python scripts/fig_architecture.py
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
EDGE = "#34495e"
A_BG = "#fdf2e9"
A_BOX = "#f5cba7"
B_BG = "#eaf2f8"
B_BOX = "#aed6f1"
CORE = "#85c1e9"
OUTC = "#d5f5e3"


def box(ax, x, y, w, h, text, fc, fs=8.0, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 linewidth=1.1, edgecolor=EDGE, facecolor=fc, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=("bold" if bold else "normal"),
            color="#1b2631", zorder=4)


def arrow(ax, p1, p2, label="", rad=0.0, fs=6.3, off=(0, 0.12), color=EDGE, lw=1.2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                 linewidth=lw, color=color, zorder=2,
                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color="#566573", style="italic", zorder=5)


fig, ax = plt.subplots(figsize=(7.1, 3.5))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")

# phase backgrounds
ax.add_patch(FancyBboxPatch((0.2, 0.5), 5.5, 6.0, boxstyle="round,pad=0.02,rounding_size=0.1",
             linewidth=1.0, edgecolor="#b9770e", facecolor="#fdf2e9", zorder=1))
ax.add_patch(FancyBboxPatch((6.3, 0.5), 7.5, 6.0, boxstyle="round,pad=0.02,rounding_size=0.1",
             linewidth=1.0, edgecolor="#2471a3", facecolor=B_BG, zorder=1))
ax.text(2.95, 6.18, "A. Trajectory generation  (online, per seed)",
        ha="center", fontsize=7.6, fontweight="bold", color="#b9770e")
ax.text(10.05, 6.18, "B. Behavioral audit  (offline, black-box)",
        ha="center", fontsize=7.6, fontweight="bold", color="#2471a3")

# --- phase A: simulator <-> LLM loop
box(ax, 0.7, 4.2, 1.9, 1.0, "Calibrated\nsimulator", A_BOX)
box(ax, 3.5, 4.2, 1.9, 1.0, "Candidate\nmenu (K=5)", A_BOX)
box(ax, 1.7, 1.7, 2.8, 1.05, "LLM policy\n(system prompt)", A_BOX)
arrow(ax, (2.6, 4.7), (3.5, 4.7), r"$s_t$")
arrow(ax, (4.45, 4.2), (3.6, 2.75), r"menu", rad=-0.25, off=(0.35, 0))
arrow(ax, (1.9, 2.75), (1.45, 4.2), r"$a_t$", rad=-0.25, off=(-0.32, 0))
ax.text(1.0, 3.5, r"$\times\,T$ turns", fontsize=6.5, color="#b9770e", style="italic")
ax.text(3.1, 1.45, r"$\to k_t,\ \mathrm{CoT}$", fontsize=6.4, color="#566573", ha="center")

# trajectory hand-off A -> B
box(ax, 1.9, 0.7, 2.4, 0.55, r"trajectory  $\tau=\{s_t,k_t,\mathrm{CoT}_t\}$", "white", fs=6.6, bold=False)
arrow(ax, (3.0, 1.7), (3.1, 1.25))
arrow(ax, (4.3, 0.97), (6.6, 4.3), r"$\tau$", rad=0.18, off=(0.2, -0.15))

# --- phase B: feature -> IRL -> three heads
box(ax, 6.55, 4.2, 1.7, 1.0, "Feature\nextraction $\\phi$", B_BOX)
box(ax, 8.55, 4.2, 2.0, 1.0, "Bayesian IRL\n(NUTS) $\\to\\theta_{\\mathrm{rec}}$", CORE)
arrow(ax, (8.25, 4.7), (8.55, 4.7))

box(ax, 11.05, 5.15, 2.55, 0.95, "IRD audit\n$\\cos,\\ \\mathrm{ROPE}$", OUTC, fs=7.5)
box(ax, 11.05, 3.5, 2.55, 0.95, "Harm\ntranslation", OUTC, fs=7.5)
box(ax, 11.05, 1.85, 2.55, 0.95, "RPC coherence\n(CoT vs $\\theta_{\\mathrm{rec}}$)", OUTC, fs=7.5)
arrow(ax, (10.55, 4.7), (11.05, 5.6), rad=-0.2)
arrow(ax, (10.55, 4.7), (11.05, 3.97), rad=0.0)
arrow(ax, (10.55, 4.6), (11.05, 2.32), rad=0.22)

# stated reward + baselines into IRD audit
box(ax, 8.4, 0.7, 2.4, 0.62, r"$\theta_{\mathrm{stated}}$ + B1/B2/B3", "white", fs=6.6, bold=False)
arrow(ax, (10.8, 1.01), (12.3, 5.15), rad=-0.28, off=(0.55, 0), color="#7f8c8d", lw=1.0)

fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_architecture.{ext}", dpi=300, bbox_inches="tight")
print("[fig] architecture ok")
plt.close(fig)
