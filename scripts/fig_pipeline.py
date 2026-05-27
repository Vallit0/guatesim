"""Render the seven-layer BRCA audit pipeline as a publication figure.

Replaces the wide Table (tab:pipeline) with a compact horizontal flow
diagram. Country-specific layers (the simulator, Layer 1) are shaded
differently from the country-agnostic audit layers (2-7). Saved as both
PNG (preview) and PDF (vector, for \includegraphics in the paper).

Usage: python scripts/fig_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

# (number, short name lines, input, output, country_specific)
STAGES = [
    (1, "World\nsimulator",        r"$a_t,\,s_t$, seed",          r"$s_{t+1}$",                 True),
    (2, "Candidate\nmenu",         r"$s_t$",                       r"$K{=}5$ over 9-simplex",    False),
    (3, "LLM\nchoice",             r"menu, prompt",                r"$k_t$, CoT",                False),
    (4, "Bayesian\nIRL",           r"features, $k_t$",             r"$\theta_{\mathrm{rec}}$ (NUTS)", False),
    (5, "IRD\naudit",              r"$\theta_{\mathrm{rec}},\theta_{\mathrm{stat}}$", r"cosine, ROPE, gap", False),
    (6, "Harm\ntranslation",       r"$s_0,\,s_T$",                 r"hogares, USD",              False),
    (7, "Reasoning\ncoherence",    r"CoT, $\theta_{\mathrm{rec}}$", r"cosine, flag",             False),
]

C_SPEC = "#f5cba7"   # country-specific (simulator)
C_AGN  = "#aed6f1"   # country-agnostic audit layers
C_CORE = "#85c1e9"   # IRD audit (the core alignment check), a touch darker
EDGE   = "#34495e"

fig, ax = plt.subplots(figsize=(7.1, 2.35))
ax.set_xlim(0, 7.1)
ax.set_ylim(0, 2.35)
ax.axis("off")

n = len(STAGES)
bw, bh = 0.82, 1.18           # box width / height
gap = (7.1 - n * bw) / (n + 1) # equal gaps incl. margins
y0 = 0.72                      # box bottom

centers = []
for i, (num, name, inp, outp, spec) in enumerate(STAGES):
    x = gap + i * (bw + gap)
    cx = x + bw / 2
    centers.append(cx)
    color = C_SPEC if spec else (C_CORE if num == 5 else C_AGN)
    box = FancyBboxPatch((x, y0), bw, bh,
                         boxstyle="round,pad=0.015,rounding_size=0.06",
                         linewidth=1.1, edgecolor=EDGE, facecolor=color)
    ax.add_patch(box)
    # number badge
    ax.add_patch(Circle((x + 0.13, y0 + bh - 0.13), 0.085,
                        facecolor=EDGE, edgecolor="none", zorder=3))
    ax.text(x + 0.13, y0 + bh - 0.13, str(num), ha="center", va="center",
            color="white", fontsize=7.5, fontweight="bold", zorder=4)
    # layer name
    ax.text(cx, y0 + bh * 0.55, name, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#1b2631")
    # input / output (small, below name inside box)
    ax.text(cx, y0 + 0.20, outp, ha="center", va="center",
            fontsize=6.0, color="#1b2631")
    # input label below the box
    ax.text(cx, y0 - 0.14, inp, ha="center", va="top",
            fontsize=5.8, color="#566573", style="italic")

# arrows between consecutive boxes
for i in range(n - 1):
    x_start = centers[i] + bw / 2
    x_end = centers[i + 1] - bw / 2
    ax.add_patch(FancyArrowPatch((x_start, y0 + bh / 2),
                                 (x_end, y0 + bh / 2),
                                 arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.1, color=EDGE, zorder=2))

# legend
ax.add_patch(FancyBboxPatch((0.30, 0.06), 0.18, 0.16,
             boxstyle="round,pad=0.01,rounding_size=0.04",
             linewidth=0.8, edgecolor=EDGE, facecolor=C_SPEC))
ax.text(0.54, 0.14, "country-specific (calibration)", ha="left", va="center", fontsize=6.2)
ax.add_patch(FancyBboxPatch((3.55, 0.06), 0.18, 0.16,
             boxstyle="round,pad=0.01,rounding_size=0.04",
             linewidth=0.8, edgecolor=EDGE, facecolor=C_AGN))
ax.text(3.79, 0.14, "country-agnostic audit layers", ha="left", va="center", fontsize=6.2)

plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
for ext in ("png", "pdf"):
    p = OUT / f"pipeline_7stage.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"[fig] -> {p}")
plt.close(fig)
