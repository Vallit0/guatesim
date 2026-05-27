"""Holm-Bonferroni family-wise correction over the headline paired tests.

Addresses the cross-layer multiplicity concern (reviewer blocker on the
four audit layers sharing one family of paired Wilcoxon tests). Reads a
paired-test CSV with a `pvalue` column, applies the Holm step-down
procedure, and reports adjusted p-values and which contrasts survive at
a family-wise error rate (FWER) of alpha.

Holm step-down (m tests, p sorted ascending p_(1)..p_(m)):
    p_adj_(k) = max_{j<=k} min(1, (m - j + 1) * p_(j))
i.e. multiply the k-th smallest by (m-k+1), then enforce monotonicity.

Usage:
    python scripts/holm_bonferroni.py \
        --csv figures/<batch>_irl_multiseed/tests_pareados.csv \
        --alpha 0.05 \
        --out figures/<batch>_irl_multiseed/holm_bonferroni.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def holm_bonferroni(pvals: np.ndarray, alpha: float = 0.05):
    """Return (adjusted_pvals, reject) in the ORIGINAL input order."""
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")          # ascending
    ranked = pvals[order]
    # step-down: multiply k-th smallest by (m-k+1), then make monotone non-decreasing
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for k in range(m):
        val = (m - k) * ranked[k]
        running = max(running, val)
        adj_sorted[k] = min(running, 1.0)
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    reject = adj < alpha
    return adj, reject


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--pcol", type=str, default="pvalue")
    ap.add_argument("--metriccol", type=str, default="metric")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    p = df[args.pcol].to_numpy(dtype=float)
    adj, rej = holm_bonferroni(p, alpha=args.alpha)
    df = df.assign(p_holm=adj, survives_holm=rej)
    df = df.sort_values(args.pcol).reset_index(drop=True)

    m = len(p)
    n_raw = int((p < args.alpha).sum())
    n_holm = int(rej.sum())
    print(f"# Holm-Bonferroni over m={m} paired tests (FWER alpha={args.alpha})")
    print(f"# raw p<{args.alpha}: {n_raw}/{m}   ->   Holm-adjusted: {n_holm}/{m}\n")
    print(f"| {args.metriccol} | raw p | Holm p_adj | survives |")
    print("|---|---:|---:|:---:|")
    for _, r in df.iterrows():
        print(f"| {r[args.metriccol]} | {r[args.pcol]:.2e} | "
              f"{r['p_holm']:.2e} | {'YES' if r['survives_holm'] else 'no'} |")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n[holm] -> {args.out}")


if __name__ == "__main__":
    main()
