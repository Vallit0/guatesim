"""T1.2 — Reward rescaling / normalization sweep.

Pre-registered in paper/TUNING_PREREG.md.

For every (seed, model) pair in the main batch, re-fit Bayesian IRL
five times: once with the original features (identity), and once each
under four feature-basis transformations.  Compute the cosine of the
recovered weight vector against the identity baseline.

Variants (all applied to the (T, K, d) feature tensor):
  1. identity                                                          (baseline)
  2. zscore     — per-feature z-score across (T, K)
  3. minmax     — per-feature min-max scaling to [0, 1] across (T, K)
  4. multscale  — random per-feature multiplicative scaling
                  (10 independent draws from Uniform(0.5, 2.0));
                  reports the worst-case cosine across the 10 draws
  5. centered   — per-feature mean-subtract only (no scaling)

Outputs:
  figures/<batch>_t12_normalization/per_seed.csv
  figures/<batch>_t12_normalization/summary.md

Compute cost: 4 active variants × 40 pairs × 1 NUTS fit ≈ 160 fits.
At ~30 s/fit on a laptop this is roughly 1.5 h; pass --quick for a
smoke run that only fits the first three (seed, model) pairs.

Run from repo root:
  python scripts/r7_normalization_sweep.py \\
      --batch-dir runs/20260503_181558_dceacd_multiseed \\
      --out figures/20260503_181558_dceacd_multiseed_t12_normalization
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from guatemala_sim.irl import (
    OUTCOME_FEATURE_NAMES,
    encode_prompt_to_w_stated,
    fit_bayesian_irl,
    parse_menu_run,
)

ROOT = Path(__file__).resolve().parent.parent
RE_RUN = re.compile(
    r"^seed(?P<seed>\d{3})(?:_R(?P<replica>\d+))?_(?P<label>[a-z][\w]*)\.jsonl$"
)

DEFAULT_W_STATED_INTENT: dict[str, float] = {
    "anti_pobreza":              1.0,
    "anti_deuda":                0.3,
    "pro_aprobacion":            0.2,
    "pro_crecimiento":           0.5,
    "anti_desviacion_inflacion": 0.4,
    "pro_confianza":             0.7,
}

# Variant names that always run (deterministic, no replicates).
DETERMINISTIC_VARIANTS = ("identity", "zscore", "minmax", "centered")
# Number of random multiplicative-scaling draws.
MULT_SCALE_DRAWS = 10
MULT_SCALE_RNG_SEED = 20260516


def discover(batch_dir: Path) -> list[tuple[int, str, Path]]:
    out: list[tuple[int, str, Path]] = []
    for p in sorted(batch_dir.glob("seed*.jsonl")):
        m = RE_RUN.match(p.name)
        if m is None:
            continue
        out.append((int(m.group("seed")), m.group("label"), p))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    if n < 1e-12:
        return 0.0
    return float(np.dot(a, b) / n)


def apply_variant(features: np.ndarray, variant: str, rng=None) -> np.ndarray:
    """Return a transformed copy of features (T, K, d).

    All transformations are per-feature (last axis), applied across
    the flattened (T, K) sample of that feature so that no information
    leaks from one feature dimension into another.
    """
    T, K, d = features.shape
    flat = features.reshape(-1, d)  # (T*K, d)

    if variant == "identity":
        return features.copy()

    if variant == "zscore":
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        sd_safe = np.where(sd < 1e-12, 1.0, sd)
        z = (flat - mu) / sd_safe
        return z.reshape(T, K, d)

    if variant == "minmax":
        lo = flat.min(axis=0)
        hi = flat.max(axis=0)
        rng_span = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
        mm = (flat - lo) / rng_span
        return mm.reshape(T, K, d)

    if variant == "centered":
        mu = flat.mean(axis=0)
        return (flat - mu).reshape(T, K, d)

    if variant == "multscale":
        assert rng is not None, "multscale requires an RNG"
        factors = rng.uniform(0.5, 2.0, size=d)
        return features * factors[None, None, :]

    raise ValueError(f"unknown variant: {variant}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--nuts-draws", type=int, default=2000)
    ap.add_argument("--nuts-tune", type=int, default=1000)
    ap.add_argument("--nuts-chains", type=int, default=2)
    ap.add_argument("--nuts-seed", type=int, default=11)
    ap.add_argument("--prior-sigma", type=float, default=1.0)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: only fit the first three (seed, model) pairs.",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    w_stated = encode_prompt_to_w_stated(
        DEFAULT_W_STATED_INTENT,
        feature_names=OUTCOME_FEATURE_NAMES,
        normalize=True,
    )

    runs = discover(args.batch_dir)
    if args.quick:
        runs = runs[:3]
        print(f"[t12] --quick: limiting to first {len(runs)} runs")

    rng = np.random.default_rng(MULT_SCALE_RNG_SEED)
    rows: list[dict[str, object]] = []

    for seed, label, path in runs:
        parsed = parse_menu_run(
            path, feature_seed=args.feature_seed, n_samples=args.n_samples
        )
        features = parsed.features
        chosen = parsed.chosen

        # Identity baseline first; cache its w_mean for the cosines.
        identity_w: np.ndarray | None = None

        for variant in DETERMINISTIC_VARIANTS:
            print(f"[t12] seed={seed:03d} model={label} variant={variant} fitting…")
            feats_v = apply_variant(features, variant)
            post = fit_bayesian_irl(
                features=feats_v,
                chosen=chosen,
                feature_names=OUTCOME_FEATURE_NAMES,
                prior_sigma=args.prior_sigma,
                draws=args.nuts_draws,
                tune=args.nuts_tune,
                chains=args.nuts_chains,
                seed=args.nuts_seed,
                progressbar=False,
            )
            w_mean = post.w_mean.copy()
            if variant == "identity":
                identity_w = w_mean
                cos_vs_identity = 1.0
            else:
                assert identity_w is not None
                cos_vs_identity = cosine(w_mean, identity_w)

            row = {
                "seed": seed,
                "model": label,
                "variant": variant,
                "draw": 0,
                "w_norm": float(np.linalg.norm(w_mean)),
                "cos_vs_identity": cos_vs_identity,
                "cos_to_stated": cosine(w_mean, w_stated),
                "rhat_max": float(post.rhat_max),
                "ess_bulk_min": float(post.ess_bulk_min),
            }
            for name, val in zip(OUTCOME_FEATURE_NAMES, w_mean):
                row[f"w_{name}"] = float(val)
            rows.append(row)

        # Multiplicative scaling: MULT_SCALE_DRAWS independent draws.
        assert identity_w is not None
        for draw_idx in range(MULT_SCALE_DRAWS):
            feats_v = apply_variant(features, "multscale", rng=rng)
            print(
                f"[t12] seed={seed:03d} model={label} variant=multscale "
                f"draw={draw_idx+1}/{MULT_SCALE_DRAWS} fitting…"
            )
            post = fit_bayesian_irl(
                features=feats_v,
                chosen=chosen,
                feature_names=OUTCOME_FEATURE_NAMES,
                prior_sigma=args.prior_sigma,
                draws=args.nuts_draws,
                tune=args.nuts_tune,
                chains=args.nuts_chains,
                seed=args.nuts_seed,
                progressbar=False,
            )
            w_mean = post.w_mean.copy()
            row = {
                "seed": seed,
                "model": label,
                "variant": "multscale",
                "draw": draw_idx,
                "w_norm": float(np.linalg.norm(w_mean)),
                "cos_vs_identity": cosine(w_mean, identity_w),
                "cos_to_stated": cosine(w_mean, w_stated),
                "rhat_max": float(post.rhat_max),
                "ess_bulk_min": float(post.ess_bulk_min),
            }
            for name, val in zip(OUTCOME_FEATURE_NAMES, w_mean):
                row[f"w_{name}"] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_seed.csv", index=False)

    # Summary
    lines: list[str] = []
    lines.append("# T1.2 — Normalization sweep")
    lines.append("")
    lines.append(
        f"Re-fits over {len(runs)} (seed, model) pairs under "
        f"{len(DETERMINISTIC_VARIANTS)} deterministic variants "
        f"and {MULT_SCALE_DRAWS} multiplicative-scaling draws."
    )
    lines.append("")
    lines.append("## Cosine of recovered weights vs identity baseline")
    lines.append("")
    lines.append("| model | variant | n | median cos | min cos | p10 cos |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for (model, variant), s in df.groupby(["model", "variant"])["cos_vs_identity"]:
        if variant == "identity":
            continue
        lines.append(
            f"| {model} | {variant} | {len(s)} | {s.median():.4f} | "
            f"{s.min():.4f} | {s.quantile(0.10):.4f} |"
        )
    lines.append("")
    lines.append(
        "Pre-registered decision rule (TUNING_PREREG §T1.2): robust "
        "if median cosine ≥ 0.95 for zscore/minmax/centered, and "
        "for multscale the 10-draw 10th percentile ≥ 0.85."
    )
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[t12] per-seed -> {args.out / 'per_seed.csv'}")
    print(f"[t12] summary  -> {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
