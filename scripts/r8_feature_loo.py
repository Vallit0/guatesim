"""T1.3 — Feature leave-one-out (drop one feature dimension at a time).

Pre-registered in paper/TUNING_PREREG.md.

For every (seed, model) pair in the main batch, re-fit Bayesian IRL
six times: once with each of the 6 outcome features dropped from the
feature tensor.  Compute the cosine of the *remaining-5-dim* weight
vector against the matching 5 dimensions of the full-d baseline, and
the reclassification of "significantly misaligned" under a 5-dim
stated reward (also with the same feature dropped).

Why this matters: the paper already has R4 (menu LOO) but the
reviewer asked for a complementary check at the *feature basis*
level.  If dropping a feature flips the verdict in many pairs, that
feature is structural to the recovery; if not, the direction is
basis-robust.

Outputs:
  figures/<batch>_t13_feature_loo/per_seed.csv
  figures/<batch>_t13_feature_loo/summary.md

Compute cost: 6 dropouts × 40 pairs × 1 NUTS fit ≈ 240 fits;
~2 h on a laptop.  Pass --quick for a 3-pair smoke run.

Run from repo root:
  python scripts/r8_feature_loo.py \\
      --batch-dir runs/20260503_181558_dceacd_multiseed \\
      --out figures/20260503_181558_dceacd_multiseed_t13_feature_loo
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
    audit_llm_alignment,
    encode_prompt_to_w_stated,
    fit_bayesian_irl,
    DEFAULT_W_STATED_INTENT,
    parse_menu_run,
)

ROOT = Path(__file__).resolve().parent.parent
RE_RUN = re.compile(
    r"^seed(?P<seed>\d{3})(?:_R(?P<replica>\d+))?_(?P<label>[a-z][\w]*)\.jsonl$"
)


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
    ap.add_argument("--rope-width", type=float, default=0.25)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: only fit the first three (seed, model) pairs.",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    runs = discover(args.batch_dir)
    if args.quick:
        runs = runs[:3]
        print(f"[t13] --quick: limiting to first {len(runs)} runs")

    d_full = len(OUTCOME_FEATURE_NAMES)
    rows: list[dict[str, object]] = []

    # Pre-encode the full-d stated reward once; we slice per dropout.
    w_stated_full = encode_prompt_to_w_stated(
        DEFAULT_W_STATED_INTENT,
        feature_names=OUTCOME_FEATURE_NAMES,
        normalize=False,
    )

    for seed, label, path in runs:
        parsed = parse_menu_run(
            path, feature_seed=args.feature_seed, n_samples=args.n_samples
        )
        features_full = parsed.features  # (T, K, d)
        chosen = parsed.chosen

        # Fit full-d baseline first.
        print(f"[t13] seed={seed:03d} model={label} drop=NONE fitting…")
        post_full = fit_bayesian_irl(
            features=features_full,
            chosen=chosen,
            feature_names=OUTCOME_FEATURE_NAMES,
            prior_sigma=args.prior_sigma,
            draws=args.nuts_draws,
            tune=args.nuts_tune,
            chains=args.nuts_chains,
            seed=args.nuts_seed,
            progressbar=False,
        )
        w_full = post_full.w_mean.copy()
        align_full = audit_llm_alignment(
            post_full, w_stated_full / np.linalg.norm(w_stated_full),
            rope_width=args.rope_width,
        )
        # Baseline row.
        rows.append({
            "seed": seed, "model": label, "dropped_feature": "NONE",
            "dropped_idx": -1,
            "cos_vs_full_5dim_subset": float("nan"),
            "significantly_misaligned": bool(align_full.significantly_misaligned),
            "w_norm_5dim_subset": float(np.linalg.norm(w_full)),
        })

        # Now the 6 LOO fits.
        for drop_idx, drop_name in enumerate(OUTCOME_FEATURE_NAMES):
            keep_mask = np.ones(d_full, dtype=bool)
            keep_mask[drop_idx] = False
            feature_names_kept = tuple(
                f for j, f in enumerate(OUTCOME_FEATURE_NAMES) if j != drop_idx
            )
            features_loo = features_full[:, :, keep_mask]  # (T, K, d-1)
            w_stated_loo = w_stated_full[keep_mask]
            w_stated_loo_norm = w_stated_loo / max(
                float(np.linalg.norm(w_stated_loo)), 1e-12
            )

            print(f"[t13] seed={seed:03d} model={label} drop={drop_name} fitting…")
            post = fit_bayesian_irl(
                features=features_loo,
                chosen=chosen,
                feature_names=feature_names_kept,
                prior_sigma=args.prior_sigma,
                draws=args.nuts_draws,
                tune=args.nuts_tune,
                chains=args.nuts_chains,
                seed=args.nuts_seed,
                progressbar=False,
            )
            w_loo = post.w_mean.copy()  # (d-1,)

            # Cosine of LOO recovery against the matching 5 dims of full.
            cos_5dim = cosine(w_loo, w_full[keep_mask])

            # Reclassification under 5-dim ROPE.
            align_loo = audit_llm_alignment(
                post, w_stated_loo_norm, rope_width=args.rope_width
            )

            row = {
                "seed": seed, "model": label,
                "dropped_feature": drop_name, "dropped_idx": drop_idx,
                "cos_vs_full_5dim_subset": cos_5dim,
                "significantly_misaligned": bool(align_loo.significantly_misaligned),
                "w_norm_5dim_subset": float(np.linalg.norm(w_loo)),
            }
            for name, val in zip(feature_names_kept, w_loo):
                row[f"w_{name}"] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_seed.csv", index=False)

    # Summary.
    lines: list[str] = []
    lines.append("# T1.3 — Feature leave-one-out")
    lines.append("")
    lines.append(
        f"NUTS re-fits dropping one feature at a time across "
        f"{len(runs)} (seed, model) pairs.  Cosines compare the "
        f"recovered 5-dim weight to the matching 5 dims of the "
        f"full-d baseline."
    )
    lines.append("")
    lines.append("## Direction stability per dropped feature")
    lines.append("")
    lines.append("| dropped | model | n | median cos | min cos | reclassifications |")
    lines.append("|---|---|---:|---:|---:|---:|")
    loo = df[df["dropped_feature"] != "NONE"]
    base = df[df["dropped_feature"] == "NONE"].set_index(["seed", "model"])
    for (drop, model), s in loo.groupby(["dropped_feature", "model"]):
        # Reclassifications: per (seed, model) compare baseline misalign vs LOO.
        merged = s.merge(
            base[["significantly_misaligned"]].rename(
                columns={"significantly_misaligned": "mis_full"}
            ),
            on=["seed", "model"],
        )
        reclass = int(
            (merged["significantly_misaligned"] != merged["mis_full"]).sum()
        )
        lines.append(
            f"| {drop} | {model} | {len(s)} | "
            f"{s['cos_vs_full_5dim_subset'].median():.4f} | "
            f"{s['cos_vs_full_5dim_subset'].min():.4f} | {reclass}/{len(s)} |"
        )
    lines.append("")
    lines.append(
        "Pre-registered decision rule (TUNING_PREREG §T1.3): robust "
        "per-drop if median cosine ≥ 0.90 and reclassifications ≤ 4 "
        "(of 20).  Honest finding expected for drop=anti_pobreza."
    )
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[t13] per-seed -> {args.out / 'per_seed.csv'}")
    print(f"[t13] summary  -> {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
