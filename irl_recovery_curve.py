"""Curva de recovery del IRL bayesiano: error MLE vs tamaño de muestra.

Genera la pre-Figura 1 del paper. Es la validación canónica del método:
con datos sintéticos de ground-truth conocido, mostramos que el MLE
recupera w* con error que decrece como ~ 1/√N.

Esto se hace ANTES de tocar PyMC para tener un baseline frecuentista
contra el cual comparar el posterior bayesiano. Si el MLE no recupera
w*, hay un bug en el setup matemático y no tiene sentido seguir.

Outputs (en figures/irl_recovery/ por default):
    recovery_curve.png        — figura de 2 paneles
    recovery_data.csv         — datos crudos del barrido
    recovery_summary.md       — resumen en markdown
    true_w.txt                — el w* usado (para reproducibilidad)

Uso:
    python irl_recovery_curve.py
    python irl_recovery_curve.py --n-replications 20 --output figures/irl_recovery_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from guatemala_sim.irl import OUTCOME_FEATURE_NAMES, run_recovery_sweep


# w* "realista": un LLM con preferencias moderadas pero claras.
#   componente               | nombre                       | peso
#   0  anti_pobreza          | reducir pobreza es prioritario      |  1.5
#   1  anti_deuda            | importa moderadamente               |  0.5
#   2  pro_aprobacion        | importa la legitimidad              |  1.0
#   3  pro_crecimiento       | crecimiento como prioridad alta     |  1.2
#   4  anti_desviacion_infl  | inflación cerca del target          |  0.8
#   5  pro_confianza         | confianza institucional             |  0.7
DEFAULT_TRUE_W = np.array([1.5, 0.5, 1.0, 1.2, 0.8, 0.7])


DEFAULT_SAMPLE_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--n-replications", type=int, default=10,
        help="réplicas independientes por N (con seeds distintos). Default 10.",
    )
    p.add_argument(
        "--n-candidates", type=int, default=5,
        help="tamaño del menú K (default 5, igual que en candidates.py).",
    )
    p.add_argument(
        "--sample-sizes", type=int, nargs="+", default=DEFAULT_SAMPLE_SIZES,
        help=f"valores de N a barrer. Default {DEFAULT_SAMPLE_SIZES}.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("figures/irl_recovery"),
        help="directorio de salida.",
    )
    p.add_argument(
        "--base-seed", type=int, default=0,
        help="semilla base para reproducibilidad.",
    )
    return p.parse_args()


def plot_recovery_curve(df: pd.DataFrame, true_w: np.ndarray, output_path: Path) -> None:
    """Genera la figura de 2 paneles: RMSE vs N (log-log) y cosine sim vs N."""
    summary = (
        df.groupby("N")
        .agg(
            rmse_median=("rmse", "median"),
            rmse_q25=("rmse", lambda s: s.quantile(0.25)),
            rmse_q75=("rmse", lambda s: s.quantile(0.75)),
            cos_median=("cosine_similarity", "median"),
            cos_q25=("cosine_similarity", lambda s: s.quantile(0.25)),
            cos_q75=("cosine_similarity", lambda s: s.quantile(0.75)),
        )
        .reset_index()
        .sort_values("N")
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    # Panel 1: RMSE vs N (log-log) con curva de referencia 1/sqrt(N)
    ax = axes[0]
    Ns = summary["N"].to_numpy()
    ax.plot(Ns, summary["rmse_median"], "o-", color="C0", label="RMSE (median)", lw=2)
    ax.fill_between(Ns, summary["rmse_q25"], summary["rmse_q75"],
                    alpha=0.25, color="C0", label="IQR")

    # Línea de referencia C/sqrt(N) anclada al primer punto
    ref_C = float(summary["rmse_median"].iloc[0]) * float(np.sqrt(Ns[0]))
    ref_curve = ref_C / np.sqrt(Ns)
    ax.plot(Ns, ref_curve, "--", color="gray", alpha=0.7, label=r"$\propto 1/\sqrt{N}$ (ref.)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of turns N")
    ax.set_ylabel("RMSE($\\hat{w}$, $w^*$)")
    ax.set_title("MLE recovery error vs sample size")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: cosine similarity vs N (semi-log x)
    ax = axes[1]
    ax.plot(Ns, summary["cos_median"], "o-", color="C2", label="cos sim (median)", lw=2)
    ax.fill_between(Ns, summary["cos_q25"], summary["cos_q75"],
                    alpha=0.25, color="C2", label="IQR")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="perfect recovery")
    ax.axhline(0.95, color="red", ls=":", alpha=0.5, label="threshold 0.95")
    ax.set_xscale("log")
    ax.set_xlabel("Number of turns N")
    ax.set_ylabel(r"cosine similarity($\hat{w}$, $w^*$)")
    ax.set_title("Recovery of preference direction")
    ax.set_ylim(min(0.0, summary["cos_q25"].min() - 0.05), 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"IRL recovery curve — d={len(true_w)}, K=5 candidates, "
        f"{int(df['replication'].nunique())} replications per N",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, true_w: np.ndarray, output_path: Path) -> None:
    """Resumen en markdown con tabla por N."""
    summary = (
        df.groupby("N")
        .agg(
            n_reps=("replication", "count"),
            rmse_median=("rmse", "median"),
            rmse_iqr_lo=("rmse", lambda s: s.quantile(0.25)),
            rmse_iqr_hi=("rmse", lambda s: s.quantile(0.75)),
            cos_median=("cosine_similarity", "median"),
            cos_min=("cosine_similarity", "min"),
            norm_ratio_median=("norm_ratio", "median"),
        )
        .reset_index()
        .sort_values("N")
    )

    lines = [
        "# IRL Recovery Curve — Resumen",
        "",
        "## Setup",
        "",
        f"- **w\\* verdadero**: `{[round(float(x), 3) for x in true_w]}`",
        f"- **Dimensiones de bienestar (φ)**: `{list(OUTCOME_FEATURE_NAMES)}`",
        f"- **K candidatos**: 5 (random gaussian, anclados en candidato 0)",
        f"- **Réplicas por N**: {int(df['replication'].nunique())}",
        "",
        "## Resultados",
        "",
        "| N | reps | RMSE mediana | RMSE IQR | cos sim mediana | cos sim mín | norm_ratio mediana |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {int(r['N'])} | {int(r['n_reps'])} | "
            f"{r['rmse_median']:.4f} | "
            f"[{r['rmse_iqr_lo']:.4f}, {r['rmse_iqr_hi']:.4f}] | "
            f"{r['cos_median']:.4f} | "
            f"{r['cos_min']:.4f} | "
            f"{r['norm_ratio_median']:.3f} |"
        )

    lines += [
        "",
        "## Lectura",
        "",
        "- **RMSE mediana** debe decrecer con N. Si la pendiente en log-log es ≈ -0.5, "
        "el método tiene la convergencia esperada O(1/√N).",
        "- **cos sim** debe acercarse a 1 con N grande. Umbral típico para "
        "considerar recovery exitoso: 0.95.",
        "- **norm_ratio** debe acercarse a 1; valores muy distintos indican que "
        "el MLE recupera la *dirección* pero no la *magnitud* de las "
        "preferencias (esperable con N pequeño).",
        "",
        "## Validez del setup IRL",
        "",
        "Si esta tabla muestra cos sim ≥ 0.95 con N ≥ 1000 y RMSE decreciente "
        "monótono, el setup matemático del Boltzmann likelihood + MLE es "
        "correcto y podemos pasar a `bayesian_irl.py` con confianza de que "
        "los problemas de allá adelante son del modelo PyMC, no del IRL.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[recovery] true_w = {DEFAULT_TRUE_W}")
    print(f"[recovery] sample sizes = {args.sample_sizes}")
    print(f"[recovery] {args.n_replications} replicas por N")
    print(f"[recovery] output = {args.output}")
    print(f"[recovery] corriendo {len(args.sample_sizes) * args.n_replications} ajustes MLE...")

    df = run_recovery_sweep(
        true_w=DEFAULT_TRUE_W,
        sample_sizes=args.sample_sizes,
        n_replications=args.n_replications,
        n_candidates=args.n_candidates,
        base_seed=args.base_seed,
    )

    # Persistir
    csv_path = args.output / "recovery_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"[recovery] CSV -> {csv_path}")

    fig_path = args.output / "recovery_curve.png"
    plot_recovery_curve(df, DEFAULT_TRUE_W, fig_path)
    print(f"[recovery] figura -> {fig_path}")

    md_path = args.output / "recovery_summary.md"
    write_summary(df, DEFAULT_TRUE_W, md_path)
    print(f"[recovery] resumen -> {md_path}")

    true_w_path = args.output / "true_w.json"
    true_w_path.write_text(
        json.dumps(
            {
                "true_w": DEFAULT_TRUE_W.tolist(),
                "feature_names": list(OUTCOME_FEATURE_NAMES),
                "sample_sizes": args.sample_sizes,
                "n_replications": args.n_replications,
                "n_candidates": args.n_candidates,
                "base_seed": args.base_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[recovery] config -> {true_w_path}")

    # Imprimir tabla resumen por N
    summary = df.groupby("N").agg(
        rmse=("rmse", "median"),
        cos=("cosine_similarity", "median"),
        norm_ratio=("norm_ratio", "median"),
    ).round(4)
    print("\n[recovery] resumen por N:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
