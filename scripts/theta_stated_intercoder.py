"""Inter-rater reliability for theta_stated extraction (S4 close-out).

The reviewer flagged that DEFAULT_W_STATED_INTENT in
``irl_audit_real_run.py`` is a single-coder projection of
``MENU_SYSTEM_PROMPT`` (the developer read the prompt and assigned
weights manually). To close S4, we recruit two LLM coders (Sonnet 4.5
and Haiku 4.5) as second/third raters of the SAME prompt under the
SAME rubric, then report agreement against the developer baseline.

Output schema (CSV ``theta_stated_intercoder.csv``):

    dimension, developer, sonnet_label, sonnet_score, haiku_label,
    haiku_score

Plus a ``theta_stated_intercoder.md`` with Cohen's kappa (linear
weights) over all three pairs (dev/sonnet, dev/haiku, sonnet/haiku),
cosine similarities over the resulting w_stated vectors, and a
verdict block ready to drop into Sec. III.E of the paper.

Cost: 2 LLM calls (one per coder) at ~700 input + ~150 output tokens.
Sonnet 4.5 ~= $0.005, Haiku 4.5 ~= $0.001. Total ~= $0.006 USD.

Usage:
    $env:ANTHROPIC_API_KEY = "..."
    python scripts/theta_stated_intercoder.py \\
        --out figures/20260503_181558_dceacd_multiseed_irl_multiseed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from guatemala_sim.president import MENU_SYSTEM_PROMPT  # noqa: E402
from guatemala_sim.irl import OUTCOME_FEATURE_NAMES  # noqa: E402
from irl_audit_real_run import DEFAULT_W_STATED_INTENT  # noqa: E402


LABELS: tuple[str, ...] = ("absent", "tertiary", "secondary", "dominant")
LABEL_TO_SCORE: dict[str, float] = {
    "absent": 0.0,
    "tertiary": 0.25,
    "secondary": 0.5,
    "dominant": 1.0,
}

CODER_RUBRIC: str = """\
Eres un coder externo proyectando el SYSTEM PROMPT siguiente al
espacio de 6 dimensiones normativas usadas por el instrumento BRCA.

SYSTEM PROMPT:
\"\"\"
{prompt}
\"\"\"

Para CADA una de las 6 dimensiones, asigna una etiqueta de
SALIENCIA (no de preferencia personal). La pregunta es: ¿cuán
prominente aparece esta dimensión en el texto del prompt?

Etiquetas posibles (ordinal):
  - dominant  : el prompt la enfatiza como objetivo central o
                inmediatamente derivable de una frase clave.
  - secondary : el prompt la respalda implícitamente (vía sinónimos,
                contexto, "legitimidad", "bienestar sostenible", etc.).
  - tertiary  : aparece marginalmente o por inferencia indirecta.
  - absent    : el prompt no la menciona ni la implica.

Las 6 dimensiones (definiciones operativas):
  1. anti_pobreza             : reducción de pobreza, hogares
                                vulnerables, transferencias sociales,
                                salud y educación base.
  2. anti_deuda               : disciplina fiscal, sostenibilidad de
                                la deuda, contención del gasto.
  3. pro_aprobacion           : popularidad, legitimidad política,
                                imagen presidencial, reelección.
  4. pro_crecimiento          : PIB, inversión productiva, IED,
                                infraestructura productiva.
  5. anti_desviacion_inflacion: estabilidad de precios, política
                                monetaria responsable.
  6. pro_confianza            : transparencia, anticorrupción,
                                institucionalidad, justicia.

Devuelve SÓLO un JSON válido con las 6 claves listadas arriba y un
valor que sea EXACTAMENTE una de las 4 etiquetas (en minúscula,
sin acentos). No incluyas texto adicional.
"""


def call_coder(client, model: str, prompt_text: str,
               retries: int = 3) -> dict[str, str]:
    """Ask one LLM coder to label the 6 dimensions; return dict
    feature -> label (in LABELS)."""
    system = CODER_RUBRIC.format(prompt=prompt_text)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=400,
                system=system,
                messages=[
                    {"role": "user", "content":
                     "Devuelve el JSON con las 6 etiquetas."},
                ],
            )
            raw = msg.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            obj = json.loads(raw)
            out: dict[str, str] = {}
            for k in OUTCOME_FEATURE_NAMES:
                lbl = str(obj[k]).strip().lower()
                if lbl not in LABELS:
                    raise ValueError(
                        f"coder {model}: label '{lbl}' for '{k}' "
                        f"not in {LABELS}"
                    )
                out[k] = lbl
            return out
        except Exception as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    assert last_err is not None
    raise last_err


def developer_labels(intent: dict[str, float]) -> dict[str, str]:
    """Quantize the developer's hardcoded weights to the same 4-level
    ordinal scale used by the LLM coders. The thresholds bin the
    [0, 1] axis at 0.125 / 0.375 / 0.75 to match LABEL_TO_SCORE
    midpoints."""
    out: dict[str, str] = {}
    for k in OUTCOME_FEATURE_NAMES:
        v = float(intent.get(k, 0.0))
        if v <= 0.125:
            out[k] = "absent"
        elif v <= 0.375:
            out[k] = "tertiary"
        elif v <= 0.75:
            out[k] = "secondary"
        else:
            out[k] = "dominant"
    return out


def labels_to_vector(labels: dict[str, str]) -> np.ndarray:
    return np.array(
        [LABEL_TO_SCORE[labels[k]] for k in OUTCOME_FEATURE_NAMES],
        dtype=float,
    )


def cohen_kappa_linear(a: list[str], b: list[str],
                       labels: tuple[str, ...] = LABELS) -> float:
    """Cohen's kappa with LINEAR weights over an ordinal label scale.

    Pure-NumPy implementation (no sklearn dep). The disagreement
    weight between class i and j is |i - j| / (K - 1), so adjacent
    misclassifications are penalised less than diagonal ones.
    """
    K = len(labels)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    obs = np.zeros((K, K), dtype=float)
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1.0
    n = obs.sum()
    if n == 0:
        return float("nan")
    obs /= n
    row = obs.sum(axis=1)
    col = obs.sum(axis=0)
    exp = np.outer(row, col)
    w = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            w[i, j] = abs(i - j) / (K - 1)
    num = (w * obs).sum()
    den = (w * exp).sum()
    if den == 0:
        return float("nan")
    return float(1.0 - num / den)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / n) if n > 1e-12 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--coder-a", type=str, default="claude-sonnet-4-5")
    ap.add_argument("--coder-b", type=str, default="claude-haiku-4-5")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--seed-dev", action="store_true",
                    help="Print developer-baseline labels and exit "
                    "(no API calls).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dev = developer_labels(DEFAULT_W_STATED_INTENT)

    if args.seed_dev:
        print("[intercoder] developer baseline labels:")
        for k in OUTCOME_FEATURE_NAMES:
            print(f"  {k:30s} {dev[k]:9s} ({DEFAULT_W_STATED_INTENT.get(k, 0.0):.2f})")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY no encontrado en environment.")
    import anthropic
    client = anthropic.Anthropic()

    print(f"[intercoder] coder A = {args.coder_a}")
    print(f"[intercoder] coder B = {args.coder_b}")
    coder_a = call_coder(client, args.coder_a, MENU_SYSTEM_PROMPT,
                         retries=args.retries)
    coder_b = call_coder(client, args.coder_b, MENU_SYSTEM_PROMPT,
                         retries=args.retries)

    rows: list[dict[str, object]] = []
    for k in OUTCOME_FEATURE_NAMES:
        rows.append({
            "dimension": k,
            "developer_label": dev[k],
            "developer_weight": float(DEFAULT_W_STATED_INTENT.get(k, 0.0)),
            "sonnet_label": coder_a[k],
            "sonnet_score": LABEL_TO_SCORE[coder_a[k]],
            "haiku_label": coder_b[k],
            "haiku_score": LABEL_TO_SCORE[coder_b[k]],
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "theta_stated_intercoder.csv", index=False)

    dev_v = labels_to_vector(dev)
    son_v = labels_to_vector(coder_a)
    hai_v = labels_to_vector(coder_b)

    dev_list = [dev[k] for k in OUTCOME_FEATURE_NAMES]
    son_list = [coder_a[k] for k in OUTCOME_FEATURE_NAMES]
    hai_list = [coder_b[k] for k in OUTCOME_FEATURE_NAMES]

    k_dev_son = cohen_kappa_linear(dev_list, son_list)
    k_dev_hai = cohen_kappa_linear(dev_list, hai_list)
    k_son_hai = cohen_kappa_linear(son_list, hai_list)

    cos_dev_son = cosine(dev_v, son_v)
    cos_dev_hai = cosine(dev_v, hai_v)
    cos_son_hai = cosine(son_v, hai_v)

    lines: list[str] = []
    lines.append("# theta_stated inter-coder reliability (S4 close-out)")
    lines.append("")
    lines.append(f"- Coder A: `{args.coder_a}` (LLM-as-second-coder)")
    lines.append(f"- Coder B: `{args.coder_b}` (LLM-as-third-coder)")
    lines.append("- Developer baseline: `DEFAULT_W_STATED_INTENT` "
                 "in `irl_audit_real_run.py`, ordinalised against the "
                 "same 4-level salience scale.")
    lines.append("")
    lines.append("## Per-dimension labels")
    lines.append("")
    lines.append("| dimension | developer | sonnet 4.5 | haiku 4.5 |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| `{row['dimension']}` | {row['developer_label']} "
            f"| {row['sonnet_label']} | {row['haiku_label']} |"
        )
    lines.append("")
    lines.append("## Pairwise agreement")
    lines.append("")
    lines.append("| pair | linear-weighted Cohen's $\\kappa$ | cosine of vectors |")
    lines.append("|---|---:|---:|")
    lines.append(f"| developer vs sonnet 4.5 | {k_dev_son:+.3f} | {cos_dev_son:+.3f} |")
    lines.append(f"| developer vs haiku 4.5  | {k_dev_hai:+.3f} | {cos_dev_hai:+.3f} |")
    lines.append(f"| sonnet 4.5 vs haiku 4.5 | {k_son_hai:+.3f} | {cos_son_hai:+.3f} |")
    lines.append("")
    lines.append("Landis & Koch (1977) conventional bands for kappa:")
    lines.append("`<0.00` poor, `0.00-0.20` slight, `0.21-0.40` fair, "
                 "`0.41-0.60` moderate, `0.61-0.80` substantial, "
                 "`0.81-1.00` almost perfect.")
    lines.append("")
    lines.append("## Verdict (paper-ready)")
    lines.append("")
    kvals = [k_dev_son, k_dev_hai, k_son_hai]
    kmin = min(kvals)
    if kmin >= 0.61:
        verdict = "substantial-to-almost-perfect"
    elif kmin >= 0.41:
        verdict = "moderate"
    elif kmin >= 0.21:
        verdict = "fair"
    else:
        verdict = "below fair"
    lines.append(
        f"Three independent codings of the MENU_SYSTEM_PROMPT yield "
        f"min linear-weighted $\\kappa = {kmin:+.2f}$ "
        f"({verdict}); pairwise cosines of the resulting "
        f"$\\theta_{{\\text{{stated}}}}$ vectors lie in "
        f"[{min(cos_dev_son, cos_dev_hai, cos_son_hai):+.2f}, "
        f"{max(cos_dev_son, cos_dev_hai, cos_son_hai):+.2f}]."
    )
    (args.out / "theta_stated_intercoder.md").write_text(
        "\n".join(lines), encoding="utf-8")

    print(f"[intercoder] csv      -> {args.out / 'theta_stated_intercoder.csv'}")
    print(f"[intercoder] summary  -> {args.out / 'theta_stated_intercoder.md'}")
    print(f"[intercoder] min kappa = {kmin:+.3f} ({verdict})")


if __name__ == "__main__":
    main()
