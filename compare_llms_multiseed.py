"""Runner multi-seed Anthropic vs. OpenAI con réplicas opcionales.

Itera sobre una lista de seeds; para cada (seed, modelo) corre `--replicas`
veces. Mismo seed → mismos shocks (la dinámica del mundo es determinista
dado el seed); las diferencias entre réplicas vienen exclusivamente del
sampler estocástico del LLM (softmax sobre tokens, T~1).

Esto habilita ICC (test-retest dentro de modelo): cuánto de la varianza
del comportamiento del decisor se debe al modelo (entre seeds, sustantivo)
vs. al sampler (intra-seed, ruido).

Uso:

    # comparativa básica (5 seeds, 1 réplica): chequeo de pipeline
    python compare_llms_multiseed.py --seeds 11,12,13,14,15 --turnos 8

    # robusta (30 seeds, 1 réplica): tests pareados + mixed-effects
    python compare_llms_multiseed.py --seeds-from 1 --seeds-to 30 --turnos 8 \\
        --continuar-si-falla

    # con réplicas (30 seeds × 3 réplicas): además ICC
    python compare_llms_multiseed.py --seeds-from 1 --seeds-to 30 --turnos 8 \\
        --replicas 3 --continuar-si-falla

Costo aproximado @ Haiku 4.5 + gpt-4o-mini, 8 turnos:
  - 30 seeds × 1 réplica:  ≈ USD 25-35,  ~30 min
  - 30 seeds × 3 réplicas: ≈ USD 75-105, ~90 min
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from guatemala_sim.logging_ import new_run_id
from guatemala_sim.multiseed import SeedRun, analyze
from guatemala_sim.president import ClaudePresidente
from guatemala_sim.president_openai import GPTPresidente

from compare_llms import _correr, _nueva_mundo  # type: ignore[reportPrivateUsage]


ROOT = Path(__file__).resolve().parent


def _parse_seeds(args) -> list[int]:
    if args.seeds_from is not None and args.seeds_to is not None:
        return list(range(args.seeds_from, args.seeds_to + 1))
    return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="11,12,13,14,15",
                    help="lista CSV de seeds (ej '11,12,13')")
    ap.add_argument("--seeds-from", type=int, default=None)
    ap.add_argument("--seeds-to", type=int, default=None)
    ap.add_argument("--turnos", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=1,
                    help="N corridas con mismo (seed, modelo). >1 habilita ICC.")
    ap.add_argument("--claude-modelo", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--openai-modelo", type=str, default="gpt-4o-mini")
    ap.add_argument("--skip-claude", action="store_true")
    ap.add_argument("--skip-openai", action="store_true")
    ap.add_argument("--continuar-si-falla", action="store_true")
    ap.add_argument("--menu-mode", action="store_true",
                    help=("modo menu-choice (LLM elige UNO de 5 presupuestos). "
                          "JSONL incluye chosen_index para alimentar el "
                          "pipeline de IRL bayesiano post-corrida."))
    args = ap.parse_args()

    seeds = _parse_seeds(args)
    batch_id = new_run_id() + "_multiseed"
    runs_dir = ROOT / "runs" / batch_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "figures" / f"{batch_id}_analysis"
    err_log = runs_dir / "errores.log"

    n_models = int(not args.skip_claude) + int(not args.skip_openai)
    n_total = len(seeds) * args.replicas * n_models
    print(f"[multiseed] batch={batch_id}")
    print(f"[multiseed] seeds={seeds}  turnos={args.turnos}  replicas={args.replicas}")
    print(f"[multiseed] runs → {runs_dir}")
    print(f"[multiseed] análisis → {out_dir}")
    print(f"[multiseed] total corridas planeadas: {n_total}")
    if args.menu_mode:
        print(f"[multiseed] menu-mode ACTIVO — JSONL tendrá chosen_index por turno")
    print()

    runs: list[SeedRun] = []
    t_start = time.time()
    n_done = 0
    n_fail = 0

    def _ejecutar(seed: int, replica: int, label: str, build_dm) -> None:
        nonlocal n_done, n_fail
        n_done += 1
        suffix = f"_R{replica}" if args.replicas > 1 else ""
        run_id = f"{batch_id}/seed{seed:03d}{suffix}_{label.lower()}"
        try:
            rng, state, agentes, territory = _nueva_mundo(seed)
            dm = build_dm()
            p = _correr(
                f"{label}/{getattr(dm, 'model', '?')}", dm, territory, agentes,
                rng, state, args.turnos, run_id,
                menu_mode=args.menu_mode,
            )
            runs.append(SeedRun(seed=seed, model_label=label,
                                log_path=p, replica=replica))
            print(f"[multiseed] ({n_done}/{n_total}) "
                  f"seed={seed} R={replica} {label} OK")
        except Exception as e:
            n_fail += 1
            msg = (f"seed={seed} R={replica} {label} FAIL: "
                   f"{type(e).__name__}: {e}")
            print(f"[multiseed] ({n_done}/{n_total}) {msg}")
            err_log.open("a", encoding="utf-8").write(
                msg + "\n" + traceback.format_exc() + "\n---\n"
            )
            if not args.continuar_si_falla:
                print("[multiseed] abortando. usá --continuar-si-falla para tolerar.")
                sys.exit(1)

    for seed in seeds:
        for replica in range(args.replicas):
            if not args.skip_claude:
                _ejecutar(seed, replica, "Claude",
                          lambda: ClaudePresidente(model=args.claude_modelo))
            if not args.skip_openai:
                _ejecutar(seed, replica, "OpenAI",
                          lambda: GPTPresidente(model=args.openai_modelo))

    elapsed = time.time() - t_start
    print()
    print(f"[multiseed] {len(runs)}/{n_total} corridas exitosas en "
          f"{elapsed/60:.1f} min ({n_fail} fallos)")

    if len(runs) < 2:
        print("[multiseed] insuficiente para análisis (≥2 corridas requeridas).")
        return

    print(f"[multiseed] analizando…")
    paths = analyze(runs, out_dir, model_a="Claude", model_b="OpenAI")
    print(f"[multiseed] reporte: {paths['summary']}")
    for k, v in paths.items():
        print(f"           - {k}: {v}")


if __name__ == "__main__":
    main()
