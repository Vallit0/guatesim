"""Runner multi-seed × multi-modelo con réplicas opcionales.

Itera sobre `(seed, replica, modelo)` y loguea cada corrida a JSONL.
Cada modelo se resuelve via `guatemala_sim.models_registry`, así que
agregar Gemini, DeepSeek, Llama o Qwen no requiere cambios en este
runner — sólo agregarlos al registry y pasarlos en `--models`.

El análisis pareado de fin de batch (Wilcoxon + Holm + mixed-effects)
sigue siendo binario: necesita un `model_a` y un `model_b`. Cuando
corrés con >2 modelos, pasá `--reference-model` para fijar el lado A
y se reportan tests pareados contra ese modelo para todos los demás
(o el primero, si no especificás).

Uso:

    # Comparativa básica (5 seeds, 1 réplica, default Claude+GPT):
    python compare_llms_multiseed.py --seeds 11,12,13,14,15 --turnos 8

    # Cross-vendor completo (paper-grade, 20 seeds × 6 modelos):
    python compare_llms_multiseed.py --seeds-from 1 --seeds-to 20 --turnos 8 \\
        --models claude-haiku-4-5,gpt-4o-mini,gemini-2-0-flash,\\
deepseek-v3,llama-3-1-405b,qwen-2-5-72b \\
        --reference-model claude-haiku-4-5 --menu-mode --continuar-si-falla

    # Backward-compat: --skip-claude / --skip-openai siguen funcionando si
    # NO pasás --models.

Costos aproximados @ 8 turnos × 20 seeds, menu-mode (input ~3k tok, output ~1k tok):
    claude-haiku-4-5      ~ USD  10
    gpt-4o-mini           ~ USD   8
    gemini-2-0-flash      ~ USD   3-5
    deepseek-v3           ~ USD   2-4
    llama-3-1-405b (Together) ~ USD 25-35
    qwen-2-5-72b (DashScope)  ~ USD  6-10
    ----
    cross-vendor total    ~ USD  60-90, ~3-4 h en serie.
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
from guatemala_sim.models_registry import (
    ModelSpec,
    get_spec,
    make_decision_maker,
    parse_models_csv,
)
from guatemala_sim.multiseed import SeedRun, analyze

from compare_llms import _correr, _nueva_mundo  # type: ignore[reportPrivateUsage]


ROOT = Path(__file__).resolve().parent


def _parse_seeds(args) -> list[int]:
    if args.seeds_from is not None and args.seeds_to is not None:
        return list(range(args.seeds_from, args.seeds_to + 1))
    return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]


def _resolve_models(args) -> list[ModelSpec]:
    """Resuelve qué modelos correr.

    Prioridad:
      1. `--models <csv>` si se pasó (modo declarativo nuevo).
      2. defaults Claude + OpenAI con respeto a `--skip-claude/--skip-openai`
         + `--claude-modelo/--openai-modelo` (modo backward-compat).
    """
    if args.models:
        return parse_models_csv(args.models)

    # Modo legacy: armar specs ad-hoc respetando --skip-* y --*-modelo
    specs: list[ModelSpec] = []
    if not args.skip_claude:
        # Si el usuario sobreescribió el modelo, hacemos un ModelSpec ad-hoc;
        # si usa el default, tomamos del registry directamente.
        if args.claude_modelo == "claude-haiku-4-5-20251001":
            specs.append(get_spec("claude-haiku-4-5"))
        else:
            specs.append(ModelSpec(
                model_id=f"claude-{args.claude_modelo}",
                display_name=f"Claude/{args.claude_modelo}",
                provider="anthropic",
                model=args.claude_modelo,
                env_key="ANTHROPIC_API_KEY",
                family="anthropic",
            ))
    if not args.skip_openai:
        if args.openai_modelo == "gpt-4o-mini":
            specs.append(get_spec("gpt-4o-mini"))
        else:
            specs.append(ModelSpec(
                model_id=f"openai-{args.openai_modelo}",
                display_name=f"OpenAI/{args.openai_modelo}",
                provider="openai",
                model=args.openai_modelo,
                env_key="OPENAI_API_KEY",
                family="openai",
            ))
    return specs


def _label_for(spec: ModelSpec) -> str:
    """Etiqueta corta y estable para JSONL paths y reportes pareados.

    Convertimos guiones a guion-bajo para que el filename siga siendo
    legible y compatible con paths Windows. La etiqueta debe ser única
    entre los modelos del batch — el registry ya lo garantiza por
    `model_id`.
    """
    return spec.model_id.replace("-", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="11,12,13,14,15",
                    help="lista CSV de seeds (ej '11,12,13')")
    ap.add_argument("--seeds-from", type=int, default=None)
    ap.add_argument("--seeds-to", type=int, default=None)
    ap.add_argument("--turnos", type=int, default=8)
    ap.add_argument("--replicas", type=int, default=1,
                    help="N corridas con mismo (seed, modelo). >1 habilita ICC.")
    ap.add_argument(
        "--models", type=str, default="",
        help=("CSV de model_ids del registry (ej "
              "'claude-haiku-4-5,gpt-4o-mini,gemini-2-0-flash'). "
              "Si está, ignora --skip-claude/--skip-openai/--*-modelo. "
              "Ver guatemala_sim/models_registry.py para la lista completa."),
    )
    ap.add_argument("--reference-model", type=str, default="",
                    help=("model_id contra el que se pareará Wilcoxon en el "
                          "análisis. Default: el primero de --models."))
    # --- legacy flags (backward-compat para scripts existentes) ---
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
    specs = _resolve_models(args)
    if len(specs) < 1:
        print("[multiseed] sin modelos a correr (todo skipped). saliendo.")
        sys.exit(2)

    batch_id = new_run_id() + "_multiseed"
    runs_dir = ROOT / "runs" / batch_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "figures" / f"{batch_id}_analysis"
    err_log = runs_dir / "errores.log"

    n_total = len(seeds) * args.replicas * len(specs)
    print(f"[multiseed] batch={batch_id}")
    print(f"[multiseed] seeds={seeds}  turnos={args.turnos}  replicas={args.replicas}")
    print(f"[multiseed] modelos ({len(specs)}):")
    for s in specs:
        print(f"           - {s.model_id:25s}  {s.family:10s}  {s.display_name}")
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

    def _ejecutar(seed: int, replica: int, spec: ModelSpec) -> None:
        nonlocal n_done, n_fail
        n_done += 1
        label_fs = _label_for(spec)
        suffix = f"_R{replica}" if args.replicas > 1 else ""
        run_id = f"{batch_id}/seed{seed:03d}{suffix}_{label_fs}"
        try:
            rng, state, agentes, territory = _nueva_mundo(seed)
            dm = make_decision_maker(spec.model_id)
            p = _correr(
                f"{spec.display_name}", dm, territory, agentes,
                rng, state, args.turnos, run_id,
                menu_mode=args.menu_mode,
            )
            runs.append(SeedRun(
                seed=seed,
                model_label=spec.model_id,  # estable, machacable en CSVs
                log_path=p,
                replica=replica,
            ))
            print(f"[multiseed] ({n_done}/{n_total}) "
                  f"seed={seed} R={replica} {spec.model_id} OK")
        except Exception as e:
            n_fail += 1
            msg = (f"seed={seed} R={replica} {spec.model_id} FAIL: "
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
            for spec in specs:
                _ejecutar(seed, replica, spec)

    elapsed = time.time() - t_start
    print()
    print(f"[multiseed] {len(runs)}/{n_total} corridas exitosas en "
          f"{elapsed/60:.1f} min ({n_fail} fallos)")

    if len(runs) < 2:
        print("[multiseed] insuficiente para análisis (≥2 corridas requeridas).")
        return

    print(f"[multiseed] analizando…")

    # Análisis pareado: fija reference_model como model_a y compara contra
    # cada otro modelo. Con 2 modelos es idéntico al flujo previo.
    model_ids_present = sorted({r.model_label for r in runs})
    if args.reference_model:
        if args.reference_model not in model_ids_present:
            print(f"[multiseed] reference-model {args.reference_model} no está "
                  f"en los runs exitosos {model_ids_present}; usando el primero.")
            ref = model_ids_present[0]
        else:
            ref = args.reference_model
    else:
        ref = specs[0].model_id

    others = [m for m in model_ids_present if m != ref]
    if not others:
        print("[multiseed] solo un modelo con runs exitosos; sin pareo.")
        return

    for m in others:
        sub_dir = out_dir if len(others) == 1 else out_dir / f"vs_{m}"
        print(f"[multiseed] tests pareados: {ref} vs {m} → {sub_dir}")
        paths = analyze(runs, sub_dir, model_a=ref, model_b=m)
        for k, v in paths.items():
            print(f"           - {k}: {v}")


if __name__ == "__main__":
    main()
