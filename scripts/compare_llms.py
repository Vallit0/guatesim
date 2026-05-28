"""Runner comparativo Claude vs OpenAI sobre el mismo mundo.

Garantiza que ambos decisores vean EXACTAMENTE los mismos shocks, el mismo
territorio y el mismo ruido macro, para que la diferencia sea atribuible al
LLM y no al ambiente.

Uso:
    python compare_llms.py --turnos 8 --seed 11
    python compare_llms.py --turnos 8 --claude-modelo claude-haiku-4-5-20251001 \\
                           --openai-modelo gpt-4o-mini
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from guatemala_sim.comparison import CorridaEtiquetada, generar_comparativa
from guatemala_sim.engine import DummyDecisionMaker, DummyMenuDecisionMaker
from guatemala_sim.logging_ import new_run_id
from guatemala_sim.president import ClaudePresidente
from guatemala_sim.president_openai import GPTPresidente, qwen_via_ollama
from guatemala_sim.resilient import ResilientDecisionMaker
from guatemala_sim.runners import correr, nueva_mundo

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turnos", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--claude-modelo", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--openai-modelo", type=str, default="gpt-4o-mini")
    ap.add_argument("--qwen-url", type=str, default=None,
                    help="ej http://192.168.0.17:11434/v1  (Ollama OpenAI-compat)")
    ap.add_argument("--qwen-modelo", type=str, default="qwen2.5:0.5b")
    ap.add_argument("--qwen-key", type=str, default="ollama")
    ap.add_argument("--skip-claude", action="store_true")
    ap.add_argument("--skip-openai", action="store_true")
    ap.add_argument("--incluir-dummy", action="store_true",
                    help="agrega una corrida con DummyDecisionMaker como baseline")
    ap.add_argument("--menu-mode", action="store_true",
                    help=("modo menu-choice: el LLM elige UNO de 5 presupuestos "
                          "predefinidos en vez de componer libremente. Habilita el "
                          "pipeline de IRL bayesiano. JSONL incluye chosen_index."))
    args = ap.parse_args()

    ts = new_run_id()
    outputs: list[tuple[str, Path]] = []

    # Cada corrida con mundo independiente pero seed idéntico → shocks idénticos.
    if not args.skip_claude:
        rng, state, agentes, territory = nueva_mundo(args.seed)
        dm = ClaudePresidente(model=args.claude_modelo)
        p = correr(f"Claude/{args.claude_modelo}", dm, territory, agentes,
                   rng, state, args.turnos, f"{ts}_claude",
                   runs_dir=RUNS_DIR, menu_mode=args.menu_mode)
        outputs.append(("Claude", p))

    if not args.skip_openai:
        rng, state, agentes, territory = nueva_mundo(args.seed)
        dm = GPTPresidente(model=args.openai_modelo)
        p = correr(f"OpenAI/{args.openai_modelo}", dm, territory, agentes,
                   rng, state, args.turnos, f"{ts}_openai",
                   runs_dir=RUNS_DIR, menu_mode=args.menu_mode)
        outputs.append(("OpenAI", p))

    if args.qwen_url:
        if args.menu_mode:
            print("[qwen] WARNING: --menu-mode no soportado para Qwen via Resilient. "
                  "Skipping Qwen en esta corrida.")
        else:
            rng, state, agentes, territory = nueva_mundo(args.seed)
            qwen = qwen_via_ollama(
                model=args.qwen_modelo,
                base_url=args.qwen_url,
                api_key=args.qwen_key,
            )
            # envuelto en resiliente: si Qwen falla, cae a Dummy para no abortar
            dm = ResilientDecisionMaker(
                primario=qwen,
                fallback=DummyDecisionMaker(rng),
                label=f"Qwen/{args.qwen_modelo}",
            )
            p = correr(f"Qwen/{args.qwen_modelo}", dm, territory, agentes,
                       rng, state, args.turnos, f"{ts}_qwen",
                       runs_dir=RUNS_DIR)
            outputs.append((f"Qwen-{args.qwen_modelo}", p))
            print(f"\n[qwen] tasa de fallo: {dm.n_fallos}/{dm.n_llamadas} "
                  f"({dm.tasa_fallo:.1f}%)")
            if dm.fallos:
                print("[qwen] fallos capturados:")
                for fl in dm.fallos[:5]:
                    print(f"  - {fl[:180]}")

    if args.incluir_dummy:
        rng, state, agentes, territory = nueva_mundo(args.seed)
        # En menu-mode el dummy debe usar choose_from_menu; auto-swap.
        dm = DummyMenuDecisionMaker(rng) if args.menu_mode else DummyDecisionMaker(rng)
        p = correr("Dummy/baseline", dm, territory, agentes,
                   rng, state, args.turnos, f"{ts}_dummy",
                   runs_dir=RUNS_DIR, menu_mode=args.menu_mode)
        outputs.append(("Dummy", p))

    if len(outputs) < 2:
        print("\n[comparativa] se necesitan ≥2 corridas para comparar.")
        return

    corridas = [CorridaEtiquetada.from_path(lbl, p) for lbl, p in outputs]
    out_dir = ROOT / "figures" / f"{ts}_compare"
    archivos = generar_comparativa(corridas, out_dir)
    print(f"\n[comparativa] {len(archivos)} archivos en {out_dir}")
    for a in archivos:
        print(f"   - {a.name}")
    print(f"\nreporte: {out_dir / 'reporte.md'}")


if __name__ == "__main__":
    main()
