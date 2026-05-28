"""Demo end-to-end: corre N turnos con agentes + territorio + logging + gráficas.

Uso:
    python demo.py                      # 10 turnos, decisiones dummy
    python demo.py --turnos 15
    python demo.py --claude              # usa Claude (requiere ANTHROPIC_API_KEY)
    python demo.py --claude --turnos 5 --modelo claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# cargar .env si existe (ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from guatemala_sim.agents import (
    AgentesModel,
    CACIF,
    CongresoOposicion,
    PartidoOficialista,
    ProtestaSocial,
)
from guatemala_sim.bootstrap import initial_state
from guatemala_sim.engine import DummyDecisionMaker, run_turn
from guatemala_sim.indicators import (
    coherencia_temporal,
    diversidad_valores,
    resumen_presupuesto,
)
from guatemala_sim.logging_ import (
    JsonlLogger,
    new_run_id,
    print_corrida_resumen,
    print_turn_resumen,
    read_run,
)
from guatemala_sim.plotting import generar_todo
from guatemala_sim.world.territory import Territory


ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--turnos", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--claude", action="store_true", help="usar Claude como tomador de decisiones")
    ap.add_argument("--modelo", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--figuras", action="store_true", default=True, help="generar gráficas al final")
    ap.add_argument("--no-figuras", dest="figuras", action="store_false")
    ap.add_argument("--run-id", type=str, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    state = initial_state()
    territory = Territory.load_default()
    agentes = AgentesModel(
        [PartidoOficialista, CongresoOposicion, CACIF, ProtestaSocial],
        seed=args.seed,
    )

    if args.claude:
        from guatemala_sim.president import ClaudePresidente
        decision_maker = ClaudePresidente(model=args.modelo)
        # el president necesita el resumen territorial cada turno
        decision_maker.territory_provider = lambda: territory.summary().as_dict()
    else:
        decision_maker = DummyDecisionMaker(rng)

    run_id = args.run_id or new_run_id()
    log_path = ROOT / "runs" / f"{run_id}.jsonl"

    print(f"[corrida] run_id={run_id}  turnos={args.turnos}  "
          f"decisor={'Claude/' + args.modelo if args.claude else 'Dummy'}")

    with JsonlLogger(log_path) as lg:
        def hook(record):
            lg.log(record)
            print_turn_resumen(record)
            # propagar eventos de agentes al siguiente contexto del presidente
            extra = getattr(record, "extra", {}) or {}
            if hasattr(decision_maker, "ultimos_eventos"):
                decision_maker.ultimos_eventos = extra.get("eventos_agentes", [])

        for _ in range(args.turnos):
            state, rec = run_turn(
                state,
                decision_maker,
                rng,
                hooks=[hook],
                agentes=agentes,
                territorio=territory,
            )

    print(f"\n[corrida] terminada. log={log_path}")
    records = read_run(log_path)
    print_corrida_resumen(records)

    # --- indicadores transversales ---
    decisiones = [r["decision"] for r in records]
    print("\n[métricas constitucionales del decisor]")
    print(f"  coherencia temporal: {coherencia_temporal(decisiones):.1f}/100")
    print(f"  diversidad valores (entropía): {diversidad_valores(decisiones):.2f}")
    print(f"  presupuesto promedio: {resumen_presupuesto(decisiones)}")

    if args.figuras:
        fig_dir = ROOT / "figures" / run_id
        figs = generar_todo(records, fig_dir)
        print(f"\n[figuras] {len(figs)} archivos en {fig_dir}")
        for f in figs:
            print(f"   - {f.name}")


if __name__ == "__main__":
    main()
