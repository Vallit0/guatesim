"""Parser de JSONL de menu-mode → (features, chosen) para IRL bayesiano.

Lee un archivo `runs/<run_id>_<llm>.jsonl` producido por el simulador en
`menu_mode=True` y produce el tensor `(T, K, d)` listo para
`fit_bayesian_irl`, junto con los razonamientos para el chequeo de
consistencia y los estados inicial/final para harm quantification.

Cada turno con `menu_choice` contribuye:
  - φ(s_t, a_k) para k=0..K-1 vía Monte Carlo del simulador
    (`features.extract_outcome_features` con `n_samples` corridas)
  - chosen_t = menu_choice.chosen_index
  - razonamiento_t = decision.razonamiento

`features` se devuelve ya con `subtract_reference(ref_idx=0)` aplicado
para anclar la utilidad: por construcción, `features[:, 0, :] ≈ 0`.

Si el JSONL no fue producido en menu-mode (sin campo `menu_choice`) el
parser levanta `RunFormatError` apuntando a cómo arreglarlo:
`compare_llms.py --menu-mode` o `compare_llms_multiseed.py --menu-mode`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..actions import PresupuestoAnual
from ..logging_ import read_run
from ..state import GuatemalaState
from .boltzmann import subtract_reference
from .features import (
    N_OUTCOME_FEATURES,
    OUTCOME_FEATURE_NAMES,
    extract_outcome_features,
)


class RunFormatError(ValueError):
    """JSONL no compatible con menu-mode (falta campo `menu_choice`)."""


@dataclass(frozen=True)
class ParsedRun:
    """Datos extraídos de un run JSONL listos para alimentar el IRL.

    Convenciones:
        - `features` ya está reference-subtracted (ref_idx=0). El primer
          candidato del menú es el ancla `R(s, a_0) = 0`.
        - `chosen` son los índices elegidos por el LLM en cada turno.
        - El menú es **fijo entre turnos** por contrato (`generate_candidate_menu`
          no depende del state); el parser lo verifica.
    """

    run_path: Path
    features: np.ndarray             # (T, K, d), reference-subtracted
    chosen: np.ndarray               # (T,) ints en [0, K)
    razonamientos: list[str]         # T strings
    candidate_names: tuple[str, ...] # K strings, orden estable
    state_initial: GuatemalaState
    state_final: GuatemalaState
    feature_names: tuple[str, ...] = OUTCOME_FEATURE_NAMES

    @property
    def n_turns(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[2])

    @property
    def label(self) -> str:
        """Etiqueta corta derivada del nombre del archivo (e.g. 'claude').

        Heurística: toma el segmento después del último '_' antes de '.jsonl'.
        Si no parsea, devuelve el stem completo.
        """
        stem = self.run_path.stem
        parts = stem.rsplit("_", 1)
        return parts[-1] if len(parts) == 2 else stem


def parse_menu_run(
    jsonl_path: str | Path,
    feature_seed: int = 0,
    n_samples: int = 20,
) -> ParsedRun:
    """Parsea un JSONL de menu-mode a un `ParsedRun`.

    Args:
        jsonl_path: ruta al archivo JSONL.
        feature_seed: semilla base para Monte Carlo de features. Misma
            semilla → mismos vectores φ. Mantener fijo entre runs para
            que la comparación entre LLMs no se vea contaminada por
            ruido del extractor.
        n_samples: número de simulaciones Monte Carlo por (state_t, a_k).
            Default 20 (≈ σ/√20 ≈ 22% del ruido original del simulador).

    Returns:
        ParsedRun con `features (T, K, d)` reference-subtracted, `chosen
        (T,)`, razonamientos, nombres de candidatos y estados extremos.

    Raises:
        FileNotFoundError: si el path no existe.
        RunFormatError: si el JSONL no tiene records en menu-mode.
        ValueError: si el menú es inconsistente entre turnos o si
            chosen_index está fuera de rango.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL no encontrado: {path}")

    records = read_run(path)
    if not records:
        raise RunFormatError(f"{path} está vacío")

    menu_records = [r for r in records if r.get("menu_choice") is not None]
    if not menu_records:
        raise RunFormatError(
            f"{path} no contiene records en menu-mode (campo `menu_choice` "
            f"ausente en los {len(records)} turnos). Re-ejecutá la corrida "
            f"con `compare_llms.py --menu-mode` (o `compare_llms_multiseed.py "
            f"--menu-mode`) para producir un JSONL compatible con el IRL."
        )
    if len(menu_records) < len(records):
        # Tolerable: tomamos sólo los menu, pero avisamos.
        # No es un error porque podría haber turnos en modo dummy o legacy
        # mezclados en runs viejos.
        pass

    # Verificar que el menú sea idéntico entre turnos (contrato del simulador
    # actual: `generate_candidate_menu()` no depende del state).
    first_menu = menu_records[0]["menu_choice"]["candidates"]
    candidate_names = tuple(c["name"] for c in first_menu)
    K = len(first_menu)
    if K < 2:
        raise RunFormatError(f"Menú con sólo {K} candidato(s); IRL requiere K ≥ 2")

    candidate_presupuestos = [
        PresupuestoAnual.model_validate(c["presupuesto"]) for c in first_menu
    ]

    for t, rec in enumerate(menu_records):
        names_t = tuple(c["name"] for c in rec["menu_choice"]["candidates"])
        if names_t != candidate_names:
            raise ValueError(
                f"Menú inconsistente en turno {t}: nombres {names_t} ≠ "
                f"{candidate_names}. El parser asume menú fijo entre turnos."
            )

    # Extraer state_before, chosen, razonamiento por turno.
    T = len(menu_records)
    chosen = np.empty(T, dtype=int)
    razonamientos: list[str] = []
    states_before: list[GuatemalaState] = []

    for t, rec in enumerate(menu_records):
        ci = int(rec["menu_choice"]["chosen_index"])
        if not 0 <= ci < K:
            raise ValueError(
                f"chosen_index={ci} en turno {t} fuera de rango [0, {K})"
            )
        chosen[t] = ci
        razonamientos.append(rec["decision"]["razonamiento"])
        states_before.append(GuatemalaState.model_validate(rec["state_before"]))

    # Estados extremos para harms: state_before[0] como inicial; state_after
    # del último turno como final.
    state_initial = states_before[0]
    state_final = GuatemalaState.model_validate(menu_records[-1]["state_after"])

    # Calcular φ(s_t, a_k) para cada (turno, candidato).
    # NOTA: feature_seed depende del turno + candidato para evitar
    # correlaciones espurias entre vectores que comparten ruido.
    features = np.zeros((T, K, N_OUTCOME_FEATURES), dtype=float)
    for t, s_t in enumerate(states_before):
        for k, presu in enumerate(candidate_presupuestos):
            seed_tk = int(np.uint64(feature_seed)
                          + np.uint64(t) * np.uint64(7919)
                          + np.uint64(k) * np.uint64(104_729))
            features[t, k, :] = extract_outcome_features(
                s_t, presu, feature_seed=seed_tk, n_samples=n_samples
            )

    features = subtract_reference(features, ref_idx=0)

    return ParsedRun(
        run_path=path,
        features=features,
        chosen=chosen,
        razonamientos=razonamientos,
        candidate_names=candidate_names,
        state_initial=state_initial,
        state_final=state_final,
    )
