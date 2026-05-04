"""Bayesian Inverse Reinforcement Learning para LLMs como decisores.

Recupera la función de recompensa latente que un LLM parece estar
optimizando, a partir de elecciones observadas sobre un menú discreto de
asignaciones presupuestarias candidatas.

Modelo (Ramachandran & Amir 2007 + Hadfield-Menell et al. 2017):

    R(s, a) = wᵀ φ(s, a)
    P(a | s, w, T) = exp(wᵀ φ(s, a) / T) / Σ_{a' ∈ A(s)} exp(wᵀ φ(s, a') / T)

donde:
    - A(s) es el menú de 5 candidatos generado por `candidates.generate_candidate_menu`
    - φ(s, a) ∈ ℝ⁶ es el vector de outcomes esperados, computado por
      `features.extract_outcome_features` vía Monte Carlo del simulador
    - w ∈ ℝ⁶ son los pesos sobre dimensiones de bienestar (a recuperar)
    - T es la temperatura del sampler (a recuperar conjuntamente con w)

El anchor R(s, a_ref) = 0 se logra restando φ(s, a_ref) de todos los
candidatos antes de entrar al likelihood (a_ref = status_quo_uniforme,
índice 0 del menú).
"""

from .boltzmann import (
    boltzmann_choice_probs,
    boltzmann_log_likelihood,
    boltzmann_log_probs,
    sample_boltzmann_choices,
    subtract_reference,
)
from .candidates import Candidate, REFERENCE_CANDIDATE_INDEX, generate_candidate_menu
from .features import (
    N_OUTCOME_FEATURES,
    OUTCOME_FEATURE_NAMES,
    extract_outcome_features,
)
from .recovery import (
    RecoveryDataset,
    RecoveryMetrics,
    compute_recovery_metrics,
    fit_mle_boltzmann,
    generate_synthetic_dataset,
    run_recovery_sweep,
)
# bayesian_irl importa PyMC de forma diferida en _require_pymc, así que
# importar el módulo es seguro aunque PyMC no esté instalado.
from .bayesian_irl import (
    IRLPosterior,
    fit_bayesian_irl,
    fit_bayesian_irl_point_estimate,
)
from .audit import (
    AlignmentGap,
    audit_llm_alignment,
    encode_prompt_to_w_stated,
)
from .run_parser import ParsedRun, RunFormatError, parse_menu_run

__all__ = [
    # candidates
    "Candidate",
    "generate_candidate_menu",
    "REFERENCE_CANDIDATE_INDEX",
    # features
    "extract_outcome_features",
    "OUTCOME_FEATURE_NAMES",
    "N_OUTCOME_FEATURES",
    # boltzmann
    "subtract_reference",
    "boltzmann_log_probs",
    "boltzmann_choice_probs",
    "boltzmann_log_likelihood",
    "sample_boltzmann_choices",
    # recovery
    "RecoveryDataset",
    "RecoveryMetrics",
    "generate_synthetic_dataset",
    "fit_mle_boltzmann",
    "compute_recovery_metrics",
    "run_recovery_sweep",
    # bayesian IRL
    "IRLPosterior",
    "fit_bayesian_irl",
    "fit_bayesian_irl_point_estimate",
    # IRD audit
    "AlignmentGap",
    "audit_llm_alignment",
    "encode_prompt_to_w_stated",
    # JSONL parser
    "ParsedRun",
    "RunFormatError",
    "parse_menu_run",
]
